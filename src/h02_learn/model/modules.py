import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from utils import constants

import numpy as np


class PointerLSTM(nn.Module):
    def __init__(self, id, prev_lstm, input_size, hidden_size, dropout, batch_first, bidirectional=False):
        super().__init__()
        self.is_top = False
        self.is_root = False
        self.is_final = True
        self.prev_lstm = prev_lstm
        self.next_lstm = None
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstm_cell = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                                       dropout=dropout, batch_first=batch_first, bidirectional=False).to(
            device=constants.device)

        self.id = id

        if self.prev_lstm is None:
            self.is_root = True

    def forward(self, input):
        return self.lstm_cell(input)

    def set_previous(self, prev):
        self.prev_lstm = prev
        self.is_root = False

    def set_next(self, next):
        self.next_lstm = next
        self.is_final = False


class HiddenOutput():
    def __init__(self, weight):
        self.weight = weight
        self.hidden_weight = None
        self.prev = None
        self.next = []
        self.is_top = False
        self.is_root = False


# adapted from stack-lstm-ner (https://github.com/clab/stack-lstm-ner)
class StackRNN(nn.Module):
    def __init__(self, cell, initial_state, initial_hidden, dropout, p_empty_embedding=None):
        super().__init__()
        self.cell = cell
        self.dropout = dropout
        # self.s = [(initial_state, None)]
        self.s = [(initial_state, initial_hidden)]

        self.empty = None
        if p_empty_embedding is not None:
            self.empty = p_empty_embedding

    def push_first(self, expr, stack_rep):
        expr = expr.unsqueeze(0).unsqueeze(1)

        out, hidden = self.cell(expr, stack_rep[1])
        self.pop()
        items = []
        while self.__len__() > 0:
            items.append(self.pop(0))
        self.s.append((out, hidden))
        # items = (out,hidden) + items
        for i in items:
            self.push(i[0].unsqueeze(0))
        # self.s.append((out, hidden))  # +self.s.pop(0)

    def push(self, expr, extra=None):

        out, hidden = self.cell(expr, self.s[-1][1])
        self.s.append((out, hidden))

    def pop(self, ind=-1):
        if ind == 0:
            ind += 1
        return self.s.pop(ind)[0]  # [0]

    def embedding(self):
        return self.s[-1][0] if len(self.s) > 1 else self.empty

    def back_to_init(self):
        while self.__len__() > 0:
            self.pop()

    def clear(self):
        self.s.reverse()
        self.back_to_init()

    def __len__(self):
        return len(self.s) - 1

class StackLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, batch_size, batch_first, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.batch_size = batch_size
        self.num_layers = 2
        self.bidirectional = bidirectional

        self.root = None
        self.top = None


        self.curr_len = 0

        # A list of HiddenOutput
        self.hidden_list = []

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layers,
                                  dropout=dropout, batch_first=batch_first, bidirectional=bidirectional) \
            .to(device=constants.device)

    def push(self, x, first=False):
        item = HiddenOutput(x)
        if first:
            self.top = item
            item.is_root = True

        else:
            item.prev = self.top
            self.top.next.append(x)
            self.top = item
        self.curr_len += 1

    def pop(self):
        try:
            self.top = self.top.prev
            # self.curr_len -= 1
        except:
            self.top = self.top
            # self.curr_len -= 1

    def forward(self, input, first=False):
        self.push(input,first)

        if self.top.prev is None:
            #print("INIT")
            h_0 = torch.zeros((self.num_layers, self.top.weight.shape[1], self.top.weight.shape[2])).to(device=constants.device)
            c_0 = torch.zeros((self.num_layers, self.top.weight.shape[1], self.top.weight.shape[2])).to(device=constants.device)
            h_0 = nn.init.xavier_normal_(h_0)
            c_0 = nn.init.xavier_normal_(c_0)
            h = (h_0,c_0)
        else:
            h = self.top.prev.hidden

        out, hidden = self.lstm(self.top.weight,h)
        self.top.hidden = hidden
        return out

class GrowingStackLSTM(nn.Module):
    """Useless lol"""
    def __init__(self, input_size, hidden_size, dropout, batch_first, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.top_index = 0
        self.root = None
        self.top = None  # PointerLSTM(id=0, prev_lstm=None, input_size=self.input_size, hidden_size=self.hidden_size,
        #            dropout=self.dropout, batch_first=self.batch_first,
        #            bidirectional=self.bidirectional)  # None
        self.lstm_list = []  # [self.top]
        # self.lstm2indx = {}

    def set_top(self, index):
        self.top = self.lstm_list[index]
        self.top.is_top = True

    def pop(self):
        if self.top.prev_lstm is None:
            # do nothing
            pass
        else:
            # self.top.is_final = False
            self.top.is_top = False
            self.top = self.top.prev_lstm
            self.top.is_final = False
            self.top.is_top = True
            # self.top.is_final = True

            # self.top_index = self.lstm2indx[self.top]
            # self.top_index = self.top.id

    def push(self, lstm=None, initialize=False):
        # if top = length of list
        # create and push
        # else, just push (don't add new lstm)
        # lstm has to be a PointerLSTM
        # assert(isinstance(lstm,PointerLSTM))

        # if self.top.is_final: #self.top_index >= len(self.lstm_list):
        # if initialize:
        #    self.top = self.root

        self.create_and_push()
        # else:
        #    #self.create_and_push()
        #    # move top_lstm by one
        #    self.top.is_top = False
        #    self.top = self.top.next_lstm
        #    self.top.is_top = True
        #    #self.top_index =
        #    #prev = self.top
        #    #self.top = self.top.next_lstm #self.lstm_list[self.top_index]
        #    #self.top.prev_lstm = prev

        # lstm.prev_lstm = self.top
        # self.top = lstm

        # self.top_index = self.lstm2indx[self.top]
        # self.lstm_list.append(lstm)
        # self.lstm2indx[lstm] = len(self.lstm_list) - 1

    def create_and_push(self, x=None, lstm=None, make_root=False):
        if lstm is None:
            # if len(self.lstm_list) > 0:
            lstm = PointerLSTM(id=len(self.lstm_list), prev_lstm=self.top, input_size=self.input_size
                               , hidden_size=self.hidden_size, dropout=self.dropout,
                               batch_first=self.batch_first, bidirectional=self.bidirectional)
            # else:
            #    lstm = PointerLSTM(id=0, prev_lstm=None, input_size=self.input_size, hidden_size=self.hidden_size,
            #                       dropout=self.dropout, batch_first=self.batch_first,
            #
            #                       bidirectional=self.bidirectional)
        lstm.is_top = True
        lstm.is_final = True
        # if make_root:
        #    lstm.is_root = True
        #    self.top = lstm

        if lstm.id == 0:
            lstm.is_root = True
            if self.top is not None:
                lstm(self.stack_summary(x))
            self.top = lstm
            self.root = lstm
        else:
            lstm.prev_lstm = self.top
            self.top.next_lstm = lstm
            self.top.is_top = False
            self.top.is_final = False
            self.top = lstm

        self.lstm_list.append(lstm)

    def forward(self, x):

        current_branch, _ = self.get_branch(self.top)
        # fix this
        if len(x.shape) < 2:
            x = x.reshape(1, 1, x.shape[0])
        else:
            x = x.reshape(x.shape[0], 1, x.shape[1])
        # print(x.shape)
        # output = torch.empty()
        # print("output shape {}".format(output.shape))

        hidden, _ = current_branch[0](x)
        # print(hidden.shape)
        for lstm in current_branch[1:]:
            hidden, _ = lstm(hidden)
            # print(hidden.shape)

        # print("output shape {}".format(hidden.shape))
        return hidden

    def stack_summary(self, x):
        out, hidden = self.top(x)
        return out, hidden

    def reset(self):
        self.lstm_list = []

    def get_branch(self, start_node):
        current_branch = []
        # current_branch.append(self.lstm_list[self.top_index])
        curr_lstm = start_node
        edges = []
        while not curr_lstm.is_root:
            current_branch.append(curr_lstm)
            edges.append((curr_lstm.prev_lstm.id, curr_lstm.id))
            curr_lstm = curr_lstm.prev_lstm

        # add root
        current_branch.append(curr_lstm)
        current_branch = current_branch[::-1]
        return current_branch, edges

    def plot_structure(self, show=False, save=False, path=None):

        G = nx.DiGraph()
        G.add_nodes_from(range(len(self.lstm_list)))

        edges = set()
        color_set = []
        for node in G:
            if self.lstm_list[node].is_top:
                color_set.append("red")
            elif self.lstm_list[node].is_root:
                color_set.append("green")
            else:
                color_set.append("blue")
        for node in self.lstm_list:
            _, edge_list = self.get_branch(node)
            edges.update(edge for edge in edge_list)

        edges = list(edges)
        G.add_edges_from(edges)
        nx.draw(G, node_color=color_set)
        if save:
            plt.savefig(path)
        if show:
            plt.show()


class Biaffine(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, dim_left, dim_right):
        super().__init__()
        self.dim_left = dim_left
        self.dim_right = dim_right

        self.matrix = nn.Parameter(torch.Tensor(dim_left, dim_right))
        self.bias = nn.Parameter(torch.Tensor(1))

        self.linear_l = nn.Linear(dim_left, 1)
        self.linear_r = nn.Linear(dim_right, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.bias, 0.)
        nn.init.xavier_uniform_(self.matrix)

    def forward(self, x_l, x_r):
        # x shape [batch, length_l, length_r]
        x = torch.matmul(x_l, self.matrix)
        x = torch.bmm(x, x_r.transpose(1, 2)) + self.bias

        # x shape [batch, length_l, 1] and [batch, 1, length_r]
        x += self.linear_l(x_l) + self.linear_r(x_r).transpose(1, 2)
        return x


class Bilinear(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, dim_left, dim_right, dim_out):
        super().__init__()
        self.dim_left = dim_left
        self.dim_right = dim_right
        self.dim_out = dim_out

        self.bilinear = nn.Bilinear(dim_left, dim_right, dim_out)
        self.linear_l = nn.Linear(dim_left, dim_out)
        self.linear_r = nn.Linear(dim_right, dim_out)

    def forward(self, x_l, x_r):
        # x shape [batch, length, dim_out]
        x = self.bilinear(x_l, x_r)

        # x shape [batch, length, dim_out] and [batch, length, dim_out]
        x += self.linear_l(x_l) + self.linear_r(x_r)
        return x
