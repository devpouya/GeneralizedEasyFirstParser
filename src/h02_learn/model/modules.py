import torch
import torch.nn as nn
import networkx as nx
import matplotlib.pyplot as plt
from utils import constants


class PointerLSTM(nn.Module):
    def __init__(self, id, prev_lstm, input_size, hidden_size, dropout, batch_first, bidirectional=False):
        super().__init__()
        self.is_root = False
        self.prev_lstm = prev_lstm
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstm_cell = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                                       dropout=dropout, batch_first=batch_first, bidirectional=False).to(device=constants.device)

        self.id = id

        if self.prev_lstm is None:
            self.is_root = True

    def forward(self, input):
        return self.lstm_cell(input)

    def set_previous(self, prev):
        self.prev_lstm = prev


class StackLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, dropout, batch_first, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.top_index = 0
        self.top = PointerLSTM(id=0, prev_lstm=None, input_size=self.input_size, hidden_size=self.hidden_size,
                               dropout=self.dropout, batch_first=self.batch_first,
                               bidirectional=self.bidirectional)  # None
        self.lstm_list = [self.top]
        # self.lstm2indx = {}

    def set_top(self, index):
        self.top = self.lstm_list[index]
        self.top_index = index

    def pop(self):
        if self.top.prev_lstm is None:
            # do nothing
            pass
        else:
            self.top = self.top.prev_lstm

            # self.top_index = self.lstm2indx[self.top]
            self.top_index = self.top.id

    def push(self, lstm=None):
        # if top = length of list
        # create and push
        # else, just push (don't add new lstm)
        # lstm has to be a PointerLSTM
        # assert(isinstance(lstm,PointerLSTM))
        if self.top_index >= len(self.lstm_list)-1:
            self.create_and_push()
        else:
            # move top_lstm by one
            self.top_index += 1
            prev = self.top
            self.top = self.lstm_list[self.top_index]
            self.top.prev_lstm = prev

        #lstm.prev_lstm = self.top
        #self.top = lstm

        # self.top_index = self.lstm2indx[self.top]
        # self.lstm_list.append(lstm)
        # self.lstm2indx[lstm] = len(self.lstm_list) - 1

    def create_and_push(self, lstm=None):
        if lstm is None:
            if len(self.lstm_list) > 0:
                lstm = PointerLSTM(id=len(self.lstm_list), prev_lstm=self.top, input_size=self.input_size
                                   , hidden_size=self.hidden_size, dropout=self.dropout,
                                   batch_first=self.batch_first, bidirectional=self.bidirectional)
            else:
                lstm = PointerLSTM(id=0, prev_lstm=None, input_size=self.input_size, hidden_size=self.hidden_size,
                                   dropout=self.dropout, batch_first=self.batch_first,
                                   bidirectional=self.bidirectional)

        lstm.prev_lstm = self.top
        self.top = lstm

        self.top_index = self.top.id  # self.lstm2indx[self.top]
        self.lstm_list.append(lstm)
        # self.lstm2indx[lstm] = len(self.lstm_list) - 1

    def forward(self, x):

        current_branch, _ = self.get_branch(self.top)
        # fix this
        if len(x.shape) < 2:
            x = x.reshape(1,1,x.shape[0])
        else:
            x = x.reshape(x.shape[0], 1, x.shape[1])
        #print(x.shape)
        # output = torch.empty()
        # print("output shape {}".format(output.shape))

        hidden, _ = current_branch[0](x)
        # print(hidden.shape)
        for lstm in current_branch[1:]:
            hidden, _ = lstm(hidden)
            # print(hidden.shape)

        #print("output shape {}".format(hidden.shape))
        return hidden

    def stack_summary(self, x):
        out, hidden = self.lstm_list[self.top_index](x)
        return out

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
        for node in self.lstm_list:
            _, edge_list = self.get_branch(node)
            edges.update(edge for edge in edge_list)

        edges = list(edges)
        G.add_edges_from(edges)
        nx.draw(G)
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
