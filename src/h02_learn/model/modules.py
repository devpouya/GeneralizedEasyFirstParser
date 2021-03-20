import torch
import torch.nn as nn

from utils import constants

import torch.nn.functional as F

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

    def replace(self, expr):
        out, hidden = self.cell(expr, self.s[-1][1])
        self.s[-1] = (out, hidden)

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

    def forward(self, x,replace=False):
        if replace:
            self.replace(x)
        else:
            self.push(x)

    def __len__(self):
        return len(self.s) - 1


class StackCell():
    def __init__(self, cell, initial_state, initial_hidden, dropout, p_empty_embedding=None):
        super().__init__()
        self.cell = cell
        self.dropout = dropout
        self.s = [initial_state]
        # initial_hidden is a tuple (h,c)
        #self.s = [(initial_state, initial_hidden)]
        #self.s = [(initial_state, initial_hidden)]

        self.empty = None
        if p_empty_embedding is not None:
            self.empty = p_empty_embedding

    def replace(self, expr):
        h, c = self.cell(expr, self.s[-1])
        #self.s[-1][0].detach()
        #self.s[-1][1].detach()
        self.s[-1] = (h,c)
    def put_first(self,expr):
        h, c = self.cell(expr, self.s[0])
        self.s[0] = (h,c)

    def push(self, expr):
        h,c = self.cell(expr, self.s[-1])
        self.s.append((h,c))

    def pop(self,ind=-1):
        return self.s.pop(ind)[1]

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
        self.push(input, first)

        if self.top.prev is None:
            # print("INIT")
            h_0 = torch.zeros((self.num_layers, self.top.weight.shape[1], self.top.weight.shape[2])).to(
                device=constants.device)
            c_0 = torch.zeros((self.num_layers, self.top.weight.shape[1], self.top.weight.shape[2])).to(
                device=constants.device)
            h_0 = nn.init.xavier_normal_(h_0)
            c_0 = nn.init.xavier_normal_(c_0)
            h = (h_0, c_0)
        else:
            h = self.top.prev.hidden

        out, hidden = self.lstm(self.top.weight, h)
        self.top.hidden = hidden
        return out


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

class SoftmaxLegal(nn.Module):
    # __constants__ = ['dim']
    # dim: Optional[int]
    def __init__(self, dim, parser, actions,is_relation=False):
        super(SoftmaxLegal, self).__init__()
        self.dim = dim
        # self.parser = parser
        self.actions = actions
        self.num_actions = actions#len(actions)
        self.indices = self.legal_indices(parser)
        #elf.relate = self.rel_or_not()
        self.is_relation = is_relation
        # self.inds_zero = list(set(range(self.num_actions)).difference(set(self.indices)))

    def legal_indices(self, parser):
        if len(parser.stack) < 2:
            return [0]
        elif len(parser.buffer) < 1:
            return list(range(self.actions))[1:]
        else:

            return list(range(self.actions))

    def rel_or_not(self):
        if self.indices == [0]:
            return False
        else:
            return True

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        if self.is_relation:
            if self.rel_or_not():
                tmp = F.softmax(input[1:], self.dim, _stacklevel=5)
                ret = torch.zeros_like(input)
                ret[1:] = tmp  #.detach().clone()
                return ret
            else:
                tmp = F.softmax(input[0], self.dim, _stacklevel=5)
                ret = torch.zeros_like(input)
                ret[0] = tmp  # .detach().clone()
                return ret
        else:
            tmp = F.softmax(input[self.indices], self.dim, _stacklevel=5)
            # print(tmp)
            ret = torch.zeros_like(input)
            ret[self.indices] = tmp  # .detach().clone()
            return ret  # F.softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)

