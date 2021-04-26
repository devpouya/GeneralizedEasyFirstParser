import torch
import torch.nn as nn

from utils import constants
import torch.nn.functional as F
import heapq
from termcolor import colored


def has_head(node, arcs):
    for (u, v, _) in arcs:
        if v == node:
            return True
    return False


class Item(object):

    def __init__(self, i, j, h, item_l, item_r):
        self.i, self.j, self.h = i, j, h
        self.l = item_l
        self.r = item_r
        self.subtrees = {}
        self.arcs = []
        self.vector_rep = None
        self.score = torch.tensor([0]).to(device=constants.device)
        self.rel = None

    def set_vector_rep(self, x):
        self.vector_rep = x
        return self
    def update_score(self, s):
        self.score = s
    def add_rel(self, rel):
        self.rel = rel
        return self
    def __str__(self):
        return "Item:\t" + str((self.i, self.j, self.h))

    def __repr__(self):
        return "Item:\t" + str((self.i, self.j, self.h))

    def __hash__(self):
        return hash((self.i, self.j, self.h))


class Chart(object):

    def __init__(self):
        self.chart = {}

    def __getitem__(self, key):
        if isinstance(key, Item):
            return self.chart[(key.i, key.j, key.h)]
        else:
            return self.chart[key]

    def __setitem__(self, key, item):
        if isinstance(key, Item):
            self.chart[(key.i, key.j, key.h)] = item
        else:
            self.chart[key] = item

    def __iter__(self):
        return iter(self.chart)

    def __delitem__(self, key):
        del self.chart[key]


class Agenda(object):

    def __init__(self):
        self.locator = {}
        self.Q = []

    def pop(self):
        """ super slow ! """
        tmp = self.Q[0]
        self.Q = self.Q[1:]
        return tmp

    def empty(self):
        return True if len(self.Q) == 0 else False

    def locate(self, item):
        pass

    def percolate(self):
        pass

    def __len__(self):
        return len(self.Q)

    def __iter__(self):
        return iter(self.locator)

    def __getitem__(self, key):
        return self.locator[key]

    def __setitem__(self, key, item):
        """ super slow ! """
        if (item.i, item.j, item.h) not in self:
            self.locator[(item.i, item.j, item.h)] = item
            self.Q.append(item)
            self.Q.sort(key=lambda x: -x.w)
        else:
            self.locator[key] = item
            self.Q.sort(key=lambda x: -x.w)


class PendingRNN():
    def __init__(self, cell, initial_state, initial_hidden, dropout, p_empty_embedding=None):
        super().__init__()
        self.cell = cell
        self.dropout = dropout
        self.s = [initial_state]
        # initial_hidden is a tuple (h,c)
        # self.s = [(initial_state, initial_hidden)]
        # self.s = [(initial_state, initial_hidden)]

        self.empty = None
        if p_empty_embedding is not None:
            self.empty = p_empty_embedding

    def replace(self, expr, ind):
        out, hidden = self.cell(expr, self.s[ind])
        self.s[ind] = (out, hidden)
    def put_first(self, expr):
        h, c = self.cell(expr, self.s[0])
        self.s[0] = (h, c)

    def push(self, expr):
        h, c = self.cell(expr, self.s[-1])
        self.s.append((h, c))

    def pop(self, ind=-1):
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

    def forward(self, x, replace=False):
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
        # self.s = [(initial_state, initial_hidden)]
        # self.s = [(initial_state, initial_hidden)]

        self.empty = None
        if p_empty_embedding is not None:
            self.empty = p_empty_embedding

    def replace(self, expr):
        h, c = self.cell(expr, self.s[-1])
        # self.s[-1][0].detach()
        # self.s[-1][1].detach()
        self.s[-1] = (h, c)

    def put_first(self, expr):
        h, c = self.cell(expr, self.s[0])
        self.s[0] = (h, c)

    def push(self, expr):
        h, c = self.cell(expr, self.s[-1])
        self.s.append((h, c))

    def pop(self, ind=-1):
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



class BiaffineChart(nn.Module):
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

    def forward(self, x_l, x_r, legal_pairs, hypergraph,s_ind):
        # x shape [batch, length_l, length_r]
        x = torch.matmul(x_l, self.matrix)
        x = torch.bmm(x, x_r.transpose(1, 2)) + self.bias

        # x shape [batch, length_l, 1] and [batch, 1, length_r]
        x += self.linear_l(x_l) + self.linear_r(x_r).transpose(1, 2)
        x = nn.ReLU()(x)
        x = x.squeeze(0)
        scores = torch.zeros_like(x)
        # scores *= -float('inf')
        legal_pairs = list(set(legal_pairs))
        for (u, v) in legal_pairs:
            item = hypergraph.locator[(u,v)]
            if item.l in hypergraph.bucket or item.r in hypergraph.bucket:
                #print(colored("PRUNE","red"))
                continue
            #scores[s_ind[u], s_ind[v]] = x[s_ind[u], s_ind[v]]
            scores[u, v] = x[u, v]

            item2 = hypergraph.locator[(v, u)]
            if item2.l in hypergraph.bucket or item2.r in hypergraph.bucket:
                #print(colored("PRUNE","red"))
                continue
            #scores[s_ind[v], s_ind[u]] = x[s_ind[v], s_ind[u]]
            scores[v, u] = x[v, u]
        return scores, x


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
    def __init__(self, dim, parser, num_actions, num_rels, transition_system):
        super(SoftmaxLegal, self).__init__()
        self.dim = dim
        self.num_actions = num_actions
        self.all_ind = list(range(num_actions))
        self.num_rels = num_rels
        self.transition_system = transition_system
        if transition_system == constants.arc_standard:
            legal_indices = self.legal_indices_arc_standard
        elif transition_system == constants.arc_eager:
            legal_indices = self.legal_indices_arc_eager
        elif transition_system == constants.hybrid:
            legal_indices = self.legal_indices_hybrid
        elif transition_system == constants.mh4:
            legal_indices = self.legal_indices_mh4

        self.indices = legal_indices(parser)

    def legal_indices_arc_standard(self, parser):
        if len(parser.stack) < 2:
            return [0]
        elif len(parser.buffer) < 1:
            return list(range(self.num_actions))[1:]  # [1, 2]
        else:

            return list(range(self.num_actions))  # [0, 1, 2]

    def legal_indices_arc_eager(self, parser):
        if len(parser.stack) < 1:
            # can only shift
            return [0]

        elif len(parser.buffer) < 1:
            return [1]
        else:
            if not has_head(parser.stack[-1], parser.arcs):
                # can left, can't reduce
                return [0] + self.all_ind[1:]
            else:
                # can't left, can reduce
                return [0] + self.all_ind[1 + self.num_rels:]

    def legal_indices_hybrid(self, parser):
        if len(parser.stack) < 1:
            # can only shift
            return [0]
        elif len(parser.stack) == 1 and len(parser.buffer) > 0:
            # can't right reduce
            return self.all_ind[:self.num_rels + 1]
        elif len(parser.buffer) > 0:
            return self.all_ind
        else:
            return self.all_ind[self.num_rels + 1:]

    def legal_indices_mh4(self, parser):
        if len(parser.stack) < 1:
            return [0]
        elif len(parser.buffer) < 1:
            return self.all_ind[2 * self.num_rels + 1:5 * self.num_rels + 1] + self.all_ind[6 * self.num_rels + 1:]
        elif 3 > len(parser.stack) >= 2:
            return self.all_ind[:self.num_rels * 4 + 1] + self.all_ind[self.num_rels * 5 + 1:self.num_rels * 6 + 1]
        elif len(parser.stack) < 2 and len(parser.buffer) >= 1:
            return self.all_ind[1:self.num_rels]
        elif len(parser.buffer) >= 1 and len(parser.stack) >= 3:
            return self.all_ind

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        tmp = F.softmax(input[:, self.indices], self.dim, _stacklevel=5)
        ret = torch.zeros_like(input)
        ret[:, self.indices] = tmp  # .detach().clone()
        return ret

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)


class SoftmaxActions(nn.Module):

    def __init__(self, dim, parser, transition_system, temperature, aggresive=True):
        super(SoftmaxActions, self).__init__()
        self.dim = dim

        self.transition_system = transition_system
        if transition_system == constants.arc_standard:
            legal_indices = self.legal_indices_arc_standard
        elif transition_system == constants.arc_eager:
            legal_indices = self.legal_indices_arc_eager
        elif transition_system == constants.hybrid:
            legal_indices = self.legal_indices_hybrid
        else:
            legal_indices = self.legal_indices_mh4

        self.indices = legal_indices(parser)


        if aggresive:
            self.temp = len(self.indices)
        else:
            self.temp = temperature

    def legal_indices_arc_standard(self, parser):
        if len(parser.stack) < 2:
            return [0]
        elif len(parser.buffer) < 1:
            return [1, 2]
        else:
            return [0, 1, 2]

    def legal_indices_arc_eager(self, parser):
        if len(parser.stack) < 1:
            # can only shift
            return [0]

        elif len(parser.buffer) < 1:
            return [3]
        else:
            if not has_head(parser.stack[-1], parser.arcs):
                # can left, can't reduce
                return [0, 1, 2]
            else:
                # can't left, can reduce
                return [0, 2, 3]

    def legal_indices_hybrid(self, parser):
        if len(parser.stack) < 1:
            # can only shift
            return [0]
        elif len(parser.stack) == 1 and len(parser.buffer) > 0:
            # can't right reduce
            return [0, 1]
        elif len(parser.buffer) > 0:
            return [0, 1, 2]
        else:
            return [2]

    def legal_indices_mh4(self, parser):
        # la: stack >= 1 and buffer >= 1 (1)
        # ra: stack >= 2 (2)
        # la': stack >= 2 (3)
        # ra': stack >= 3 (4)
        # la2: stack >= 2 and buffer >= 1 (5)
        # ra2: stack >= 3 (6)
        sgt3 = len(parser.stack) >= 3
        sgt2 = len(parser.stack) >= 2
        bgt1 = len(parser.buffer) >= 1
        sgt1 = len(parser.stack) >= 1
        legal_ind = []
        if sgt3:
            legal_ind.append(4)
            legal_ind.append(6)
        if sgt2:
            legal_ind.append(2)
            legal_ind.append(3)
        if sgt2 and bgt1:
            legal_ind.append(5)
        if sgt1 and bgt1:
            legal_ind.append(1)
        if bgt1:
            legal_ind.append(0)
        return legal_ind

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input):
        tmp = F.softmax(input[:, self.indices], self.dim, _stacklevel=5)
        tmp = torch.div(tmp, 2 ** 8)
        ret = torch.zeros_like(input)
        ret[:, self.indices] = tmp  # .detach().clone()
        return ret

    def extra_repr(self):
        return 'dim={dim}'.format(dim=self.dim)
