import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from utils import constants


# CONSTANTS AND NAMES


# Classes for basic transition based data structures

# Holds the arcs created during parsing
class Arcs:

    def __init__(self):
        # a list of tuples (u,v) indicating an arc u --> v
        self.arcs = []

    def add_arc(self, i, j):
        # adds arc i --> j
        self.arcs.append((i, j))

    def has_incoming(self, v):
        # check if node v has an incoming arc
        for (src, dst) in self.arcs:
            if dst == v:
                return True
        return False


# Basic stack used for parsing
class Stack:

    def __init__(self):
        # stack is implemented as a list
        # each element is a tuple (TENSOR A, INT INDEX)
        self.stack = []

    def push(self, x):
        # add to the end of the list (slow)
        self.stack.append(x)

    def pop(self):
        # remove and return last element
        return self.stack.pop(-1)

    def pop_second(self):
        # remove and return second to last element
        return self.stack.pop(-2)

    def pop_third(self):
        return self.stack.pop(-3)

    def second(self):
        # return the second element
        return self.stack[-2]

    def third(self):
        return self.stack[-3]

    def set_second(self, x):
        # set the second element's tensor to x, keeping the index
        (_, ind) = self.stack[-2]
        self.stack[-2] = (x, ind)

    def top(self):
        # return the top of the stack
        return self.stack[-1]

    def set_top(self, x):
        # set the top element's tensor to x, keeping the index
        (_, ind) = self.stack[-1]
        self.stack[-1] = (x, ind)

    def get_len(self):
        # lol get the length not necessary lol
        return len(self.stack)


# Buffer holding the sentence to be parser
class Buffer:

    def __init__(self, sentence):
        # initialized with the words in the sentence
        # each element is a tuple (TENSOR WORD, INT INDEX)
        s = sentence.clone().detach()
        self.buffer = [(word, i) for i, word in enumerate(s)]

    def pop_left(self):
        # pop left (first) element
        return self.buffer.pop(0)

    def put_left(self, x):
        (_, ind) = self.buffer[0]
        self.buffer[0] = (x, ind)

    def left(self):
        # return first element
        return self.buffer[0]

    def get_len(self):
        # lol why did I implement this?
        return len(self.buffer)


# arc-standard shift reduce parser
class ShiftReduceParser():

    def __init__(self, sentence, embedding_size, transition_system):
        # data structures
        # data structures are buggyyy
        # regular lists do fine for now
        # self.stack = Stack()
        init_sent = []
        self.buffer = []
        #self.buffer.append((torch.rand_like(sentence[0]),0))
        for i, word in enumerate(sentence):
            self.buffer.append((word.clone(), i))
        self.stack = []
        self.arcs = []
        # self.buffer = Buffer(sentence)
        # self.arcs = Arcs()

        # hold the action history (embedding) and names (string)
        self.action_history_names = []
        self.actions_probs = torch.zeros((1, len(transition_system[0]))).to(device=constants.device)
        self.oracle_action_history = []

        # sentence to parse
        self.sentence = sentence
        self.learned_repr = sentence
        self.embedding_size = embedding_size

        # used for learning representation for partial parse trees
        #self.linear = nn.Linear(7 * embedding_size+16, 3 * embedding_size).to(device=constants.device)
        #torch.nn.init.xavier_uniform_(self.linear.weight)

        #self.tanh = nn.Tanh().to(device=constants.device)

    def rec_subtree(self, top, second, rel_embed, rnn,linear):
        reprs = torch.cat([top[0], second[0], rel_embed.reshape(100)],
                          dim=-1)
        seq = torch.stack([top[0],second[0]])
        #print(seq.shape)
        #print(reprs.shape)
        c,_ = rnn(seq.unsqueeze(1))
        c = c.squeeze(1)
        c = torch.cat([c[0,:],c[1,:],rel_embed.reshape(self.embedding_size)],dim=-1)
        c = nn.Tanh()(linear(c))
        (_, ind) = self.stack[-1]
        self.stack[-1] = (c, ind)
        return c

    def subtree_rep(self, top, second, rel_embed,linear):

        reprs = torch.cat([top[0], second[0], rel_embed.reshape(100)],
                          dim=-1)

        c = nn.Tanh()(linear(reprs))
        (_, ind) = self.stack[-1]
        self.stack[-1] = (c, ind)
        return c

    def subtree_rep_hybrid(self, top, second, rel_embed,act_embed,linear):

        reprs = torch.cat([top[0], second[0], rel_embed.reshape(60)],
                          dim=-1)

        c = nn.Tanh()(linear[1](F.relu(linear[0](reprs))))

        (_, ind) = self.stack[-1]
        self.stack[-1] = (c, ind)
        return c

    def shift(self):
        item = self.buffer.pop(0)
        self.stack.append(item)
        self.action_history_names.append(constants.shift)
        return item[0]

    def reduce_l(self, rel,rel_embed,rnn,linear):
        top = self.stack[-1]
        second = self.stack[-2]
        #left = self.buffer[0]
        self.stack.pop(-2)
        self.arcs.append((top[1], second[1], rel))
        self.action_history_names.append(constants.reduce_l)
        c = self.subtree_rep(top, second,rel_embed,linear)
        return c

    def reduce_r(self, rel,rel_embed,rnn,linear):
        second = self.stack[-2]
        top = self.stack[-1]
        self.arcs.append((second[1], top[1], rel))
        self.action_history_names.append(constants.reduce_r)
        self.stack.pop(-1)
        #self.buffer[0] = top
        c = self.subtree_rep(second, top,rel_embed,linear)
        return c

    def reduce(self):
        # stack_top = self.stack.top()
        # if self.arcs.has_incoming(stack_top):
        self.stack.pop(-1)
        self.action_history_names.append(constants.reduce)

    def left_arc_eager(self, act_embed, rel,rel_embed,linear):
        top = self.stack[-1]
        left = self.buffer[0]
        # if not self.arcs.has_incoming(stack_top):
        self.stack.pop(-1)
        self.arcs.append((left[1], top[1],rel))
        self.action_history_names.append(constants.left_arc_eager)
        c = self.subtree_rep(left, top, rel_embed,act_embed,linear)
        return c

    def right_arc_eager(self, act_embed, rel,rel_embed,linear):
        top = self.stack[-1]
        left = self.buffer[0]
        self.stack.append(left)
        self.arcs.append((top[1], left[1],rel))
        self.action_history_names.append(constants.right_arc_eager)
        c = self.subtree_rep(top, left, rel_embed,act_embed,linear)
        self.buffer.pop(0)

        return c

    def left_arc_hybrid(self, act_emb,rel,rel_embed,linear):
        stack_top = self.stack[-1]
        buffer_first = self.buffer[0]
        self.arcs.append((buffer_first[1], stack_top[1],rel))
        self.action_history_names.append(constants.left_arc_hybrid)
        c = self.subtree_rep_hybrid(buffer_first, stack_top, rel_embed,act_emb,linear)
        self.stack.pop(-1)

        return c

    def reduce_r_hybrid(self, act_emb, rel, rel_embed, linear):
        top = self.stack[-1]
        second = self.stack[-2]
        self.arcs.append((second[1],top[1],rel))
        self.action_history_names.append(constants.reduce_r)
        self.stack.pop(-1)
        c = self.subtree_rep_hybrid(second,top,rel_embed,act_emb,linear)
        return c

    def left_arc_second(self, act_emb):
        third = self.stack.pop(-3)
        top = self.stack[-1]
        self.arcs.append((top[1], third[1]))
        self.action_history_names.append(constants.left_arc_2)
        c = self.subtree_rep(top, third, act_emb)
        return c

    def right_arc_second(self, act_emb):
        # third = self.stack.third()
        third = self.stack[-3]
        top = self.stack.pop(-1)
        self.arcs.append((third[1], top[1]))
        self.action_history_names.append(constants.right_arc_2)
        c = self.subtree_rep(third, top, act_emb)
        return c

    def is_parse_complete(self, special_op=False):
        if special_op:
            return True
        # buffer has to be empty and stack only contains ROOT item
        buffer_empty = len(self.buffer) == 0
        stack_empty = len(self.stack) == 0
        complete = buffer_empty and stack_empty
        return complete

    def heads_from_arcs(self):
        heads = [0] * (len(self.sentence)+1)
        rels = [0] * (len(self.sentence)+1)
        #self.arcs.append((0,0,1))
        #print(self.arcs)
        tmp = self.arcs.copy()
        tmp.append((0,0,1))
        for i in range(len(self.sentence)):
            for (u, v, r) in tmp:
                if v == i:
                    heads[i+1] = u
                    rels[i+1] = r
        heads.pop(0)
        rels.pop(0)

        return torch.tensor(heads).to(device=constants.device), torch.tensor(rels).to(device=constants.device)

    def set_oracle_action(self, act):
        self.oracle_action_history.append(act)

    def set_action_probs(self, probs):

        self.actions_probs = torch.cat([self.actions_probs, probs.transpose(1, 0)], dim=0)

