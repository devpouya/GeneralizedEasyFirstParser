import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from utils import constants

# CONSTANTS AND NAMES

shift = "SHIFT"

reduce_l = "REDUCE_L"

reduce_r = "REDUCE_R"

reduce = "REDUCE"

left_arc_eager = "LEFT_ARC_EAGER"

right_arc_eager = "RIGHT_ARC_EAGER"

left_arc_hybrid = "LEFT_ARC_H"

arc_standard = ([shift, reduce_l, reduce_r], range(3))  # {shift: 0, reduce_l: 1, reduce_r: 2}
# arc_standard_actions = {0: fshift, 1: freduce_l, 2: freduce_r}

arc_eager = ([shift, left_arc_eager, right_arc_eager, reduce],
             range(4))  # {shift: 0, left_arc_eager: 1, right_arc_eager: 2, reduce: 3}
# arc_eager_actions = {0: fshift, 1: fleft_arc_eager, 2: fright_arc_eager, 3: freduce}

hybrid = ([shift, left_arc_hybrid, reduce_r], range(3))  # {shift: 0, left_arc_hybrid: 1, reduce_r: 2}


# hybrid_actions = {0: fshift, 1: fleft_arc, 2: freduce_r}


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

    def second(self):
        # return the second element
        return self.stack[-2]

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
        self.buffer = [(word, i) for i, word in enumerate(sentence)]

    def pop_left(self):
        # pop left (first) element
        return self.buffer.pop(0)

    def left(self):
        # return first element
        return self.buffer[0]

    def get_len(self):
        # lol why did I implement this?
        return len(self.buffer)


# arc-standard shift reduce parser
class ShiftReduceParser():

    def __init__(self, sentence, embedding_size):
        # data structures
        self.stack = Stack()
        self.buffer = Buffer(sentence)
        self.arcs = Arcs()

        # hold the action history (embedding) and names (string)
        self.action_history = []
        self.action_history_names = []

        # sentence to parse
        self.sentence = sentence
        self.learned_repr = sentence
        self.embedding_size = embedding_size

        # used to reconstruct heads and parse tree
        self.ind2continous = {i: vec for (vec, i) in self.buffer.buffer}

        # used for learning representation for partial parse trees
        self.linear = nn.Linear(5 * embedding_size, 2 * embedding_size).to(device=constants.device)
        self.tanh = nn.Tanh().to(device=constants.device)

    def get_stack_content(self):
        return [item[0] for item in self.stack.stack]

    def get_buffer_content(self):
        return [item[0] for item in self.buffer.buffer]

    def subtree_rep(self, top, second, act_emb):
        reprs = torch.cat([top, second, act_emb.reshape(self.embedding_size)],
                          dim=-1)
        c = self.tanh(self.linear(reprs))
        self.stack.set_top(c)

    def shift(self, act_emb):
        item = self.buffer.pop_left()
        self.stack.push(item)
        self.action_history_names.append(shift)
        self.action_history.append(act_emb)

    def reduce_l(self, act_emb):
        item_top = self.stack.top()
        item_second = self.stack.pop_second()
        self.arcs.add_arc(item_top[1], item_second[1])
        self.action_history_names.append(reduce_l)
        self.action_history.append(act_emb)

        # compute build representation and use this from now on
        self.subtree_rep(item_top[0], item_second[0], act_emb)

    def reduce_r(self, act_emb):
        second_item = self.stack.second()
        top_item = self.stack.pop()
        self.arcs.add_arc(second_item[1], top_item[1])
        self.action_history_names.append(reduce_r)
        self.action_history.append(act_emb)

        # compute build representation and use this from now on
        self.subtree_rep(second_item[0], top_item[0], act_emb)

    def reduce(self, act_emb):
        # stack_top = self.stack.top()
        # if self.arcs.has_incoming(stack_top):
        self.stack.pop()
        self.action_history.append(act_emb)
        self.action_history_names.append(reduce)

    def left_arc_eager(self, act_emb):
        stack_top = self.stack.top()
        buffer_first = self.buffer.left()
        # if not self.arcs.has_incoming(stack_top):
        self.stack.pop()
        self.arcs.add_arc(buffer_first[1], stack_top[1])
        self.action_history.append(act_emb)
        self.action_history_names.append(left_arc_eager)
        self.subtree_rep(buffer_first[0], stack_top[0], act_emb)

    def right_arc_eager(self, act_emb):
        stack_top = self.stack.top()
        buffer_first = self.buffer.left()
        self.stack.push(buffer_first)
        self.buffer.pop_left()
        self.arcs.add_arc(stack_top[1], buffer_first[1])
        self.action_history.append(act_emb)
        self.action_history_names.append(right_arc_eager)
        self.subtree_rep(stack_top[0], buffer_first[0], act_emb)

    def left_arc_hybrid(self, act_emb):
        stack_top = self.stack.top()
        buffer_first = self.buffer.left()
        self.stack.pop()
        self.arcs.add_arc(buffer_first[1], stack_top[1])
        self.action_history.append(act_emb)
        self.action_history_names.append(left_arc_hybrid)
        self.subtree_rep(buffer_first[0], stack_top[0], act_emb)

    def is_parse_complete(self, special_op=False):
        if special_op:
            return True
        # buffer has to be empty and stack only contains ROOT item
        buffer_empty = self.buffer.get_len() == 0
        stack_empty = self.stack.get_len() <= 1
        complete = buffer_empty and stack_empty
        return complete

    def get_heads(self):
        # return heads
        heads = torch.zeros((1, len(self.sentence), self.embedding_size * 2)).to(device=constants.device)
        with torch.no_grad():
            for i in range(len(self.sentence)):
                for j in range(len(self.sentence)):
                    if (j, i) in self.arcs.arcs:
                        heads[0, i, :] = self.ind2continous[j]
                        continue

        return heads
