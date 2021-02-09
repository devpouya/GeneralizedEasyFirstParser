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

left_arc_2 = "LEFT_ARC_2"
right_arc_2 = "RIGHT_ARC_2"

arc_standard = ([shift, reduce_l, reduce_r], range(3))  # {shift: 0, reduce_l: 1, reduce_r: 2}
# arc_standard_actions = {0: fshift, 1: freduce_l, 2: freduce_r}

arc_eager = ([shift, left_arc_eager, right_arc_eager, reduce],
             range(4))  # {shift: 0, left_arc_eager: 1, right_arc_eager: 2, reduce: 3}
# arc_eager_actions = {0: fshift, 1: fleft_arc_eager, 2: fright_arc_eager, 3: freduce}

hybrid = ([shift, left_arc_hybrid, reduce_r], range(3))  # {shift: 0, left_arc_hybrid: 1, reduce_r: 2}

non_projective = ([shift,reduce_l,reduce_r,left_arc_2,right_arc_2],range(5))
# hybrid_actions = {0: fshift, 1: fleft_arc, 2: freduce_r}

easy_first = (0,0)
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
        # it uses the most recent representation (from the combinations)
        self.ind2continous = {i: vec for (vec, i) in self.buffer.buffer}

        self.heads = torch.zeros((1, len(self.sentence), len(self.sentence))).to(device=constants.device)
        self.head_list = torch.zeros((1,len(self.sentence))).to(device=constants.device)
        # used for learning representation for partial parse trees
        self.linear = nn.Linear(5 * embedding_size, 2 * embedding_size).to(device=constants.device)
        self.tanh = nn.Tanh().to(device=constants.device)

    def get_stack_content(self):
        return [item[0] for item in self.stack.stack]

    def get_buffer_content(self):
        return [item[0] for item in self.buffer.buffer]

    def subtree_rep(self, top, second, act_emb):
        reprs = torch.cat([top[0], second[0], act_emb.reshape(self.embedding_size)],
                          dim=-1)
        c = self.tanh(self.linear(reprs))
        self.ind2continous[top[1]] = c
        self.stack.set_top(c)
        return c

    def shift(self, act_emb):
        item = self.buffer.pop_left()
        self.stack.push(item)
        self.action_history_names.append(shift)
        self.action_history.append(act_emb)
        return item[0]

    def reduce_l(self, act_emb):
        item_top = self.stack.top()
        item_second = self.stack.pop_second()
        self.arcs.add_arc(item_top[1], item_second[1])
        self.heads[0, item_top[1], item_second[1]] = 1
        self.head_list[0,item_top[1]] = item_second[1]
        self.action_history_names.append(reduce_l)
        self.action_history.append(act_emb)

        # compute build representation and use this from now on
        c = self.subtree_rep(item_top, item_second, act_emb)
        return c

    def reduce_r(self, act_emb):
        second_item = self.stack.second()
        top_item = self.stack.pop()
        self.arcs.add_arc(second_item[1], top_item[1])
        self.heads[0, second_item[1], top_item[1]] = 1
        self.head_list[0,second_item[1]] = top_item[1]
        self.action_history_names.append(reduce_r)
        self.action_history.append(act_emb)

        # compute build representation and use this from now on
        c = self.subtree_rep(second_item, top_item, act_emb)
        return c

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
        self.heads[0, buffer_first[1], stack_top[1]] = 1
        self.head_list[0,buffer_first[1]] = stack_top[1]
        self.action_history.append(act_emb)
        self.action_history_names.append(left_arc_eager)
        c = self.subtree_rep(buffer_first, stack_top, act_emb)
        return c

    def right_arc_eager(self, act_emb):
        stack_top = self.stack.top()
        buffer_first = self.buffer.left()
        self.stack.push(buffer_first)
        self.buffer.pop_left()
        self.arcs.add_arc(stack_top[1], buffer_first[1])
        self.heads[0, stack_top[1], buffer_first[1]] = 1
        self.head_list[0,stack_top[1]] = buffer_first[1]
        self.action_history.append(act_emb)
        self.action_history_names.append(right_arc_eager)
        c = self.subtree_rep(stack_top, buffer_first, act_emb)
        return c

    def left_arc_hybrid(self, act_emb):
        stack_top = self.stack.top()
        buffer_first = self.buffer.left()
        self.stack.pop()
        self.arcs.add_arc(buffer_first[1], stack_top[1])
        self.heads[0, buffer_first[1], stack_top[1]] = 1
        self.head_list[0,buffer_first[1]] = stack_top[1]
        self.action_history.append(act_emb)
        self.action_history_names.append(left_arc_hybrid)
        c = self.subtree_rep(buffer_first, stack_top, act_emb)
        return c

    def left_arc_second(self, act_emb):
        third = self.stack.pop_third()
        top = self.stack.top()
        self.arcs.add_arc(top[1],third[1])
        self.heads[0,top[1],third[1]] = 1
        self.head_list[0, top[1]] = third[1]
        self.action_history.append(act_emb)
        self.action_history_names.append(left_arc_2)
        c = self.subtree_rep(top,third,act_emb)
        return c

    def right_arc_second(self, act_emb):
        third = self.stack.third()
        top = self.stack.pop()
        self.arcs.add_arc(third,top)
        self.heads[0,third[1],top[1]] = 1
        self.head_list[0,third[1]] = top[1]
        self.action_history.append(act_emb)
        self.action_history_names.append(right_arc_2)
        c = self.subtree_rep(third,top,act_emb)
        return c


    def is_parse_complete(self, special_op=False):
        if special_op:
            return True
        # buffer has to be empty and stack only contains ROOT item
        buffer_empty = self.buffer.get_len() == 0
        stack_empty = self.stack.get_len() <= 1
        complete = buffer_empty and stack_empty
        return complete

    def get_head_embeddings(self,non_proj=False):
        heads_embed = torch.zeros((1, len(self.sentence), self.embedding_size * 2)).to(device=constants.device)
        if non_proj:
            return heads_embed
        else:
            for i in range(len(self.sentence)):
                try:
                    index = torch.where(self.heads[0, i] == 1)[0].item()
                    heads_embed[0, i, :] = self.ind2continous[index]
                except:
                    pass


            return heads_embed

    def get_heads(self):
        # return heads
        heads_embed = torch.zeros((1, len(self.sentence), self.embedding_size * 2)).to(device=constants.device)
        heads = torch.zeros((1, len(self.sentence), len(self.sentence))).to(device=constants.device)
        with torch.no_grad():
            for i in range(len(self.sentence)):
                for j in range(len(self.sentence)):
                    if (j, i) in self.arcs.arcs:
                        heads_embed[0, i, :] = self.ind2continous[j]
                        heads[0, i, j] = 1  # j
                        continue

        return heads, heads_embed
