import torch
import torch.nn as nn
from utils import constants


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


    def subtree_rep(self, top, second, rel_embed,linear):

        reprs = torch.cat([top[0], second[0], rel_embed.reshape(self.embedding_size)],
                          dim=-1)

        c = nn.Tanh()(linear(reprs))
        (_, ind) = self.stack[-1]
        self.stack[-1] = (c, ind)
        return c

    def subtree_rep_prime(self, top, second, rel_embed,linear):

        reprs = torch.cat([top[0], second[0], rel_embed.reshape(self.embedding_size)],
                          dim=-1)

        c = nn.Tanh()(linear(reprs))
        (_, ind) = self.stack[-2]
        self.stack[-2] = (c, ind)
        return c
    def subtree_rep_eager(self,top,second,rel_embed,linear,is_right=False):
        if not is_right:
            return self.subtree_rep(top,second,rel_embed,linear)
        else:
            return self.subtree_rep(second,top,rel_embed,linear)

    def subtree_rep_hybrid(self, top, second, rel_embed,linear):

        reprs = torch.cat([top[0], second[0], rel_embed.reshape(self.embedding_size)],
                          dim=-1)

        c = nn.Tanh()(linear(reprs))
        (_, ind) = self.buffer[0]
        self.buffer[0] = (c, ind)
        return c



    def shift(self):
        item = self.buffer.pop(0)
        self.stack.append(item)
        self.action_history_names.append(constants.shift)
        return item[0]

    def reduce_l(self, rel,rel_embed,linear):
        top = self.stack[-1]
        second = self.stack[-2]
        #left = self.buffer[0]
        self.stack.pop(-2)
        self.arcs.append((top[1], second[1], rel))
        self.action_history_names.append(constants.reduce_l)
        c = self.subtree_rep(top, second,rel_embed,linear)
        return c

    def reduce_r(self, rel,rel_embed,linear):
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

    def left_arc_eager(self,rel,rel_embed,linear):
        top = self.stack[-1]
        left = self.buffer[0]
        # if not self.arcs.has_incoming(stack_top):
        self.stack.pop(-1)
        self.arcs.append((left[1], top[1],rel))
        self.action_history_names.append(constants.left_arc_eager)
        c = self.subtree_rep_hybrid(left, top, rel_embed,linear)
        return c
    def left_arc_prime(self,rel,rel_embed,linear):
        top = self.stack[-1]
        second = self.stack[-2]
        self.arcs.append((top[1],second[1],rel))
        self.action_history_names.append(constants.left_arc_prim)
        c = self.subtree_rep(top,second,rel_embed,linear)
        return c

    def right_arc_prime(self, rel, rel_embed, linear):
        second = self.stack[-2]
        third = self.stack[-3]
        self.arcs.append((third[1],second[1],rel))
        self.stack.pop(-2)
        c = self.subtree_rep_prime(third, second, rel_embed, linear)
        return c

    def left_arc_2(self,rel,rel_embed,linear):
        front = self.buffer[0]
        second = self.stack[-2]
        self.arcs.append((front[1],second[1],rel))
        self.stack.pop(-2)
        c = self.subtree_rep_hybrid(front, second, rel_embed, linear)
        return c
    def right_arc_2(self,rel,rel_embed,linear):
        third = self.stack[-3]
        top = self.stack[-1]
        self.arcs.append((third[1],top[1],rel))
        self.stack.pop(-1)
        c = self.subtree_rep_prime(third, top, rel_embed, linear)
        return c
    def right_arc_eager(self,rel,rel_embed,linear):
        top = self.stack[-1]
        left = self.buffer[0]
        self.stack.append(left)
        self.arcs.append((top[1], left[1],rel))
        self.action_history_names.append(constants.right_arc_eager)
        c = self.subtree_rep(top, left, rel_embed,linear)
        self.buffer.pop(0)
        return c

    def left_arc_hybrid(self, act_emb,rel,rel_embed,linear):
        stack_top = self.stack[-1]
        buffer_first = self.buffer[0]
        self.arcs.append((buffer_first[1], stack_top[1],rel))
        self.action_history_names.append(constants.left_arc_hybrid)
        c = self.subtree_rep(buffer_first, stack_top, rel_embed,linear)
        self.stack.pop(-1)

        return c

    def reduce_r_hybrid(self, act_emb, rel, rel_embed, linear):
        top = self.stack[-1]
        second = self.stack[-2]
        self.arcs.append((second[1],top[1],rel))
        self.action_history_names.append(constants.reduce_r)
        self.stack.pop(-1)
        c = self.subtree_rep(second,top,rel_embed,linear)
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

