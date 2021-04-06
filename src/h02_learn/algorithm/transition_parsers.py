import torch
import torch.nn as nn
from utils import constants
from ..model.modules import Agenda, Chart, Item
import heapq
from collections import defaultdict


# arc-standard shift reduce parser


class ChartParser():
    def __init__(self, sentence, embedding_size, transition_system, mlp, hypergraph=None, prune=False):

        self.mlp = mlp
        self.sentence = sentence
        self.n = len(sentence)
        self.embedding_size = embedding_size
        #self.transition_system = transition_system

        self.axioms = self.axioms(transition_system)
        self.prune = prune
        self.pops = 0
        self.prunes = 0

        self.agenda = Agenda()
        self.chart = Chart()
        W = torch.tensor((self.n+1,self.n+1)).to(device=constants.device)
        nn.init.xavier_uniform_(W)
        self.hypergraph = hypergraph(self.n, self.chart, W,mlp,sentence)
        for item in self.axioms:
            i,j,h = item.i, item.j, item.h
            self.agenda[(i,j,h)] = Item(i,j,h,hypergraph.W[(j,i)],i,i)

        self.bucket = defaultdict(lambda: 0)
        self.popped = defaultdict(lambda: 0)
        self.spans = []
        for i, word in enumerate(sentence):
            self.spans.append(word.clone())

        self.arcs = []

    def axioms(self, transition_system):
        if transition_system == constants.arc_standard:
            return None
        elif transition_system == constants.arc_eager:
            return None
        elif transition_system == constants.hybrid:
            return None
        elif transition_system == constants.mh4:
            return None
        else:
            raise Exception("A valid transition system needs to be specified")

    def curr_item(self):
        pass

    def next_item(self):
        pass

    def parse_step(self):
        item = self.agenda.pop()
        if self.prune:
            if item.l in self.bucket or item.r in self.bucket:
                return "pruned"
            self.bucket[item.l] += 1
            self.bucket[item.r] += 1
        self.chart[item] = item
        self.popped[(item.i, item.j, item.h)] += 1
        self.pops += 1
        # score in hgraph?????
        for item_new in self.hypergraph.outgoint(item):
            self.agenda[(item_new.i, item_new.j, item_new.h)] = item_new
        return "STEP"

    def finished(self):
        # check if final item is derived
        return self.chart[(0,self.n+1,0)] > 0



class ShiftReduceParser():

    def __init__(self, sentence, embedding_size, transition_system, hypergraph=None, prune=False):

        init_sent = []
        self.buffer = []
        self.bucket = []
        self.pending = []

        self.n = len(sentence)
        # self.buffer.append((torch.rand_like(sentence[0]),0))
        for i, word in enumerate(sentence):
            self.buffer.append((word.clone(), i))
            self.pending.append((word.clone(), i))
        self.stack = []
        self.arcs = []
        # self.buffer = Buffer(sentence)
        # self.arcs = Arcs()

        self.hypergraph = hypergraph
        self.prune = prune
        self.pops = 0
        self.prunes = 0



        self.action_history_names = []
        self.actions_probs = torch.zeros((1, len(transition_system[0]))).to(device=constants.device)
        self.oracle_action_history = []

        # sentence to parse
        self.sentence = sentence
        self.embedding_size = embedding_size

    def config_to_item(self):
        max_head = -1000
        for (u, v) in self.arcs:
            if u > max_head:
                max_head = u
        return self.agenda[(self.stack[-1][1], self.buffer[0][1], max_head)]

    def take_action(self, item,**args):
        current_item = self.config_to_item()
        i_curr,j_curr,h_curr = current_item.i, current_item.j, current_item.h
        i_next, j_next, h_next = item.i,item.j,item.h
        if i_next == j_curr:
            self.shift()
        elif i_next == self.stack[-2][1]:
            self.reduce_r(*args)
        else:
            self.reduce_l(*args)


    def parse_step(self,**args):
        item = self.agenda.pop()
        if self.prune:
            if item.l in self.bucket or item.r in self.bucket:
                self.prunes+=1
                return "PRUNED"
            self.bucket[item.l] += 1
            self.bucket[item.r] += 1
        self.chart[item] = item
        self.take_action(item,*args)
        self.popped[(item.i, item.j, item.h)] += 1
        self.pops += 1
        for item_new in self.hypergraph.outgoint(item):
            self.agenda[(item_new.i, item_new.j, item_new.h)] = item_new

        return "stuff we did"

    def is_complete(self):
        return self.agenda.empty()

    def window(self, i):
        if i - 2 >= 0 and i + 3 <= len(self.pending):
            return slice(i - 2, i + 3)
        elif i - 2 < 0:
            lower_bound = 0
            if i == 0:
                upper_bound = min(5, len(self.pending))
            else:
                upper_bound = min(4, len(self.pending))
            return slice(lower_bound, upper_bound)
        else:
            return slice(i - 2, min(i + 3, len(self.pending)))

    def score_pending(self, mlp_u, mlp_l, lstm, emb_left, emb_right):
        action_s = []
        rel_s = []

        # calculate score for every (action,label) pair in every position i
        # scores = torch.zeros((len(self.pending),2,50)).to(device=constants.device)
        """
            have a history of actions, plus a stack-lstm representation of pending as input to mlp'ss
        """
        scores = []
        tree = torch.stack(list(list(zip(*self.pending))[0])).unsqueeze(1).to(device=constants.device)
        rep, (h, c) = lstm(tree)
        for i in range(len(self.pending)):
            window_trees = list(list(zip(*self.pending))[0][self.window(i)])
            if len(window_trees) < 5:
                while len(window_trees) < 5:
                    window_trees.append(torch.zeros_like(window_trees[0]).to(device=constants.device))
            window_trees_vec = torch.cat(window_trees, dim=0)
            window_trees_left = torch.cat([window_trees_vec, emb_left, h.squeeze(0).squeeze(0)], dim=0)
            window_trees_right = torch.cat([window_trees_vec, emb_right, h.squeeze(0).squeeze(0)], dim=0)
            score_uil = mlp_u(window_trees_left)
            score_uir = mlp_u(window_trees_right)
            score_lil = mlp_l(window_trees_left)
            score_lir = mlp_l(window_trees_right)
            score_il = score_uil + torch.max(score_lil, dim=0)[0]
            score_ir = score_uir + torch.max(score_lir, dim=0)[0]
            scores.append(score_il)
            scores.append(score_ir)
            action_s.append(score_uil)
            action_s.append(score_uir)
            rel_s.append(score_lil)
            rel_s.append(score_lir)
        action_probabilities = nn.Softmax(dim=-1)(torch.stack(action_s))
        best_i = torch.argmax(torch.stack(scores), dim=0)
        i = best_i.item()
        if best_i % 2 == 0:
            direction = 1
            index = int(i / 2)
        else:
            direction = 0
            index = int((i - 1) / 2)
        rel_probabilities = nn.Softmax(dim=-1)(rel_s[best_i])
        return action_probabilities, rel_probabilities, index, direction, lstm

    def easy_first_action(self, index_head, index_mod, rel, rel_embed, linear):

        self.arcs.append((self.pending[index_head][1], self.pending[index_mod][1], rel))
        ret = self.subtree_rep_pending(index_head, self.pending[index_head], self.pending[index_mod], rel_embed, linear)
        self.pending.pop(index_mod)
        return ret

    def subtree_rep_pending(self, i, head, mod, rel_embed, linear):

        reprs = torch.cat([head[0], mod[0], rel_embed.reshape(self.embedding_size)],
                          dim=-1)

        c = nn.Tanh()(linear(reprs))
        (_, ind) = self.pending[i]
        self.pending[i] = (c, ind)
        return c

    def subtree_rep(self, top, second, rel_embed, linear):

        reprs = torch.cat([top[0], second[0], rel_embed.reshape(self.embedding_size)],
                          dim=-1)

        c = nn.Tanh()(linear(reprs))
        (_, ind) = self.stack[-1]
        self.stack[-1] = (c, ind)
        return c

    def subtree_rep_prime(self, top, second, rel_embed, linear):

        reprs = torch.cat([top[0], second[0], rel_embed.reshape(self.embedding_size)],
                          dim=-1)

        c = nn.Tanh()(linear(reprs))
        (_, ind) = self.stack[-2]
        self.stack[-2] = (c, ind)
        return c

    def subtree_rep_hybrid(self, top, second, rel_embed, linear):

        reprs = torch.cat([top[0], second[0], rel_embed.reshape(self.embedding_size)],
                          dim=-1)

        c = nn.Tanh()(linear(reprs))
        (_, ind) = self.buffer[0]
        self.buffer[0] = (c, ind)
        return c

    def legal_indices_mh4(self):
        if len(self.stack) < 1:
            return [0]
        elif len(self.buffer) < 1:
            return [2, 3, 4, 6]
        elif 3 > len(self.stack) >= 2:
            return [0, 1, 2, 3, 5]
        elif len(self.stack) < 2 and len(self.buffer) >= 1:
            return [1]
        elif len(self.buffer) >= 1 and len(self.stack) >= 3:
            return [0, 1, 2, 3, 4, 5, 6]

    def shift(self):
        item = self.buffer.pop(0)
        self.stack.append(item)
        return item[0]

    def reduce_l(self, rel, rel_embed, linear):
        top = self.stack[-1]
        second = self.stack[-2]
        # left = self.buffer[0]
        self.stack.pop(-2)
        self.arcs.append((top[1], second[1], rel))
        c = self.subtree_rep(top, second, rel_embed, linear)
        return c

    def reduce_r(self, rel, rel_embed, linear):
        second = self.stack[-2]
        top = self.stack[-1]
        self.arcs.append((second[1], top[1], rel))
        self.stack.pop(-1)
        # self.buffer[0] = top
        c = self.subtree_rep(second, top, rel_embed, linear)
        return c

    def reduce(self):
        # stack_top = self.stack.top()
        # if self.arcs.has_incoming(stack_top):
        self.stack.pop(-1)
        self.action_history_names.append(constants.reduce)

    def left_arc_eager(self, rel, rel_embed, linear):
        top = self.stack[-1]
        left = self.buffer[0]
        # if not self.arcs.has_incoming(stack_top):
        self.stack.pop(-1)
        self.arcs.append((left[1], top[1], rel))
        c = self.subtree_rep_hybrid(left, top, rel_embed, linear)
        return c

    def left_arc_prime(self, rel, rel_embed, linear):
        top = self.stack[-1]
        second = self.stack[-2]
        self.arcs.append((top[1], second[1], rel))
        c = self.subtree_rep(top, second, rel_embed, linear)
        return c

    def right_arc_prime(self, rel, rel_embed, linear):
        second = self.stack[-2]
        third = self.stack[-3]
        self.arcs.append((third[1], second[1], rel))
        self.stack.pop(-2)
        c = self.subtree_rep_prime(third, second, rel_embed, linear)
        return c

    def left_arc_2(self, rel, rel_embed, linear):
        front = self.buffer[0]
        second = self.stack[-2]
        self.arcs.append((front[1], second[1], rel))
        self.stack.pop(-2)
        c = self.subtree_rep_hybrid(front, second, rel_embed, linear)
        return c

    def right_arc_2(self, rel, rel_embed, linear):
        third = self.stack[-3]
        top = self.stack[-1]
        self.arcs.append((third[1], top[1], rel))
        self.stack.pop(-1)
        c = self.subtree_rep_prime(third, top, rel_embed, linear)
        return c

    def right_arc_eager(self, rel, rel_embed, linear):
        top = self.stack[-1]
        left = self.buffer[0]
        self.stack.append(left)
        self.arcs.append((top[1], left[1], rel))
        c = self.subtree_rep(top, left, rel_embed, linear)
        self.buffer.pop(0)
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
        heads = [0] * (len(self.sentence) + 1)
        rels = [0] * (len(self.sentence) + 1)
        # self.arcs.append((0,0,1))
        # print(self.arcs)
        tmp = self.arcs.copy()
        tmp.append((0, 0, 1))
        for i in range(len(self.sentence)):
            for (u, v, r) in tmp:
                if v == i:
                    heads[i + 1] = u
                    rels[i + 1] = r
        heads.pop(0)
        rels.pop(0)

        return torch.tensor(heads).to(device=constants.device), torch.tensor(rels).to(device=constants.device)

    def set_oracle_action(self, act):
        self.oracle_action_history.append(act)

    def set_action_probs(self, probs):

        self.actions_probs = torch.cat([self.actions_probs, probs.transpose(1, 0)], dim=0)
