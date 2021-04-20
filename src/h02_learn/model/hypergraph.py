from .modules import Item, Chart
import torch
import torch.nn as nn

from utils import constants
import numpy as np
from collections import defaultdict


class Hypergraph(object):

    def __init__(self, n, chart):
        self.n = n
        self.chart = chart

        self.spans = []
        self.scores = {}

        self.bucket = defaultdict(lambda: 0)


    def subtrees_(self, item):
        i, j, h = item.i, item.j, item.h
        for k in range(i, j):
            for r in range(i, j):

                if (i, k, r) in self.chart:
                    item_l = self.chart[(k, i, r)]
                    item.subtrees[item_l] = item_l
                if (j, k, r) in self.chart:
                    item_r = self.chart[(j, k, r)]
                    item.subtrees[item_r] = item_r

        return item

    def arc(self, item):
        i, j, h = item.i, item.j, item.h
        # print((i,j,h))
        if i + 1 == j:
            if i == h:
                item.arcs.append((h, j))
            else:
                item.arcs.append((h, i))
        else:
            if item.l.h == h:
                item.arcs = item.l.arcs + item.r.arcs + [(h, item.r.h)]
            else:
                item.arcs = item.l.arcs + item.r.arcs + [(h, item.l.h)]

        return item

    def traverse_up(self, item_l, item_r, path):
        path.append(item_l)
        path.append(item_r)
        if not self.is_axiom(item_l):
            path = self.traverse_up(item_l.l, item_l.r, path)
        if not self.is_axiom(item_r):
            path = self.traverse_up(item_r.l, item_r.r, path)

        return path

    def easy_first_path(self):
        for item in self.chart:
            i, j, h = item.i, item.j, item.h
        pass

    def check_laplacian(self, L):
        return torch.all(torch.diag(L) >= 0.0) and torch.all(L - torch.diag(L) <= 0.0)

    def partition(self):
        # G = complete graph on self.n
        A = torch.exp(self.W)
        A = A - torch.diag(A)
        col_sum = torch.diag(torch.sum(A, dim=0))
        L = -1 * A + col_sum
        # A = torch.exp(A)
        # A = torch.narrow(A, 0, 0, self.n)
        # A = torch.narrow(A,-1,0,self.n)
        # B = A * (1 - torch.eye(self.n+1, self.n+1))
        # col_sum = torch.diag(torch.sum(B, dim=-1))
        # L = col_sum - A
        indices = list(range(self.n + 1))
        indices.remove(1)
        L = L[indices, :]
        L = L[:, indices]
        print(L)
        # L = torch.narrow(L,0,1,self.n)
        # L = torch.narrow(L,-1,1,self.n)
        print("XUYUYUYUYUYU {}".format(self.check_laplacian(L)))

        # L[0, :] = r[:-1]
        Z = L.det()
        return Z

    def probability(self, heads):
        heads_proper = [0] + heads.tolist()

        sentence_proper = list(range(len(heads_proper)))
        word2head = {w: h for (w, h) in zip(sentence_proper, heads_proper)}
        arcs = []
        for word in word2head:
            arcs.append((word2head[word], word))
        psum = 0
        for (u, v) in arcs:
            if u == 0:
                root = v
                continue
            psum += self.W[u, v]
        root_selection = self.W[root, :]
        psum = torch.exp(torch.sum(torch.tensor(psum).to(device=constants.device)))
        Z = self.partition()
        return psum / Z  # self.partition(root_selection)

    def best_path(self):
        for item in self.chart:
            # print(item)
            if item[0] == 0 and item[1] == self.n:
                goal = (0, self.n, item[2])
                break

        goal_item = self.chart[goal]
        arcs = []
        scores = torch.zeros(self.n)
        path = [goal_item]

        path = self.traverse_up(goal_item.l, goal_item.r, path)
        # print(path)
        arcs = set(goal_item.arcs)
        path_arcs = set([])
        for item in path:
            if item != goal_item:
                path_arcs = path_arcs.union(set(item.arcs))

        # print(path_arcs)
        # print(arcs)

        arcs = arcs.intersection(path_arcs)
        # do in order

        # scores = [self.chart[item].w for item in reversed(path)]
        # print(scores)

        # tot_sum = sum(scores)
        # cumsum = torch.cumsum(torch.tensor(scores), dim=-1)
        # probs = cumsum  # /tot_sum

        arcs = set(arcs)
        heads = torch.zeros(self.n)
        for (u, v) in arcs:
            heads[v - 1] = u
        return heads

    def siblings(self):
        pass

    def grandchild(self):
        pass

    def axiom(self, i):
        pass

    def is_axiom(self, item):
        pass


class LazyArcStandard(Hypergraph):

    def __init__(self, n, chart):
        super().__init__(n, chart)
        for i in range(n + 1):
            item = Item(i, i + 1, i, i, i)
            self.chart[item] =  item
        self.locator = Chart()#defaultdict(lambda : 0)
        self.trees = Chart()
        self.linear_tree_left = nn.Linear(300, 100)
        self.linear_tree_right = nn.Linear(300, 100)

    def axiom(self, item):
        i, j, h = item.i, item.j, item.h
        return i + 1 == j and i == h and h + 1 == j

    def is_axiom(self, item):
        i, j, h = item.i, item.j, item.h
        if j == i + 1 and i == h and j == h + 1:
            return True
        else:
            return False

    def update_chart(self, item):
        self.chart[item] = item
        return self

    def delete_from_chart(self, item):
        i, j, h = item.i, item.j, item.h
        del self.chart[(i, j, h)]
        # self.chart.chart.pop(item,None)
        return self

    def add_bucket(self, item):
        self.bucket[item.l] += 1
        self.bucket[item.r] += 1
        return self

    def compute_trees(self, item, label):
        i, j, h = item.i, item.j, item.h
        item_l = item.l
        item_r = item.r
        il, jl, hl = item_l.i, item_l.j, item_l.h
        ir, jr, hr = item_r.i, item_r.j, item_r.h

        tl = self.trees[(il, jl, hl)]
        tr = self.trees[(ir, jr, hr)]
        repr_l = torch.cat([tl, tr, label], dim=-1)
        repr_r = torch.cat([tr, tl, label], dim=-1)
        c_l = nn.Tanh()(self.linear_tree_left(repr_l))
        c_r = nn.Tanh()(self.linear_tree_right(repr_r))
        self.trees[(i, j, hl)] = c_l
        self.trees[(i, j, hr)] = c_r

    def chart_to_matrix(self, left_trees, right_trees):

        pass

    def make_legal(self, x, picks):
        scores = torch.ones_like(x)
        scores *= -float('inf')
        for (u,v) in picks:
            scores[u,v] = x[u,v]
            scores[v,u] = x[v,u]
        return scores

    def new_trees(self, item, popped):
        i, j, h = item.i, item.j, item.h
        picks_left = []
        picks_right = []
        picks = []
        for k in range(0, i + 1):
            for g in range(k, i):
                if (k, i, g) in self.chart and g not in popped:
                    item_l = self.chart[(k, i, g)]
                    if g not in popped and h not in popped:
                        picks.append((g, h))
                        self.locator[(g,h)] = Item(k, j, g, item_l, item)
                    if h not in popped and g not in popped:
                        picks.append((g, h))
                        self.locator[(h,g)] = Item(k, j, h, item_l, item)
                    #kjg = torch.tensor([[k, i, g], [i, j, h], [k, j, g]], dtype=torch.int).to(device=constants.device)
                    #kjh = torch.tensor([[k, i, g], [i, j, h], [k, j, h]], dtype=torch.int).to(device=constants.device)
                    #picks_left.append(kjg)
                    #picks_right.append(kjh)
        for k in range(j, self.n+1):
            for g in range(j, k):
                if (j, k, g) in self.chart:
                    item_r = self.chart[(j, k, g)]
                    if h not in popped and g not in popped:
                        picks.append((h, g))
                        self.locator[(h, g)] = Item(i, k, h, item, item_r)
                    if g not in popped and h not in popped:
                        picks.append((h, g))
                        self.locator[(g, h)] = Item(i, k, g, item, item_r)
                    #ikh = torch.tensor([[i, j, h], [j, k, g], [i, k, h]], dtype=torch.int).to(device=constants.device)
                    #ikg = torch.tensor([[i, j, h], [j, k, g], [i, k, g]], dtype=torch.int).to(device=constants.device)
                    #picks_left.append(ikh)
                    #picks_right.append(ikg)
        return picks

    def outgoing(self, item):
        """ Lazily Expand the Hypergraph """
        i, j, h = item.i, item.j, item.h
        # items to the left
        # w = self.score(item)
        # self.arc(item)
        all_arcs = []
        for k in range(0, i + 1):
            for g in range(k, i):
                if (k, i, g) in self.chart:
                    item_l = self.chart[(k, i, g)]
                    # if (k,j,g) != (i,j,h):
                    all_arcs.append(Item(k, j, g, item_l, item))
                    # if (k,j,h) != (i,j,h):
                    all_arcs.append(Item(k, j, h, item_l, item))

        # items to the right
        for k in range(j, self.n + 1):
            for g in range(j, k):
                if (j, k, g) in self.chart:
                    item_r = self.chart[(j, k, g)]
                    # if (i,k,h) != (i,j,h):
                    all_arcs.append(Item(i, k, h, item, item_r))
                    # if (i,k,g) != (i,j,h):
                    all_arcs.append(Item(i, k, g, item, item_r))

        return all_arcs


class LazyArcEager(Hypergraph):
    def __init__(self, n, chart, W):
        self.n = n
        self.chart = chart
        self.W = W

    def outgoing(self, item):
        pass


class LazyHybrid(Hypergraph):
    def __init__(self, n, chart, W):
        self.n = n
        self.chart = chart
        self.W = W

    def outgoing(self, item):
        pass


class LazyMH4(Hypergraph):
    def __init__(self, n, chart, W):
        self.n = n
        self.chart = chart
        self.W = W

    def outgoing(self, item):
        pass
