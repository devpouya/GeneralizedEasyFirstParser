from .modules import Item, Chart
import torch
import torch.nn as nn

from utils import constants
import numpy as np
from collections import defaultdict

from termcolor import colored


class Hypergraph(object):

    def __init__(self, n, chart, rels):
        self.n = n
        self.chart = chart

        self.spans = []
        self.scores = {}
        self.rels = rels

        self.bucket = defaultdict(lambda: 0)
        self.scored_items = {}

        self.possible_next = {}
        self.repped_items = {}
        self.right_children = {i: [i] for i in range(n)}
        self.left_children = {i: [i] for i in range(n)}

    def set_possible_next(self, items):
        for k in items.keys():
            self.possible_next[k] = items[k]
        return self

    def remove_key_possible_next(self, key):
        del self.possible_next[key]
        return self

    def return_gold_next(self, items):
        gold_index, next_item = None, None
        # for item in self.possible_next.keys():
        #    print(item)
        for i, item in enumerate(items):
            if (item.i, item.j, item.h) in self.possible_next.keys():
                gold_index = torch.tensor([i], dtype=torch.long).to(device=constants.device)
                next_item = self.possible_next[(item.i, item.j, item.h)]
                del self.possible_next[(item.i, item.j, item.h)]
                break
        return gold_index, next_item

    def set_item_vec(self, vec, item):
        item = item.set_vector_rep(vec)
        self.repped_items[item] = item
        return self

    def score_item(self, item):
        self.scored_items[item] = item

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

    def add_right_child(self, head, mod):
        self.right_children[head].append(mod)
        return self

    def get_right_children_until(self, head,i):
        rc = self.right_children[head]
        ret = []
        for item in list(rc):
            if i > item >= head:
                ret.append(item)
        ret.append(i)
        return ret

    def add_left_child(self, head, mod):
        self.left_children[head].append(mod)
        return self

    def get_left_children_from(self, head,i):
        lc = self.right_children[head]
        ret = [i]
        for item in list(lc):
            if i < item <= head:
                ret.append(item)
        return ret

    def axiom(self, i):
        pass

    def is_axiom(self, item):
        pass

    def make_arc(self, item, add_rel=False):
        pass


class LazyArcStandard(Hypergraph):

    def __init__(self, n, chart, rels):
        super().__init__(n, chart, rels)
        # for i in range(n + 1):
        #    item = Item(i, i + 1, i, i, i)
        #    self.chart[item] =  item
        self.locator = Chart()  # defaultdict(lambda : 0)
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

    def make_arc(self, item, add_rel=False):
        arc = (item.h, item.l.h if item.l.h != item.h else item.r.h)
        if add_rel:
            m = arc[1] - 1
            rel_made = self.rels[m]
            item.add_rel(rel_made)
        return arc, item

    def outgoing(self, item):
        """ Lazily Expand the Hypergraph """
        i, j, h = item.i, item.j, item.h
        # items to the left
        # w = self.score(item)
        # self.arc(item)
        #all_arcs = []
        all_arcs = {}

        for k in range(0, i + 1):
            for g in range(k, i):
                if (k, i, g) in self.chart:
                    if self.chart[(k, i, g)] not in self.bucket:
                        item_l = self.chart[(k, i, g)]

                        item1 = Item(k, j, g, item_l, item)
                        _, item1 = self.make_arc(item1, add_rel=True)
                        #all_arcs.append(item1)
                        all_arcs[item1] = item1

                        item2 = Item(k, j, h, item_l, item)
                        _, item2 = self.make_arc(item2)
                        #all_arcs.append(item2)
                        all_arcs[item2] = item2

        # items to the right
        for k in range(j, self.n + 1):
            for g in range(j, k):
                if (j, k, g) in self.chart:
                    if self.chart[(j, k, g)] not in self.bucket:
                        item_r = self.chart[(j, k, g)]

                        item_n1 = Item(i, k, h, item, item_r)
                        _, item_n1 = self.make_arc(item_n1, add_rel=True)
                        #all_arcs.append(item_n1)
                        all_arcs[item_n1] = item_n1

                        item_n2 = Item(i, k, g, item, item_r)
                        _, item_n2 = self.make_arc(item_n2, add_rel=True)
                        #all_arcs.append(item_n2)
                        all_arcs[item_n2] = item_n2

        return all_arcs


class LazyArcEager(Hypergraph):
    def __init__(self, n, chart, rels):
        super().__init__(n, chart, rels)

    def outgoing(self, item):
        pass


class LazyHybrid(Hypergraph):
    def __init__(self, n, chart, rels):
        super().__init__(n, chart, rels)

    def axiom(self, item):
        i, j, h = item.i, item.j, item.h
        return i + 1 == j and i == h and h + 1 == j

    def is_axiom(self, item):
        i, j, h = item.i, item.j, item.h
        if j == i + 1 and i == h and j == h + 1:
            return True
        else:
            return False

    def make_arc(self, item, add_rel=False):
        h = item.h
        if item.r.j == h:
            other = item.r.i
        elif item.l.i == h:
            other = item.l.j
        arc = (h, other)
        if add_rel:
            m = arc[1] - 1
            rel_made = self.rels[m]
            item.add_rel(rel_made)
        return arc, item

    def outgoing(self, item):
        i, j, h = item.i, item.j, item.h

        all_arcs = {}
        for k in range(0, i + 1):
            if (k, i, k) in self.chart:
                if self.chart[(k, i, k)] not in self.bucket:
                    item_l = self.chart[(k, i, k)]
                    item_1 = Item(k, j, k, item_l, item)
                    _, item_1 = self.make_arc(item_1, add_rel=True)
                    #all_arcs.append(item_1)
                    all_arcs[item_1] = item_1
                    if j < self.n:
                        item_2 = Item(k, j, j, item_l, item)
                        _, item_2 = self.make_arc(item_2, add_rel=True)
                        #all_arcs.append(item_2)
                        all_arcs[item_2] = item_2
            if (k, i, i) in self.chart:
                if self.chart[(k, i, i)] not in self.bucket:
                    item_l = self.chart[(k, i, i)]
                    item_1 = Item(k, j, k, item_l, item)
                    _, item_1 = self.make_arc(item_1, add_rel=True)
                    #all_arcs.append(item_1)
                    all_arcs[item_1] = item_1
                    if j < self.n:
                        item_2 = Item(k, j, j, item_l, item)
                        _, item_2 = self.make_arc(item_2, add_rel=True)
                        #all_arcs.append(item_2)
                        all_arcs[item_2] = item_2

        for k in range(j, self.n + 1):
            if (j, k, j) in self.chart:
                if self.chart[(j, k, j)] not in self.bucket:
                    item_r = self.chart[(j, k, j)]
                    item_1 = Item(i, k, i, item, item_r)
                    _, item_1 = self.make_arc(item_1, add_rel=True)
                    #all_arcs.append(item_1)
                    all_arcs[item_1] = item_1
                    if k < self.n:
                        item_2 = Item(i, k, k, item, item_r)
                        _, item_2 = self.make_arc(item_2, add_rel=True)
                        #all_arcs.append(item_2)
                        all_arcs[item_2] = item_2
            if (j, k, k) in self.chart:
                if self.chart[(j, k, k)] not in self.bucket:
                    item_r = self.chart[(j, k, k)]
                    item_1 = Item(i, k, i, item, item_r)
                    _, item_1 = self.make_arc(item_1, add_rel=True)
                    #all_arcs.append(item_1)
                    all_arcs[item_1] =item_1
                    if k < self.n:
                        item_2 = Item(i, k, k, item, item_r)
                        _, item_2 = self.make_arc(item_2, add_rel=True)
                        #all_arcs.append(item_2)
                        all_arcs[item_2] = item_2

        return all_arcs


class LazyMH4(Hypergraph):
    def __init__(self, n, chart, rels):
        super().__init__(n, chart, rels)

    def outgoing(self, item):
        pass
