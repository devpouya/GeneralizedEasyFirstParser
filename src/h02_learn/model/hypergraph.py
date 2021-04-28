from .modules import Item, Chart
from .modules import ItemW
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
        for iter, (i,j,h) in enumerate(items.keys()):
            if (i, j, h) in self.possible_next.keys():
                gold_index = torch.tensor([iter], dtype=torch.long).to(device=constants.device)
                next_item = self.possible_next[(i, j, h)]
                del self.possible_next[(i, j, h)]
                break
        return gold_index, next_item

    def set_item_vec(self, vec, item):
        item = item.set_vector_rep(vec)
        self.repped_items[item] = item
        return self

    def score_item(self, item):
        self.scored_items[item] = item

    def update_chart(self, item):
        self.chart[(item.i,item.j,item.h)] = item
        return self

    def delete_from_chart(self, item):
        i, j, h = item.i, item.j, item.h
        del self.chart[(i, j, h)]
        # self.chart.chart.pop(item,None)
        return self

    def remove_from_bucket(self, item):
        self.bucket[item.l] -= 1
        self.bucket[item.r] -= 1
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

        self.has_head = {i:False for i in range(n)}

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
        other = item.l.h if item.l.h != item.h else item.r.h
        if self.has_head[other]:
            arc = None
        else:
         arc = (item.h, other)
        if add_rel:
            m = arc[1] - 1
            rel_made = self.rels[m]
            item.add_rel(rel_made)
        return arc, item

    def outgoing_ryan(self, item):
        """ Lazily Expand the Hypergraph """
        i, j, h = item.i, item.j, item.h

        # items to the left
        for k in range(0, i + 1):
            for g in range(k, i):
                if (k, i, g) in self.chart:
                    item_l = self.chart[(k, i, g)]
                    p = item_l.w * item.w
                    # attach left arc
                    yield ItemW(k, j, g, p * self.W[(h, g)], item_l, item)
                    # attach right arc
                    yield ItemW(k, j, h, p * self.W[(g, h)], item_l, item)

        # items to the right
        for k in range(j, self.n + 1):
            for g in range(j, k):
                if (j, k, g) in self.chart:
                    item_r = self.chart[(j, k, g)]
                    p = item.w * item_r.w
                    # attach left arc
                    yield ItemW(i, k, h, p * self.W[(g, h)], item, item_r)
                    # attach right arc
                    yield ItemW(i, k, g, p * self.W[(h, g)], item, item_r)


    def extend_pending(self,item):
        i,j,h = item.i, item.j, item.h
        ret = []
        for k in range(0, i + 1):
            for g in range(k, i):
                if (k, i, g) in self.chart:
                    item_l = self.chart[(k, i, g)]
                    item1 = Item(k, j, g, item_l, item)
                    arc1, item1 = self.make_arc(item1)
                    if arc1 is None:
                        self.chart[(item1.i,item1.j,item1.h)] = item1
                        ret.append(item1)
                    item2 = Item(k, j, h, item_l, item)
                    arc2, item2 = self.make_arc(item2)
                    if arc2 is None:
                        self.chart[(item2.i,item2.j,item2.h)] = item2
                        ret.append(item2)

        for k in range(j, self.n + 1):
            for g in range(j, k):
                if (j, k, g) in self.chart:
                    item_r = self.chart[(j, k, g)]
                    item_n1 = Item(i, k, h, item, item_r)
                    arcn1, item_n1 = self.make_arc(item_n1)

                    if arcn1 is None:
                        self.chart[(item_n1.i,item_n1.j,item_n1.h)] = item_n1
                        ret.append(item_n1)

                    item_n2 = Item(i, k, g, item, item_r)
                    arcn2, item_n2 = self.make_arc(item_n2)

                    if arcn2 is None:
                        self.chart[(item_n2.i,item_n2.j,item_n2.h)] = item_n2
                        ret.append(item_n2)
        return ret


    def outgoing(self, item, arc_prev):
        """ Lazily Expand the Hypergraph """
        i, j, h = item.i, item.j, item.h
        # items to the left
        # w = self.score(item)
        # self.arc(item)
        #all_arcs = []

        all_arcs = {}
        arcs = []
        #pend_increase = {}
        for k in range(0, i + 1):
            for g in range(k, i):
                if (k, i, g) in self.chart and self.chart[(k, i, g)] not in self.bucket:
                    item_l = self.chart[(k, i, g)]
                    item1 = Item(k, j, g, item_l, item)
                    arc1, item1 = self.make_arc(item1)
                    if arc1 is None:
                        self.chart[(item1.i,item1.j,item1.h)] = item1
                    elif arc1 not in arcs and arc1 not in arc_prev:
                        arcs.append(arc1)
                        all_arcs[(item1.i,item1.j,item1.h)] = item1
                    item2 = Item(k, j, h, item_l, item)
                    arc2, item2 = self.make_arc(item2)
                    if arc2 is None:
                        self.chart[(item2.i,item2.j,item2.h)] = item2
                    elif arc2 not in arcs and arc2 not in arc_prev:
                        all_arcs[(item2.i,item2.j,item2.h)] = item2
                        arcs.append(arc2)

        # items to the right
        for k in range(j, self.n + 1):
            for g in range(j, k):
                if (j, k, g) in self.chart and self.chart[(j, k, g)] not in self.bucket:

                    item_r = self.chart[(j, k, g)]

                    item_n1 = Item(i, k, h, item, item_r)
                    arcn1, item_n1 = self.make_arc(item_n1)

                    if arcn1 is None:
                        self.chart[(item_n1.i,item_n1.j,item_n1.h)] = item_n1
                    elif arcn1 not in arcs and arcn1 not in arc_prev:
                        arcs.append(arcn1)
                        all_arcs[(item_n1.i,item_n1.j,item_n1.h)] = item_n1
                    item_n2 = Item(i, k, g, item, item_r)
                    arcn2, item_n2 = self.make_arc(item_n2)

                    if arcn2 is None:
                        self.chart[(item_n2.i,item_n2.j,item_n2.h)] = item_n2
                    elif arcn2 not in arcs and arcn2 not in arc_prev:
                        all_arcs[(item_n2.i,item_n2.j,item_n2.h)] = item_n2
                        arcs.append(arcn2)

        return all_arcs, arcs


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
