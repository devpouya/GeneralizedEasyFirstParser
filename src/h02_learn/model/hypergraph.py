from .modules import Item,ItemHybrid, Chart
from .modules import ItemW
import torch
import torch.nn as nn

from utils import constants
import numpy as np
from collections import defaultdict

from termcolor import colored


class Hypergraph(object):

    def __init__(self, n):
        self.n = n
        self.has_head = {i: False for i in range(n)}
        self.right_children = {i: [i] for i in range(n)}
        self.left_children = {i: [i] for i in range(n)}

    def set_possible_next(self, items):
        for k in items.keys():
            self.possible_next[k] = items[k]
        return self
    def set_head(self, m):
        self.has_head[m] = True
        return self
    def remove_key_possible_next(self, key):
        del self.possible_next[key]
        return self

    def return_gold_next(self, items):
        gold_index, next_item = None, None
        # for item in self.possible_next.keys():
        #    print(item)
        for iter, (i, j, h) in enumerate(items.keys()):
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
        self.chart[(item.i, item.j, item.h)] = item
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

    def get_right_children_until(self, head, i):
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

    def get_left_children_from(self, head, i):
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

    def make_arc(self, item):
        pass

    def iterate_spans(self, item, pending, merge=False, prev_arc=None):
        arcs = []
        items = {}
        nus = {}
        return arcs, items, nus

    def merge_pending(self, pending):
        nus = {}
        for item in pending.values():
            _, _, new_items = self.iterate_spans(item, pending, merge=True)
            nus = {**nus, **new_items}

        for item in nus.values():
            pending[item.key] = item
            if item.l.key in pending.keys():
                del pending[item.l.key]
            if item.r.key in pending.keys():
                del pending[item.r.key]

        return pending


class ArcStandard(Hypergraph):

    def __init__(self, n):
        super().__init__(n)

    def axiom(self, i):
        return Item(i, i+1, i, i, i)

    def is_axiom(self, item):
        i, j, h = item.i, item.j, item.h
        if j == i + 1 and i == h and j == h + 1:
            return True
        else:
            return False

    def make_arc(self, item):
        other = item.l.h if item.l.h != item.h else item.r.h
        arc = (item.h, other)
        return arc

    def iterate_spans(self, item, pending, merge=False, prev_arc=None):
        (i, j, h) = item.key
        arcs = []
        items = {}
        nus = {}
        for k in range(0, i + 1):
            for g in range(k, i):
                if (k, i, g) in pending.keys():
                    item_l = pending[(k, i, g)]
                    if merge:
                        if self.has_head[g] and not self.has_head[h]:
                            nu = Item(k, j, h, item_l, item)
                        elif not self.has_head[g] and self.has_head[h]:
                            nu = Item(k, j, g, item_l, item)
                        elif self.has_head[g] and self.has_head[h]:
                            nu = Item(k, j, max(g, h), item_l, item)
                        else:
                            nu = None

                        if nu is not None:
                            nus[nu.key] = nu
                    else:
                        item1 = Item(k, j, h, item_l, item)
                        arc1 = self.make_arc(item1)
                        if arc1 not in prev_arc:
                            arcs.append(arc1)
                            items[item1.key] = item1

                        item2 = Item(k, j, g, item_l, item)
                        arc2 = self.make_arc(item2)
                        if arc2 not in prev_arc:
                            arcs.append(arc2)
                            items[item2.key] = item2

        # items to the right
        for k in range(j, self.n + 1):
            for g in range(j, k):
                if (j, k, g) in pending.keys():
                    item_r = pending[(j, k, g)]
                    if merge:
                        if self.has_head[g] and not self.has_head[h]:
                            nu = Item(i, k, h, item, item_r)
                        elif not self.has_head[g] and self.has_head[h]:
                            nu = Item(i, k, g, item, item_r)
                        elif self.has_head[g] and self.has_head[h]:
                            nu = Item(i, k, max(g, h), item, item_r)
                        else:
                            nu = None
                        if nu is not None:
                            nus[nu.key] = nu
                    else:
                        item1 = Item(i, k, h, item, item_r)
                        arc1 = self.make_arc(item1)
                        if arc1 not in prev_arc:
                            arcs.append(arc1)
                            items[item1.key] = item1
                        item2 = Item(i, k, g, item, item_r)
                        arc2 = self.make_arc(item2)
                        if arc2 not in prev_arc:
                            arcs.append(arc2)
                            items[item2.key] = item2

        return arcs, items, nus




class ArcEager(Hypergraph):
    def __init__(self, n, chart, rels):
        super().__init__(n, chart, rels)

    def outgoing(self, item):
        pass


class Hybrid(Hypergraph):
    def __init__(self, n):
        super().__init__(n)

    def axiom(self, i):
        return Item(i, i+1, i, i, i)

    def is_axiom(self, item):
        (i, j) = item.key
        if i==0 and j == 1:
            return True
        else:
            return False

    def make_arc(self, item):
        h = item.h
        if item.r.j == h:
            other = item.r.i
        elif item.l.i == h:
            other = item.l.j
        arc = (h, other)
        return arc

    def iterate_spans(self, item, pending, merge=False, prev_arc=None):
        (i, j, h) = item.key
        arcs = []
        items = {}
        nus = {}
        for k in range(0, i + 1):
            if (k, i, k) in pending.keys():
                item_l = pending[(k, i, k)]
                if merge:
                    if self.has_head[k]:
                        nu = Item(k, j, j, item_l, item)
                    else:
                        nu = None
                    if nu is not None:
                        nus[nu.key] = nu
                else:

                    item1 = Item(k, j, k, item_l, item)
                    arc1 = self.make_arc(item1)
                    if arc1 not in prev_arc:
                        arcs.append(arc1)
                        items[item1.key] = item1
                    if j < self.n:
                        # la_h has precondition j < |w|
                        item2 = Item(k, j, j, item_l, item)
                        arc2 = self.make_arc(item2)
                        if arc2 not in prev_arc:
                            arcs.append(arc2)
                            items[item2.key] = item2
            if (k,i,i) in pending.keys():
                item_l = pending[(k,i,i)]
                if merge:
                    if self.has_head[k]:
                        nu = Item(k, j, j, item_l, item)
                    else:
                        nu = None
                    if nu is not None:
                        nus[nu.key] = nu
                else:
                    item1 = Item(k, j, k, item_l, item)
                    arc1 = self.make_arc(item1)
                    if arc1 not in prev_arc:
                        arcs.append(arc1)
                        items[item1.key] = item1
                    if j < self.n:
                        item2 = Item(k, j, j, item_l, item)
                        arc2 = self.make_arc(item2)
                        if arc2 not in prev_arc:
                            arcs.append(arc2)
                            items[item2.key] = item2

        for k in range(j, self.n + 1):
            if (j, k, j) in pending.keys():
                item_r = pending[(j, k, j)]
                if merge:
                    if self.has_head[i]:
                        nu = Item(i, k, k, item, item_r)
                    else:
                        nu = None
                    if nu is not None:
                        nus[nu.key] = nu
                else:
                    item1 = Item(i, k, i, item, item_r)
                    arc1 = self.make_arc(item1)
                    if arc1 not in prev_arc:
                        arcs.append(arc1)
                        items[item1.key] = item1
                    if k < self.n:
                        item2 = Item(i, k, k, item, item_r)
                        arc2 = self.make_arc(item2)
                        if arc2 not in prev_arc:
                            arcs.append(arc2)
                            items[item2.key] = item2

            if (j,k,k) in pending.keys():
                item_r = pending[(j,k,k)]
                if merge:
                    if self.has_head[i]:
                        nu = Item(i, k, k, item, item_r)
                    else:
                        nu = None
                    if nu is not None:
                        nus[nu.key] = nu
                else:
                    item1 = Item(i, k, i, item, item_r)
                    arc1 = self.make_arc(item1)
                    if arc1 not in prev_arc:
                        arcs.append(arc1)
                        items[item1.key] = item1
                    if k < self.n:
                        item2 = Item(i, k, k, item, item_r)
                        arc2 = self.make_arc(item2)
                        if arc2 not in prev_arc:
                            arcs.append(arc2)
                            items[item2.key] = item2

        return arcs, items, nus


class MH4(Hypergraph):
    def __init__(self, n, chart, rels):
        super().__init__(n, chart, rels)

    def outgoing(self, item):
        pass
