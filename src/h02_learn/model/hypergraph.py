from .modules import Item, ItemHybrid, ItemMH4, ItemEager, Chart
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

    def set_head(self, m):
        self.has_head[m] = True
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

    def update_pending(self, pending):
        return pending

    def merge_pending(self, pending):
        nus = {}
        for item in pending.values():
            _, _, new_items = self.iterate_spans(item, pending, merge=True)
            nus = {**nus, **new_items}

        for item in nus.values():
            pending[item.key] = item
            #if not eager:
            if item.l.key in pending.keys():
                del pending[item.l.key]
            if item.r.key in pending.keys():
                del pending[item.r.key]

        return pending


class ArcStandard(Hypergraph):

    def __init__(self, n):
        super().__init__(n)

    def axiom(self, i):
        return Item(i, i + 1, i, i, i)

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


class Hybrid(Hypergraph):
    def __init__(self, n):
        super().__init__(n)

    def axiom(self, i):
        return Item(i, i + 1, i, i, i)

    def is_axiom(self, item):
        (i, j) = item.key
        if i == 0 and j == 1:
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
            if (k, i, i) in pending.keys():
                item_l = pending[(k, i, i)]
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

            if (j, k, k) in pending.keys():
                item_r = pending[(j, k, k)]
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


class ArcEager(Hypergraph):
    def __init__(self, n):
        super().__init__(n)

    def axiom(self, i):
        # if i > 0:
        #    yield ItemEager(i, i + 1, i, 0, i, i)
        #    yield ItemEager(i, i + 1, i, 1, i, i)
        # else:
        #    yield ItemEager(i, i + 1, i, 0, i, i)
        return ItemEager(i, i + 1, i, 0, i, i)

    def is_axiom(self, item):
        (i, j, h, b) = item.key
        if i == 0:
            return i + 1 == j and h == i and b == 0
        return i + 1 == j and h == i

    def make_arc(self, item):
        (i, j, h, b) = item.key
        if isinstance(item.l,ItemEager) and isinstance(item.r,ItemEager):
            if item.l.key == item.r.key:
                return (item.l.i,item.l.j)
            else:
                # it's a left arc
                return (j, item.l.j)
        else:
            if item.l == item.r:
                return (item.l,i)

    def update_pending(self, pending):
        pending_up = {}
        for item in pending.values():
            (i,j,h,b) = item.key
            if self.has_head[i] and b == 0:
                updated = ItemEager(i,j,h,1,item.l,item.r)
                pending_up[updated.key]=updated
            else:
                pending_up[item.key] = item

        return pending_up

    def merge_pending(self, pending):
        nus = {}
        pending = self.update_pending(pending)
        #nus = self.rec_merge(pending,nus)
        #tmp = pending.copy()
        for item in pending.values():
            _,_,new_items = self.iterate_spans(item,pending,merge=True)
            nus = {**nus,**new_items}


        for item in nus.values():
            pending[item.key] = item
            if item.l.key in pending.keys():
                del pending[item.l.key]
            if item.r.key in pending.keys():
                del pending[item.r.key]
        return pending

    def reduce_tree(self, item, pending):
        # has to always be original pending
        pending = self.update_pending(pending)
        _,_,newly_merged = self.iterate_spans(item,pending,merge=True)

        pending[item.key] = item
        if isinstance(item.l,ItemEager):
            if item.l.key in pending.keys():
                del pending[item.l.key]
            if item.r.key in pending.keys():
                del pending[item.r.key]

        for item_n in newly_merged.values():
             pending = self.reduce_tree(item_n,pending)
        return pending

    def merge_pending(self, pending):
        # want to return all possible pendings
        pendings = []
        pending_orig = self.update_pending(pending)
        for item in pending_orig.values():
            pending_item = self.reduce_tree(item,pending_orig)
            pendings.append(pending_item)


        return pendings

    def num_mergable(self, pending):
        ret = 0
        for item in pending.values():
            (i,j,h,b) = item.key
            if b == 1:
                for k in range(i+1):
                    if (k,i,k,0) in pending.keys():
                        ret+=1
                    if (k,i,k,1) in pending.keys():
                        ret+=1
        return ret


    def iterate_spans(self, item, pending, merge=False, prev_arc=None):
        (i, j, h, b) = item.key
        arcs = []
        items = {}
        nus = {}
        # right arc eager
        if not merge and j < self.n:
            item_new = ItemEager(j, j + 1, j, 1, item, item)
            arc_new = self.make_arc(item_new)
            if arc_new not in prev_arc and not self.has_head[arc_new[1]]:
                arcs.append(arc_new)
                items[item_new.key] = item_new

        for k in range(0, i + 1):
            if (k, i, k, 0) in pending.keys():
                item_l = pending[(k, i, k, 0)]
                if merge:
                    if b == 1:
                        # reduce, have to do this in the merge step
                        item2 = ItemEager(k, j, k, 0, item_l, item)
                        nus[item2.key] = item2
                else:
                    if b == 0 and j < self.n:
                        # left-arc-eager
                        item1 = ItemEager(k, j, k, 0, item_l, item)
                        arc1 = self.make_arc(item1)
                        if arc1 not in prev_arc and not self.has_head[arc1[1]]:
                            arcs.append(arc1)
                            items[item1.key] = item1
                    elif b == 1:
                        item1 = ItemEager(k,j,k,0,item_l,item)
                        # add to pending and call again
                        pending[item1.key] = item1
                        arcs_new,items_new,_ = self.iterate_spans(item1,pending,prev_arc=arcs)
                        arcs = arcs+arcs_new
                        items = {**items,**items_new}

            if (k, i, k, 1) in pending.keys():
                item_l = pending[(k, i, k, 1)]
                if merge:
                    if b == 1:
                        # reduce, have to do this in the merge step
                        item2 = ItemEager(k, j, k, 1, item_l, item)
                        nus[item2.key] = item2
                else:
                    if b == 0 and j < self.n:
                        # left-arc-eager
                        item1 = ItemEager(k, j, k, 1, item_l, item)
                        arc1 = self.make_arc(item1)
                        if arc1 not in prev_arc and not self.has_head[arc1[1]]:
                            arcs.append(arc1)
                            items[item1.key] = item1
                    elif b == 1:
                        item1 = ItemEager(k,j,k,1,item_l,item)
                        pending[item1.key] = item1
                        arcs_new, items_new, _ = self.iterate_spans(item1, pending, prev_arc=arcs)
                        arcs = arcs + arcs_new
                        items = {**items, **items_new}

        for k in range(j, self.n + 1):
            if (j, k, j, 0) in pending.keys():
                item_r = pending[(j, k, j, 0)]
                if merge:
                    continue
                else:
                    if k < self.n:
                        item1 = ItemEager(i, k, i, b, item, item_r)
                        arc1 = self.make_arc(item1)
                        if arc1 not in prev_arc and not self.has_head[arc1[1]]:
                            arcs.append(arc1)
                            items[item1.key] = item1
            if (j, k, j, 1) in pending.keys():
                item_r = pending[(j, k, j, 1)]
                if merge:
                    item1 = ItemEager(i, k, i, b, item, item_r)
                    nus[item1.key] = item1
                else:
                    item1 = ItemEager(i, k, i, b, item, item_r)
                    pending[item1.key] = item1
                    arcs_new, items_new, _ = self.iterate_spans(item1, pending, prev_arc=arcs)
                    arcs = arcs + arcs_new
                    items = {**items, **items_new}

        return arcs, items, nus


class MH4(Hypergraph):
    def __init__(self, n):
        super().__init__(n)
        self.id2act = {
            0: self.la,
            1: self.ra,
            2: self.lap,
            3: self.rap,
            4: self.la2,
            5: self.ra2
        }
        self.act2id = {
            "la": 0,
            "ra": 1,
            "lap": 2,
            "rap": 3,
            "la2": 4,
            "ra2": 5
        }

    # def axiomold(self, i):
    #    return ItemMH4(i, None, None, i + 1, i)

    def axiom(self, i):
        return ItemMH4([i, i + 1], i, i)

    def is_axiom(self, item):
        (st, ss, sti, bf) = item.key
        return bf == st + 1

    def make_arc(self, item):
        pass

    def la(self, item):
        (s_top, s_second, s_third, b_front) = item.key
        return ItemMH4(s_second, s_third, None, b_front, item), (b_front, s_top)

    def ra(self, item):
        (s_top, s_second, s_third, b_front) = item.key
        return ItemMH4(s_second, s_third, None, b_front, item), (s_second, s_top)

    def lap(self, item):
        (s_top, s_second, s_third, b_front) = item.key
        return ItemMH4(s_top, s_third, None, b_front, item), (s_top, s_second)

    def rap(self, item):
        (s_top, s_second, s_third, b_front) = item.key
        return ItemMH4(s_top, s_third, None, b_front, item), (s_third, s_second)

    def la2(self, item):
        (s_top, s_second, s_third, b_front) = item.key
        return ItemMH4(s_top, s_third, None, b_front, item), (b_front, s_second)

    def ra2(self, item):
        (s_top, s_second, s_third, b_front) = item.key
        return ItemMH4(s_second, s_third, None, b_front, item), (s_third, s_top)

    def iterate_spans(self, item, pending, merge=False, prev_arc=None):
        (a, b, c, d) = item.key
        heads = item.heads
        len_heads = item.len
        nus = {}
        items = {}
        arcs = []
        if merge:
            for other in pending.values():
                other_heads = other.heads
                len_other = other.len
                # need to check for head first
                #
                # other is right item
                if len_other <= 4 and other_heads[0] == heads[-1]:
                    new_heads = list(set(heads + other_heads))
                    new_item = ItemMH4(new_heads, item, pending[other.key])
                    nus[new_item.key] = new_item

                # other is left item
                if len_heads <= 4 and other_heads[-1] == heads[0]:
                    new_heads = list(set(other_heads + other_heads))
                    new_item = ItemMH4(new_heads, pending[other.key], item)
                    nus[new_item.key] = new_item

        else:
            for i in range(len_heads + 1):
                for j in range(1, len_heads):
                    if i != j:
                        arc = (heads[i], heads[j])
                        new_head = heads.copy()
                        new_head.pop(j)
                        new_item = ItemMH4(new_head, item, item)
                        items[new_item.key] = new_item
                        arcs.append(arc)

        return arcs, items, nus

    def iterate_spans_old_neveragoodsignwhenfunctionnamesarelikethis(self, item, pending, merge=False, prev_arc=None):
        (h3, h2, h1, h4) = item.key
        item_heads = item.heads
        items = {}
        nus = {}
        arcs = []
        if merge:
            for other in pending.values():
                heads = other.heads
                if (heads[0] == item_heads[-1] and heads[-1] <= 4) or (heads[-1] == item_heads[0] and heads[0] <= 4):
                    new_heads = list(set(heads + item_heads))
                    b = new_heads[-1]
                    if len(new_heads) >= 4:
                        third = new_heads[0]
                        second = new_heads[1]
                        top = new_heads[2]
                    elif len(new_heads) == 3:
                        third = None
                        second = new_heads[0]
                        top = new_heads[1]
                    elif len(new_heads) == 2:
                        top = new_heads[0]
                        second = None
                        third = None
                    if heads[0] == item_heads[-1] and heads[-1] <= 4:
                        new_item = ItemMH4(top, second, third, b, item, pending[other.key])
                    else:
                        new_item = ItemMH4(top, second, third, b, pending[other.key], item)
                    nus[new_item.key] = new_item

            return None, None, nus
        # la: b_front --> s_top
        # s_top,None,None,b_front --> None, None, None, b_front

        # ra: s_scond --> s_top
        # s_top, s_second, None, None --> s_second,None,None,None

        # la': s_top --> s_second
        # s_top, s_second, None, None --> s_top, None, None, b_front

        # ra': s_third --> s_second
        # stop, ssec, sthird, None --> s_top, s_third, None, None

        # la2: b_front --> s_second
        # s_top,s_second,None,b_front --> s_top, None,None,b_front

        # ra2: s_third --> s_top
        # top,sec,third,None --> second,third,None,None
        # illegal_actions = []
        # if h4 is None:
        #    illegal_actions.append(self.act2id["la2"])
        #    illegal_actions.append(self.act2id["la"])
        # if h3 is None:
        #    illegal_actions.append(self.act2id["ra2"])
        #    illegal_actions.append(self.act2id["rap"])
        # if h2 is None:
        #    illegal_actions.append(self.act2id["ra2"])
        #    illegal_actions.append(self.act2id["rap"])
        #    illegal_actions.append(self.act2id["ra"])
        #    illegal_actions.append(self.act2id["la2"])
        # if h1 is None:
        #    illegal_actions.append(self.act2id["ra2"])
        #    illegal_actions.append(self.act2id["rap"])
        # all_actions = set(list((range(6))))
        # illegal_actions = set(illegal_actions)
        # legal_actions = list(all_actions.difference(illegal_actions))
        # for act in legal_actions:
        #    item_new, arc = self.id2act[act](item)
        #    items[item_new.key] = item_new
        #    arcs.append(arc)

        if h3 is not None and h4 is not None:
            item_new, arc = self.la(item)
            items[item_new.key] = item_new
            arcs.append(arc)
        if h3 is not None and h2 is not None:
            item_new1, arc1 = self.ra(item)
            items[item_new1.key] = item_new1
            arcs.append(arc1)
            item_new2, arc2 = self.lap(item)
            items[item_new2.key] = item_new2
            arcs.append(arc2)
        if h3 is not None and h2 is not None and h4 is not None:
            item_new3, arc3 = self.la2(item)
            items[item_new3.key] = item_new3
            arcs.append(arc3)
        if h1 is not None and h3 is not None and h2 is not None:
            item_new4, arc4 = self.rap(item)
            items[item_new4.key] = item_new4
            arcs.append(arc4)
            item_new5, arc5 = self.ra2(item)
            items[item_new5.key] = item_new5
            arcs.append(arc5)

        return arcs, items, nus
