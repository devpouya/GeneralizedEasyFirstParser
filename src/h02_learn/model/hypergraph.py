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
            # if not eager:
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
        if isinstance(item.l, ItemEager) and isinstance(item.r, ItemEager):
            if item.l.key == item.r.key:
                return (item.l.i, item.l.j)
            else:
                # it's a left arc
                return (j, item.l.j)
        else:
            if item.l == item.r:
                return (item.l, i)

    def update_pending(self, pending):
        pending_up = {}
        for item in pending.values():
            (i, j, h, b) = item.key
            if self.has_head[i] and b == 0:
                updated = ItemEager(i, j, h, 1, item.l, item.r)
                pending_up[updated.key] = updated
            else:
                pending_up[item.key] = item

        return pending_up

    def merge_pending(self, pending):
        nus = {}
        pending = self.update_pending(pending)
        # nus = self.rec_merge(pending,nus)
        # tmp = pending.copy()
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

    def reduce_tree(self, item, pending):
        # has to always be original pending
        pending = self.update_pending(pending)
        _, _, newly_merged = self.iterate_spans(item, pending, merge=True)

        pending[item.key] = item
        if isinstance(item.l, ItemEager):
            if item.l.key in pending.keys():
                del pending[item.l.key]
            if item.r.key in pending.keys():
                del pending[item.r.key]

        for item_n in newly_merged.values():
            pending = self.reduce_tree(item_n, pending)
        return pending

    def merge_pending(self, pending):
        # want to return all possible pendings
        pendings = []
        pending_orig = self.update_pending(pending)
        for item in pending_orig.values():
            pending_item = self.reduce_tree(item, pending_orig)
            pendings.append(pending_item)

        return pendings

    def num_mergable(self, pending):
        ret = 0
        for item in pending.values():
            (i, j, h, b) = item.key
            if b == 1:
                for k in range(i + 1):
                    if (k, i, k, 0) in pending.keys():
                        ret += 1
                    if (k, i, k, 1) in pending.keys():
                        ret += 1
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
                        item1 = ItemEager(k, j, k, 0, item_l, item)
                        # add to pending and call again
                        pending[item1.key] = item1
                        arcs_new, items_new, _ = self.iterate_spans(item1, pending, prev_arc=arcs)
                        arcs = arcs + arcs_new
                        items = {**items, **items_new}

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
                        item1 = ItemEager(k, j, k, 1, item_l, item)
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

    def axiom(self, i):
        return ItemMH4([i, i + 1], i, i)

    def is_axiom(self, item):
        (st, ss, sti, bf) = item.key
        return bf == st + 1

    def make_arc(self, item):
        pass

    def combine(self, item, pending):
        head = item.heads
        remaining_len = 4 - len(head)
        all_items = []
        if len(item.heads) == 1:
            for other in pending.values():
                if len(other.heads) < 4:
                    new_heads = sorted(other.heads + item.heads)
                    new_item = ItemMH4(new_heads,other,item)
                    all_items.append(new_item)
        else:
            # items to the left
            # item_l.key == (x0,x1,x2,h3)
            hm = head[0]
            hp = head[-1]
            for k in range(hm):
                #if not self.has_head[k]:
                if (k, hm, -1, -1) in pending.keys():
                    item_l = pending[(k, hm, -1, -1)]
                    if len(item.heads) == 3:
                        heads_new = sorted([k, hm, head[1], head[2]])
                        item_new = ItemMH4(heads_new, item_l, item)
                        #all_items[item_new.key] = item_new
                        all_items.append(item_new)
                    elif len(item.heads) == 2:
                        heads_new = sorted([k, hm, head[1]])
                        item_new = ItemMH4(heads_new, item_l, item)
                        #all_items[item_new.key] = item_new
                        all_items.append(item_new)

                    for j in range(k):
                        #if not self.has_head[j]:
                        if (j, k, hm, -1) in pending.keys():
                            item_l = pending[(j, k, hm, -1)]
                            if len(item.heads) == 2:
                                heads_new = sorted([j, k, hm, hp])
                                item_new = ItemMH4(heads_new, item_l, item)
                                #all_items[item_new.key] = item_new
                                all_items.append(item_new)

            # items to the right
            for k in range(hp, self.n):
                #if not self.has_head[k]:
                if (hp, k, -1, -1) in pending.keys():
                    item_r = pending[(hp, k, -1, -1)]
                    if len(item.heads) == 3:
                        heads_new = sorted([head[0], head[1], hp, k])
                        item_new = ItemMH4(heads_new, item, item_r)
                        #all_items[item_new.key] = item_new
                        all_items.append(item_new)
                    if len(item.heads) == 2:
                        heads_new = sorted([head[0], hp, k])
                        item_new = ItemMH4(heads_new, item, item_r)
                        #all_items[item_new.key] = item_new
                        all_items.append(item_new)
                    for j in range(k, self.n):
                        #if not self.has_head[j]:
                        if (hp, k, j, -1) in pending.keys():
                            item_r = pending[(hp, k, j, -1)]
                            if len(item.heads) == 2:
                                heads_new = sorted([head[0], hp, k, j])
                                item_new = ItemMH4(heads_new, item, item_r)
                                #all_items[item_new.key] = item_new
                                all_items.append(item_new)

        return all_items

    def link(self, item, prev_arcs):
        arcs = []
        all_items = []
        for i in item.heads:
            for j in item.heads:
                if j < self.n:
                    if not self.has_head[j]:
                        new_heads = item.heads.copy()
                        new_heads.remove(j)
                        new_item = ItemMH4(new_heads, item, item)
                        new_arc = (i,j)
                        if new_arc not in prev_arcs:
                            arcs.append(new_arc)
                            # all_items[new_item.key] = new_item
                            all_items.append(new_item)

        #for j in range(1, len(item.heads) - 1):
        #    for i in range(len(item.heads)):
        #        if i != j:
        #            new_heads = item.heads.copy()
        #            new_heads.pop(j)
        #            new_item = ItemMH4(new_heads, item, item)
        #            if not self.has_head[item.heads[j]]:
        #                new_arc = (item.heads[i], item.heads[j])
        #                if new_arc not in prev_arcs:
        #                    arcs.append(new_arc)
        #                    #all_items[new_item.key] = new_item
        #                    all_items.append(new_item)

        return arcs, all_items

    def list_to_dict(self, l,pending):
        for item in l:
            pending[item.key] = item
            if item.l.key in pending.keys():
                del pending[item.l.key]
            if item.r.key in pending.keys():
                del pending[item.r.key]

        return pending
    def recursive_iterate(self, pending, arcs, items,prev_arc,merge_levels):

        for item in pending.copy().values():
            arcs_new, items_new = self.link(item, arcs + prev_arc)
            arcs = arcs + arcs_new
            items = items + items_new
        for i, item in enumerate(pending.copy().values()):
            pending_copy = pending.copy()
            new_merged = self.combine(item, pending)
            if new_merged not in merge_levels:
                merge_levels.append(new_merged)
                for item_new in new_merged:
                    pending_copy[item_new.key] = item_new
                    if item_new.l.key in pending_copy.keys():
                        del pending_copy[item_new.l.key]
                    if item_new.r.key in pending_copy.keys():
                        del pending_copy[item_new.r.key]
                    arcs_new, items_new = self.link(item_new, arcs+prev_arc)
                    arcs = arcs + arcs_new
                    items = items + items_new

        for p_i in merge_levels:
            if len(p_i)>0:
                d_i = self.list_to_dict(p_i,pending.copy())
                arcs, items = self.recursive_iterate(d_i, arcs, items,prev_arc,merge_levels)

        return arcs, items

    def iterate_spans(self, item, pending, merge=False, prev_arc=None):
        arcs = []
        items = []
        nus = {}
        if merge:
            nus = self.combine(item, pending)
        else:
            #arcs, items = self.recursive_iterate(pending,arcs,items,prev_arc,[])
            for item in pending.values():
                possible_arcs, possible_items = self.link(item, arcs)
                arcs = arcs + possible_arcs
                items = items + possible_items
            #for i, item in enumerate(pending.values()):
            #    new_merged = self.combine(item, pending)
            #    pending_copy = pending.copy()
            #    for item_new in new_merged:
            #        pending_copy[item_new.key] = item_new
            #        if item_new.l.key in pending_copy.keys():
            #            del pending_copy[item_new.l.key]
            #        if item_new.r.key in pending_copy.keys():
            #            del pending_copy[item_new.r.key]


        return arcs, items, nus



    def clean_pending(self, pending):
        nu = {}
        for item in pending.values():
            h = item.heads
            updt = h.copy()
            for j in h:
                if j < self.n:
                    if self.has_head[j]:
                        updt.remove(j)
            if len(updt)>0:
                new_item = ItemMH4(updt,item.l,item.r)
                nu[new_item.key] = new_item
        return nu


    def merge_pending(self, pending):
        nus = []
        pending = self.clean_pending(pending)
        for item in pending.values():
            _, _, new_items = self.iterate_spans(item, pending, merge=True)
            nus = nus+new_items#{**nus, **new_items}

        for item in nus:#.values():
            pending[item.key] = item
            # if not eager:
            if item.l.key in pending.keys():
                del pending[item.l.key]
            if item.r.key in pending.keys():
                del pending[item.r.key]

        return pending

