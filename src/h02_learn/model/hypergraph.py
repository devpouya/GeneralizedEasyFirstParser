from .modules import ItemMH4, Chart, ItemTable
import itertools
import copy
from collections import defaultdict

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


class MH4(Hypergraph):
    def __init__(self, n):
        super().__init__(n)
        self.made_arcs = []
        self.derived_items = {}
        self.bucket = defaultdict(int)
        self.scored_items = {}
        self.item_table = ItemTable()

    def initialize_derived(self, dict):
        self.derived_items = copy.deepcopy(dict)

    def axiom(self, i):
        return ItemMH4([i, i + 1], i, i)

    def is_axiom(self, item):
        (st, ss, sti, bf) = item.key
        return bf == st + 1

    def make_arc(self, item):
        pass

    def combine2(self, item, pending):
        head = item.heads
        remaining_len = 4 - len(head)
        all_items = []
        if len(item.heads) == 1:
            val = item.heads[0]
            for other in pending.values():
                if len(other.heads) <= 4:
                    # do left and right
                    if other.heads[0] == val:
                        # other on right
                        new_heads = other.heads  # sorted(other.heads + item.heads)
                        new_item = ItemMH4(new_heads, item, other)
                        all_items.append(new_item)
                    elif other.heads[-1] == val:
                        new_heads = other.heads  # sorted(other.heads + item.heads)
                        new_item = ItemMH4(new_heads, other, item)
                        all_items.append(new_item)
        else:
            # items to the left
            # item_l.key == (x0,x1,x2,h3)
            hm = head[0]
            hp = head[-1]
            for k in range(hm):
                if not self.has_head[k]:
                    if (k, hm, -1, -1) in pending.keys():
                        item_l = pending[(k, hm, -1, -1)]
                        if len(item.heads) == 3:
                            heads_new = sorted(list(set([k, hm, head[1], head[2]])))
                            item_new = ItemMH4(heads_new, item_l, item)
                            # all_items[item_new.key] = item_new
                            all_items.append(item_new)
                        elif len(item.heads) == 2:
                            heads_new = sorted(list(set([k, hm, head[1]])))
                            item_new = ItemMH4(heads_new, item_l, item)
                            # all_items[item_new.key] = item_new
                            all_items.append(item_new)

                    for j in range(k):
                        if not self.has_head[j]:
                            if (j, k, hm, -1) in pending.keys():
                                item_l = pending[(j, k, hm, -1)]
                                if len(item.heads) == 2:
                                    heads_new = sorted(list(set([j, k, hm, hp])))
                                    item_new = ItemMH4(heads_new, item_l, item)
                                    # all_items[item_new.key] = item_new
                                    all_items.append(item_new)

            # items to the right
            for k in range(hp, self.n):
                if not self.has_head[k]:
                    if (hp, k, -1, -1) in pending.keys():
                        item_r = pending[(hp, k, -1, -1)]
                        if len(item.heads) == 3:
                            heads_new = sorted(list(set([head[0], head[1], hp, k])))
                            item_new = ItemMH4(heads_new, item, item_r)
                            # all_items[item_new.key] = item_new
                            all_items.append(item_new)
                        if len(item.heads) == 2:
                            heads_new = sorted(list(set([head[0], hp, k])))
                            item_new = ItemMH4(heads_new, item, item_r)
                            # all_items[item_new.key] = item_new
                            all_items.append(item_new)
                    for j in range(k, self.n):
                        if not self.has_head[j]:
                            if (hp, k, j, -1) in pending.keys():
                                item_r = pending[(hp, k, j, -1)]
                                if len(item.heads) == 2:
                                    heads_new = sorted(list(set([head[0], hp, k, j])))
                                    item_new = ItemMH4(heads_new, item, item_r)
                                    # all_items[item_new.key] = item_new
                                    all_items.append(item_new)

        return all_items

    def link2(self, item, prev_arcs):
        arcs = []
        all_items = []
        for i in item.heads:
            for j in item.heads:
                if j < self.n:
                    if i != j:
                        if not self.has_head[j]:
                            new_heads = item.heads.copy()
                            new_heads.remove(j)
                            new_item = ItemMH4(new_heads, item, item)
                            new_arc = (i, j)
                            if new_arc not in prev_arcs:
                                arcs.append(new_arc)
                                # all_items[new_item.key] = new_item
                                all_items.append(new_item)

        # for j in range(1, len(item.heads) - 1):
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

    def list_to_dict(self, l, pending):
        for item in l:
            pending[item.key] = item
            if item.l.key in pending.keys():
                del pending[item.l.key]
            if item.r.key in pending.keys():
                del pending[item.r.key]

        return pending

    def calculate_pending_help(self, pending, updated):
        pending_new = {}
        window_lower = updated - 3
        window_upper = updated + 4

        for item in pending.values():
            if updated in item.heads:
                new_heads = item.heads.copy()
                new_heads.remove(updated)
                if updated < self.n:
                    new_heads.append(updated + 1)
                new_heads = sorted(list(set(new_heads)))
                if 1 < len(new_heads) <= 4:
                    item_new = ItemMH4(new_heads, 0, 0)
                    pending_new[item_new.key] = item_new
            else:
                pending_new[item.key] = item
        return pending_new

    def calculate_pending_2(self):
        remaining = []
        pending = {}
        for i in range(self.n):
            if not self.has_head[i]:
                remaining.append(i)
        if len(remaining) <= 2:
            new_item = ItemMH4(remaining, 0, 0)
            pending[new_item.key] = new_item
            return pending
        combs = []
        for w in range(1, len(remaining) + 1):
            for i in range(len(remaining) - w + 1):
                l = remaining[i:i + w]
                if 1 < len(l) <= 4:
                    combs.append(l)
        for h in combs:
            new_item = ItemMH4(h, 0, 0)
            pending[new_item.key] = new_item
        return pending

    def possible_arcs_items(self, is_easy_first=True):
        # do all possible shifts

        possible_derivations = copy.deepcopy(self.derived_items)

        for item in self.derived_items.values():
            last_head = item.last_head
            if last_head <= self.n:
                    new_item = ItemMH4([last_head, last_head + 1], item, item)
                    possible_derivations[new_item.key] = new_item
        # do all possible combines
        combined = {}
        new_item_list = []
        for item_l in possible_derivations.values():
            for item_r in possible_derivations.values():
                    if item_l.last_head == item_r.first_head and not item_l.key in combined and not item_r.key in combined:
                        tmp = set(item_r.heads + item_l.heads)
                        if len(tmp) <= 4:
                            new_item = ItemMH4(list(tmp), item_l, item_r)
                            #possible_derivations[new_item.key] = new_item
                            new_item_list.append(new_item)
                            self.item_table[new_item] = new_item
                            combined[item_l.key] = True
                            combined[item_r.key] = True
        # update possible
        for item_key in combined.keys():
            possible_derivations.pop(item_key, None)
        for item in new_item_list:
            possible_derivations[item.key] = item

        # compute possible links
        possible_arcs = []
        final_derived_items = []
        for item in possible_derivations.values():
                heads = item.heads
                all_combinations = list(itertools.product(heads, heads))
                for (i, j) in all_combinations:
                    if i!= j:
                        if 1 <= i <= self.n and 1 < j < self.n:
                            possible_arcs.append((i, j))
                            new_heads = heads.copy()
                            new_heads.remove(j)
                            new_item = ItemMH4(new_heads, item, item)
                            final_derived_items.append(new_item)
        return possible_arcs, final_derived_items

    def derive_item(self, arc, item):
        self.derived_items[item.key] = item
        self.bucket[item.l] += 1
        self.bucket[item.r] += 1
        j = arc[1]
        for item in self.derived_items.values():
            if j in item.heads:
                item.pop(j)
        self.set_head(arc[1])

    def calculate_pending(self):
        remaining = []
        pending = {}
        for i in range(self.n):
            if not self.has_head[i]:
                remaining.append(i)
        if len(remaining) <= 4:
            new_item = ItemMH4(remaining, 0, 0)
            pending[new_item.key] = new_item
            return pending

        for i in range(len(remaining) - 4):
            new_item = ItemMH4(remaining[i:i + 4], 0, 0)
            pending[new_item.key] = new_item
        return pending

    def iterate_spans(self, item, pending, merge=False, prev_arc=None):
        arcs = []
        items = []
        nus = {}
        if merge:
            nus = self.combine(item, pending)
        else:

            for item in pending.values():
                possible_arcs, possible_items = self.link(item, arcs)
                for (pa, pi) in zip(possible_arcs, possible_items):
                    if pa not in arcs:
                        arcs.append(pa)
                        items.append(pi)
                # arcs = arcs + possible_arcs
                # items = items + possible_items

        return arcs, items, nus

    def combine(self, item, pending):
        if len(item.heads) == 3:
            other_len = [2]
        elif len(item.heads) == 2:
            other_len = [2, 3]
        else:
            return []
        all_items = []
        # item is the left item
        for other in pending.values():
            if len(other.heads) in other_len:
                if other.heads[0] < item.heads[-1] and len(item.heads) + len(other.heads) <= 4:
                    complete_gap = True
                    for i in range(other.heads[0] + 1, item.heads[-1]):
                        if self.has_head[i]:
                            continue
                        else:
                            complete_gap = False
                            break
                    if complete_gap:
                        new_heads = sorted(list(set(other.heads + item.heads)))
                        new_item = ItemMH4(new_heads, other, item)
                        all_items.append(new_item)
                elif other.heads[0] > item.heads[-1] and len(item.heads) + len(other.heads) <= 4:
                    complete_gap = True
                    for i in range(item.heads[-1] + 1, other.heads[0]):
                        if self.has_head[i]:
                            continue
                        else:
                            complete_gap = False
                            break
                    if complete_gap:
                        new_heads = sorted(list(set(other.heads + item.heads)))
                        new_item = ItemMH4(new_heads, item, other)
                        all_items.append(new_item)
                if other.heads[0] == item.heads[-1]:
                    new_heads = item.heads + other.heads[1:]
                    new_item = ItemMH4(new_heads, item, other)
                    all_items.append(new_item)
                if other.heads[-1] == item.heads[0]:
                    new_heads = other.heads + item.heads[1:]
                    new_item = ItemMH4(new_heads, other, item)
                    all_items.append(new_item)
        return all_items

    def link(self, item, prev_arcs):
        all_items = []
        arcs = []
        if len(item.heads) > 2:
            for j in range(len(item.heads) - 1):
                for i in range(len(item.heads)):
                    if i != j:
                        if not self.has_head[item.heads[j]]:
                            if item.heads[j] < self.n and item.heads[i] < self.n:
                                if not self.has_head[item.heads[i]]:
                                    new_heads = item.heads.copy()
                                    new_heads.pop(j)
                                    new_item = ItemMH4(new_heads, item, item)
                                    new_arc = (item.heads[i], item.heads[j])
                                    if new_arc not in prev_arcs:
                                        arcs.append(new_arc)
                                        all_items.append(new_item)
                                if not self.has_head[item.heads[i]]:
                                    new_heads = item.heads.copy()
                                    new_heads.pop(i)
                                    new_item = ItemMH4(new_heads, item, item)
                                    new_arc = (item.heads[j], item.heads[i])
                                    if new_arc not in prev_arcs:
                                        arcs.append(new_arc)
                                        all_items.append(new_item)
        else:
            new_arc_1 = (item.heads[1], item.heads[0])
            new_item_1 = ItemMH4([item.heads[1]], item, item)
            if new_arc_1 not in prev_arcs:
                arcs.append(new_arc_1)
                all_items.append(new_item_1)
            new_arc_2 = (item.heads[0], item.heads[1])
            new_item_2 = ItemMH4([item.heads[0]], item, item)
            if new_arc_2 not in prev_arcs:
                arcs.append(new_arc_2)
                all_items.append(new_item_2)

        return arcs, all_items

    def clean_pending(self, pending, dep):
        nu = {}
        for item in pending.values():
            h = item.heads
            if dep in h:
                h.remove(dep)
                if len(h) >= 2:
                    new_item = ItemMH4(h, item.l, item.r)
                    nu[new_item.key] = new_item
            else:
                nu[item.key] = item

        return nu

    def merge_pending(self, pending):
        nus = []
        # pending = self.clean_pending(pending)
        for item in pending.values():
            _, _, new_items = self.iterate_spans(item, pending, merge=True)
            nus = nus + new_items  # {**nus, **new_items}

        for item in nus:  # .values():
            pending[item.key] = item
            # if not eager:
            if item.l.key in pending.keys():
                del pending[item.l.key]
            if item.r.key in pending.keys():
                del pending[item.r.key]

        return pending
