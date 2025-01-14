from .modules import ItemMH4



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


def item_to_transition_state(item):
    heads = item.heads
    # print(heads)
    # heads = heads[heads != -1]
    # print(heads)
    sigma_1 = heads[:-1]
    beta_1 = heads[-1]
    return sigma_1, beta_1

class MH4(Hypergraph):
    def __init__(self, n, is_easy_first = True):
        super().__init__(n)
        self.made_arcs = []
        self.is_easy_first = is_easy_first
        if not is_easy_first:
            self.link = self.link_shift_reduce
        else:
            self.link = self.link_easy_first

    def axiom(self, i):
        return ItemMH4([i, i + 1], i, i)

    def is_axiom(self, item):
        (st, ss, sti, bf) = item.key
        return bf == st + 1

    def shift_reduce_arc_from_item(self, item):
        sigma_item, beta_item = item_to_transition_state(item)
        arcs = []
        items = []
        # print(item.heads)
        la_arc = (sigma_item[-1], beta_item)
        new_heads = item.heads.copy()
        new_heads_1 = new_heads
        new_heads_1.remove(sigma_item[-1])
        item_la_arc = ItemMH4(new_heads_1, item, item)
        # if self.is_item_legal_shift_reduce(item_la_arc):
        arcs.append(la_arc)
        items.append(item_la_arc)
        if len(sigma_item) >= 2:
            la_prime_arc = (sigma_item[-1], sigma_item[-2])
            new_heads_2 = item.heads.copy()
            new_heads_2.remove(sigma_item[-2])
            item_la_prime_arc = ItemMH4(new_heads_2, item, item)

            la_2_arc = (beta_item, sigma_item[-2])
            new_heads_3 = item.heads.copy()
            new_heads_3.remove(sigma_item[-2])
            item_la_2_arc = ItemMH4(new_heads_3, item, item)

            ra_arc = (sigma_item[-2], sigma_item[-1])
            new_heads_4 = item.heads.copy()
            new_heads_4.remove(sigma_item[-1])
            item_ra_arc = ItemMH4(new_heads_4, item, item)

            # if self.is_item_legal_shift_reduce(item_la_prime_arc):
            arcs.append(la_prime_arc)
            items.append(item_la_prime_arc)
            # if self.is_item_legal_shift_reduce(item_la_2_arc):
            arcs.append(la_2_arc)
            items.append(item_la_2_arc)
            # if self.is_item_legal_shift_reduce(item_ra_arc):
            arcs.append(ra_arc)
            items.append(item_ra_arc)

        if len(sigma_item) >= 3:
            ra_prime_arc = (sigma_item[-3], sigma_item[-2])
            new_heads_5 = item.heads.copy()
            new_heads_5.remove(sigma_item[-2])
            item_ra_prime_arc = ItemMH4(new_heads_5, item, item)

            ra_2_arc = (sigma_item[-3], sigma_item[-1])
            new_heads_6 = item.heads.copy()
            new_heads_6.remove(sigma_item[-1])
            item_ra_2_arc = ItemMH4(new_heads_6, item, item)
            # if self.is_item_legal_shift_reduce(item_ra_prime_arc):
            arcs.append(ra_prime_arc)
            items.append(item_ra_prime_arc)
            # if self.is_item_legal_shift_reduce(item_ra_2_arc):
            arcs.append(ra_2_arc)
            items.append(item_ra_2_arc)

        return arcs, items

    def link_shift_reduce(self, item, prev_arcs):

        arcs, items = self.shift_reduce_arc_from_item(item)
        if item.heads[-1] + 1 <= self.n:
            # can be pushed
            new_heads = item.heads
            new_heads.append(item.heads[-1] + 1)
            if len(new_heads) > 4:
                new_heads = new_heads[-4:]
            new_item = ItemMH4(new_heads, item, item)
            arcs2, items2 = self.shift_reduce_arc_from_item(new_item)
            arcs = arcs + arcs2
            items = items + items2

        return arcs, items

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
        combs = []
        for i in range(len(remaining)-4):
            new_item = ItemMH4(remaining[i:i+4], 0, 0)
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

    def link_easy_first(self, item, prev_arcs):
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