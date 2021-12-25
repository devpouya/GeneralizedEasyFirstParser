from utils import constants
from utils.data_structures import BaseStack, BaseBuffer, Chart, BaseAgenda, BaseItem
import numpy as np
import random
from collections import defaultdict, OrderedDict


def break_two_way(heads):
    for i in range(len(heads)):
        for j in range(len(heads)):
            if heads[i] == j and heads[j] == i:
                heads[i] = i
    return heads


def get_arcs(word2head):
    arcs = []
    # good luck with this lol
    for word in word2head:
        arcs.append((word2head[word], word))

    return arcs


def has_head(node, arcs):
    for (u, v) in arcs:
        if v == node:
            return True
    return False


def get_labeled_arcs(word2head):
    arcs = []
    # good luck with this lol
    for word in word2head:
        arcs.append((word2head[word][0], word, word2head[word][1]))

    return arcs


def get_sentnece_root(heads):
    arcs = get_arcs(heads)
    roots = np.zeros((len(heads)))
    for (a, b) in arcs:
        # if it has an incoming arc, it's not a root
        roots[b] = 1
    return np.argwhere(roots == 0).item()


def have_completed_expected_children(item, true_arcs, built_arcs):
    needed_arcs = []
    for (a, b) in true_arcs:
        if a == item and b != item:
            needed_arcs.append((a, b))
    if len(needed_arcs) == 0:
        return True
    for arc in needed_arcs:
        if arc in built_arcs:
            continue
        else:
            return False
    return True


def neighbors(edges, node):
    neighbors = []
    for (u, v) in edges:
        if u == node:
            neighbors.append(u)
        elif v == node:
            neighbors.append(v)
    return neighbors


def from_node(edges, node, visited, rec_stack):
    visited[node] = True
    rec_stack[node] = True
    for neighbor in neighbors(edges, node):
        if not visited[neighbor]:
            if from_node(edges, neighbor, visited, rec_stack):
                return True
            elif rec_stack[neighbor]:
                return True

    rec_stack[node] = False
    return False


def contains_cycles(heads):
    arc_list = get_arcs(heads)
    visited = [False] * len(heads)
    rec_stack = [False] * len(heads)
    for node in range(len(heads)):
        if not visited[node]:
            if from_node(arc_list, node, visited, rec_stack):
                return True
    return False


def build_hypergraph_arc_standard(ordered_arcs, n):
    derived_items = [(i, i + 1, i) for i in range(n)]
    b_hypergraph = []
    for (u, v) in ordered_arcs:
        derived_items = sorted(derived_items, key=lambda x: x[2])
        item_l, item_r, new_item = None, None, None
        for (i1, j1, h1) in derived_items:
            for (i2, j2, h2) in derived_items:
                if h1 == u and h2 == v:
                    bigger = max(u, v)
                    if bigger == v:
                        if j1 == i2:
                            new_item = (i1, j2, u)
                            derived_items.append(new_item)
                            item_l = (i1, j1, h1)
                            item_r = (i2, j2, h2)
                            break
                    else:
                        if j2 == i1:
                            new_item = (i2, j1, u)
                            derived_items.append(new_item)
                            item_l = (i2, j2, h2)
                            item_r = (i1, j1, h1)

        b_hypergraph.append((item_l, item_r, new_item))
        derived_items.remove(item_l)
        derived_items.remove(item_r)
    return b_hypergraph


def build_hypergraph_hybrid(ordered_arcs, n):
    derived_items = [(i, i + 1, i) for i in range(n)]
    b_hypergraph = []
    for (u, v) in ordered_arcs:
        derived_items = sorted(derived_items, key=lambda x: x[2])
        item_l, item_r, new_item = None, None, None
        # i1,j1,i1
        for (i1, j1, h1) in derived_items:
            # i2,j2,i2
            for (i2, j2, h2) in derived_items:
                # lah: j2 --> j1; j1 == i2
                # ra: i1 --> i2; i2 == j1
                # lah: j2 --> i2
                # ra: i1 --> j1
                if j2 == u and j1 == v and j1 == i2:
                    # lah
                    new_item = (i1, j2, j2)
                    derived_items.append(new_item)
                    item_l = (i1, j1, h1)
                    item_r = (i2, j2, h2)
                    break
                elif i1 == u and i2 == v and i2 == j1:
                    # ra
                    new_item = (i1, j2, i1)
                    derived_items.append(new_item)
                    item_l = (i1, j1, h1)
                    item_r = (i2, j2, h2)
                    break

        b_hypergraph.append((item_l, item_r, new_item))
        derived_items.remove(item_l)
        derived_items.remove(item_r)
    return b_hypergraph


def pick_random_adjacent(true_arcs, pending):
    candidates = []
    for i in range(len(pending) - 1):
        if (pending[i], pending[i + 1]) in true_arcs:
            candidates.append((pending[i], pending[i + 1], pending[i]))
        if (pending[i + 1], pending[i]) in true_arcs:
            candidates.append((pending[i], pending[i + 1], pending[i + 1]))
    (i, j, h) = random.choice(candidates)

    return random.choice(candidates)


def has_all_children(true_arcs, built_arcs, item):
    for (u, v) in true_arcs:
        if u == item:
            if (u, v) in built_arcs.values():
                continue
            else:
                return False
        else:
            continue
    return True


def build_easy_first(sentence, word2head,relations,true_arcs):
    #true_arcs = get_arcs(word2head)
    pending = sentence.copy()  # BaseBuffer(sentence)
    built_arcs = {}
    counter = 0

    #true_arcs.remove((0, 0))

    #pending = pending[1:]
    while len(pending) > 0:
        n = len(pending) - 1
        if n == 0:
            built_arcs[counter] = (0, pending[0])
            counter += 1
            pending.pop(0)
            continue

        for i in range(n):
            precondition_i = has_all_children(true_arcs, built_arcs, pending[i + 1])
            precondition_inext = has_all_children(true_arcs, built_arcs, pending[i])
            if (pending[i], pending[i + 1]) in true_arcs and precondition_i:
                built_arcs[counter] = (pending[i], pending[i + 1])
                counter += 1
                pending[i + 1] = -1
            elif (pending[i + 1], pending[i]) in true_arcs and precondition_inext:
                built_arcs[counter] = (pending[i + 1], pending[i])
                counter += 1
                pending[i] = -1

        for item in pending:
            if item == -1:
                pending.remove(-1)
    ordered_arcs = list(built_arcs.values())  # []
    cond = set(ordered_arcs) == set(true_arcs)
    ordered_arcs.remove((0, 0))
    return ordered_arcs, [],cond


def find_corresponding_relation(labeled_arcs, arc):
    for (u, v, r) in labeled_arcs:
        if arc == (u, v):
            return r


def item_mh4_oracle(sentence, word2head, relations,true_arcs):
    stack = []
    buffer = sentence.copy()

    #true_arcs = get_arcs(word2head)
    built_arcs = []
    action_history = []

    arcs_sorted = sorted(true_arcs, key=lambda tup: tup[1])[1:]
    labeled_arcs = []
    for i, (u, v) in enumerate(arcs_sorted):
        labeled_arcs.append((u, v, relations[i]))

    relations_in_order = []


    this_step = 0
    while len(buffer) > 0 or len(stack) > 1:
        this_step+=1
        if this_step > 2*len(sentence)-1:
            break
        if len(stack) > 0:
            top = stack[-1]

            if len(buffer)>=1:

                if (buffer[0],top) in true_arcs and have_completed_expected_children(top,true_arcs,built_arcs):

                    built_arcs.append((buffer[0],top))
                    relations_in_order.append(find_corresponding_relation(labeled_arcs, (buffer[0], top)))
                    stack.pop(-1)
                    action_history.append(constants.left_arc_eager)
                    continue
            if len(stack) >= 2:
                second = stack[-2]
                if len(buffer)>=1:
                    if (buffer[0], top) in true_arcs and have_completed_expected_children(top, true_arcs, built_arcs):
                        built_arcs.append((buffer[0], top))
                        relations_in_order.append(find_corresponding_relation(labeled_arcs, (buffer[0], top)))
                        stack.pop(-1)
                        action_history.append(constants.left_arc_eager)
                        continue
                    if (buffer[0],second) in true_arcs and have_completed_expected_children(second,true_arcs,built_arcs):
                        built_arcs.append((buffer[0],second))
                        relations_in_order.append(find_corresponding_relation(labeled_arcs, (buffer[0], second)))

                        stack.pop(-2)
                        action_history.append(constants.left_arc_2)
                        continue

                if (second,top) in true_arcs and have_completed_expected_children(top,true_arcs,built_arcs):
                    built_arcs.append((second,top))
                    relations_in_order.append(find_corresponding_relation(labeled_arcs, (second, top)))

                    stack.pop(-1)
                    action_history.append(constants.reduce_r)
                    continue
                if (top, second) in true_arcs and have_completed_expected_children(second, true_arcs, built_arcs):
                    built_arcs.append((top, second))
                    relations_in_order.append(find_corresponding_relation(labeled_arcs, (top, second)))

                    stack.pop(-2)
                    action_history.append(constants.left_arc_prime)
                    continue
                if len(stack)>=3:
                    third = stack[-3]
                    if (third,second) in true_arcs and have_completed_expected_children(second,true_arcs,built_arcs):
                        built_arcs.append((third,second))
                        relations_in_order.append(find_corresponding_relation(labeled_arcs, (third, second)))
                        stack.pop(-2)
                        action_history.append(constants.right_arc_prime)
                        continue

                    if (third,top) in true_arcs and have_completed_expected_children(top,true_arcs,built_arcs):
                        built_arcs.append((third,top))
                        relations_in_order.append(find_corresponding_relation(labeled_arcs, (third, top)))

                        stack.pop(-1)
                        action_history.append(constants.right_arc_2)
                        continue


            if len(buffer)>=1:
                stack.append(buffer.pop(0))
                action_history.append(constants.shift)
                continue
        else:
            stack.append(buffer.pop(0))
            action_history.append(constants.shift)
            continue
    built_arcs.append((0, 0))
    cond1 = set(built_arcs) == set(true_arcs)
    built_arcs.remove((0,0))
    #cond2 = test_oracle_mh4(action_history, sentence.copy(), true_arcs)
    return built_arcs, [], cond1


def build_easy_first_mh4(sentence, word2head,relations,true_arcs):
    #true_arcs = get_arcs(word2head)
    pending = sentence.copy()  # BaseBuffer(sentence)
    built_arcs = {}
    counter = 0

    #true_arcs.remove((0, 0))

    #pending = pending[1:]
    while len(pending) > 0:
        n = len(pending)-1
        if n == 0:
            built_arcs[counter] = (0, pending[0])
            counter += 1
            pending.pop(0)
            continue
        for i in range(n):
            precondition_inext = has_all_children(true_arcs, built_arcs, pending[i + 1])
            precondition_i = has_all_children(true_arcs, built_arcs, pending[i])
            if (pending[i], pending[i + 1]) in true_arcs and precondition_inext:
                built_arcs[counter] = (pending[i], pending[i + 1])
                counter += 1
                pending[i + 1] = -1
            elif (pending[i + 1], pending[i]) in true_arcs and precondition_i:
                built_arcs[counter] = (pending[i + 1], pending[i])
                counter += 1
                pending[i] = -1
        for i in range(n-1):
            if pending[i] != -1 and pending[i+2] != -1:
                precondition_inext = has_all_children(true_arcs, built_arcs,pending[i+2])
                precondition_i = has_all_children(true_arcs, built_arcs, pending[i])
                if (pending[i], pending[i+2]) in true_arcs and precondition_inext:
                    built_arcs[counter] = (pending[i],pending[i+2])
                    counter+=1
                    pending[i + 2] = -1
                elif (pending[i+2],pending[i]) in true_arcs and precondition_i:
                    built_arcs[counter] = (pending[i+2],pending[i])
                    counter+=1
                    pending[i]=-1
        for i in range(n-2):
            if pending[i] != -1 and pending[i+3] != -1:
                precondition_inext = has_all_children(true_arcs, built_arcs,pending[i+3])
                precondition_i = has_all_children(true_arcs, built_arcs, pending[i])
                if (pending[i], pending[i+3]) in true_arcs and precondition_inext:
                    built_arcs[counter] = (pending[i],pending[i+3])
                    counter+=1
                    pending[i + 3] = -1
                elif (pending[i+3],pending[i]) in true_arcs and precondition_i:
                    built_arcs[counter] = (pending[i+3],pending[i])
                    counter+=1
                    pending[i]=-1
        #if n > 2:
        #    for i in range(n-3):
        #        if pending[i] != -1 and pending[i+4] != -1:
        #            precondition_inext = has_all_children(true_arcs, built_arcs,pending[i+4])
        #            precondition_i = has_all_children(true_arcs, built_arcs, pending[i])
        #            if (pending[i], pending[i+4]) in true_arcs and precondition_inext:
        #                built_arcs[counter] = (pending[i],pending[i+4])
        #                counter+=1
        #                pending[i + 4] = -1
        #            elif (pending[i+4],pending[i]) in true_arcs and precondition_i:
        #                built_arcs[counter] = (pending[i+4],pending[i])
        #                counter+=1
        #                pending[i]=-1
        #if n > 3:
        #    for i in range(n-4):
        #        if pending[i] != -1 and pending[i+5] != -1:
        #            precondition_inext = has_all_children(true_arcs, built_arcs,pending[i+5])
        #            precondition_i = has_all_children(true_arcs, built_arcs, pending[i])
        #            if (pending[i], pending[i+5]) in true_arcs and precondition_inext:
        #                built_arcs[counter] = (pending[i],pending[i+5])
        #                counter+=1
        #                pending[i + 5] = -1
        #            elif (pending[i+5],pending[i]) in true_arcs and precondition_i:
        #                built_arcs[counter] = (pending[i+5],pending[i])
        #                counter+=1
        #                pending[i]=-1



        for item in pending:
            if item == -1:
                pending.remove(-1)
    ordered_arcs = list(built_arcs.values())  # []
    cond = set(ordered_arcs) == set(true_arcs)
    ordered_arcs.remove((0,0))
    #ordered_arcs = true_arcs#list(built_arcs.values())  # []
    return ordered_arcs, [],cond#set(ordered_arcs)==set(true_arcs)


def build_eager_easy_first(sentence, true_arcs):
    pending = sentence.copy()  # BaseBuffer(sentence)
    built_arcs = {}
    counter = 0

    true_arcs.remove((0, 0))
    pending = pending[1:]
    while len(pending) > 0:
        n = len(pending) - 1
        if n == 0:
            built_arcs[counter] = (0, pending[0])
            counter += 1
            pending.pop(0)
            continue

        for i in range(n):
            precondition_i = has_all_children(true_arcs, built_arcs, pending[i + 1])
            precondition_inext = has_all_children(true_arcs, built_arcs, pending[i])
            if (pending[i], pending[i + 1]) in true_arcs and precondition_i:
                built_arcs[counter] = (pending[i], pending[i + 1])
                counter += 1
                pending[i + 1] = -1
            elif (pending[i + 1], pending[i]) in true_arcs and precondition_inext:
                built_arcs[counter] = (pending[i + 1], pending[i])
                counter += 1
                pending[i] = -1

        for item in pending:
            if item == -1:
                pending.remove(-1)
    ordered_arcs = built_arcs.values()  # []
    return ordered_arcs


def item_arc_standard_oracle(sentence, word2head):
    true_arcs = get_arcs(word2head)
    n = len(sentence)
    ordered_arcs = build_easy_first(sentence, true_arcs)
    b_hypergraph = build_hypergraph_arc_standard(ordered_arcs, n)
    # print("glaph {}".format(b_hypergraph))
    ordered_arcs = list(ordered_arcs)
    ordered_heads = OrderedDict()
    for (u, v) in ordered_arcs:
        ordered_heads[u] = 1

    agenda = BaseAgenda()
    chart = Chart()
    built_arcs = []
    actions = []
    oracle = []
    for i in range(n):
        agenda[(i, i + 1, i)] = \
            BaseItem(i, i + 1, i, i, i)
    ordered_agenda = BaseAgenda()
    for left, right, derived in b_hypergraph:
        item_l = agenda[left]
        item_r = agenda[right]
        h = derived[2]
        ordered_agenda[(item_l.i, item_l.j, item_l.h)] = item_l
        ordered_agenda[(item_r.i, item_r.j, item_r.h)] = item_r
        if item_l.h == h:
            other = item_r.h
            actions.append("LEFT")
        else:
            other = item_l.h
            actions.append("RIGHT")
        built_arcs.append((h, other))
        new_item = BaseItem(derived[0], derived[1], derived[2], item_l, item_r)
        agenda[(derived[0], derived[1], derived[2])] = new_item
        ordered_agenda[(derived[0], derived[1], derived[2])] = new_item
        oracle.append((
            (item_l.i, item_l.j, item_l.h),
            (item_r.i, item_r.j, item_r.h),
            (new_item.i, new_item.j, new_item.h)
        ))

    return oracle, set(built_arcs) == set(true_arcs)


def item_hybrid_oracle(sentence, word2head):
    true_arcs = get_arcs(word2head)
    n = len(sentence)
    ordered_arcs = build_easy_first(sentence, true_arcs)
    b_hypergraph = build_hypergraph_hybrid(ordered_arcs, n)
    ordered_arcs = list(ordered_arcs)
    ordered_heads = OrderedDict()
    for (u, v) in ordered_arcs:
        ordered_heads[u] = 1

    agenda = BaseAgenda()
    chart = Chart()
    built_arcs = []
    actions = []
    oracle = []
    for i in range(n):
        agenda[(i, i + 1, i)] = \
            BaseItem(i, i + 1, i, i, i)
    ordered_agenda = BaseAgenda()
    for left, right, derived in b_hypergraph:
        item_l = agenda[left]
        item_r = agenda[right]
        h = derived[2]
        ordered_agenda[(item_l.i, item_l.j, item_l.h)] = item_l
        ordered_agenda[(item_r.i, item_r.j, item_r.h)] = item_r
        # other = item_r.i
        if item_r.j == h:
            other = item_r.i
            actions.append("LEFT")
        elif item_l.i == h:
            other = item_l.j
            actions.append("RIGHT")
        built_arcs.append((h, other))
        new_item = BaseItem(derived[0], derived[1], derived[2], item_l, item_r)
        agenda[(derived[0], derived[1], derived[2])] = new_item
        ordered_agenda[(derived[0], derived[1], derived[2])] = new_item
        oracle.append((
            (item_l.i, item_l.j, item_l.h),
            (item_r.i, item_r.j, item_r.h),
            (new_item.i, new_item.j, new_item.h)
        ))

    return oracle, set(built_arcs) == set(true_arcs)


def dfs(visited, children, node):
    if node not in visited:
        visited.append(node)
        for neighbour in children[node]:
            dfs(visited, children, neighbour)
    return visited

def recursive_topological_sort(graph, node):
    result = []
    seen = set()

    def recursive_helper(node):
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                recursive_helper(neighbor)
        result.insert(0, node)

    recursive_helper(node)
    return result


def eager_iter(derived_items, arc,built_arcs,b_hypergraph,is_left=False):
    (u,v) = arc
    for (i1, j1, h1, b1) in derived_items:
        for (i2, j2, h2, b2) in derived_items:
            if i1 == u and j1 == v and (u,v) not in built_arcs:
                item_l = (i1, j1, h1, b1)
                item_r = (i1, j1, h1, b1)
                new_item = (j1, j1 + 1, j1, 1)
                b_hypergraph.append((item_l, item_r, new_item))
                if item_l in derived_items:
                    derived_items.remove(item_l)
                if item_r in derived_items:
                    derived_items.remove(item_r)
                built_arcs.append((i1, j1))
                if new_item not in derived_items:
                    derived_items.append(new_item)
                break
            if j2 == u and j1 == v and i2 == v and b2 == 0 and (u,v) not in built_arcs:
                item_l = (i1, j1, h1, b1)
                item_r = (i2, j2, h2, b2)
                new_item = (i1, j2, i1, b1)
                b_hypergraph.append((item_l, item_r, new_item))
                if item_l in derived_items:
                    derived_items.remove(item_l)
                if item_r in derived_items:
                    derived_items.remove(item_r)
                built_arcs.append((j2, i2))
                if new_item not in derived_items:
                    derived_items.append(new_item)
                break
            if not is_left:
                if b2 == 1 and j1 == i2:
                    item_l = (i1, j1, h1, b1)
                    item_r = (i2, j2, h2, b2)
                    new_item = (i1, j2, i1, b1)
                    b_hypergraph.append((item_l, item_r, new_item))
                    if item_l in derived_items:
                        derived_items.remove(item_l)
                    if item_r in derived_items:
                        derived_items.remove(item_r)
                    if new_item not in derived_items:
                        derived_items.append(new_item)
                    break

    return b_hypergraph, derived_items, built_arcs


def build_hypergraph_eager(ordered_arcs, n):
    derived_items = [(i, i + 1, i, 0) for i in range(n)]
    derived_items = derived_items + [(i, i + 1, i, 1) for i in range(n)]
    derived_items = derived_items + [(i, i + 1, i+1, 1) for i in range(n)]
    derived_items = derived_items + [(i, i + 1, i+1, 0) for i in range(n)]
    b_hypergraph = []
    right_dependents = {i: [] for i in range(n)}
    rest = {i: [] for i in range(n)}
    children = {i: [] for i in range(n)}
    for (u, v) in ordered_arcs:
        children[u].append(v)
        if v > u:
            right_dependents[u].append(v)
        else:
            rest[u].append(v)
    visited = []
    built_arcs = []

    for (u,v) in ordered_arcs:
        b_hypergraph, derived_items, built_arcs = eager_iter(derived_items,(u,v),built_arcs,b_hypergraph)
        if (u,v) in built_arcs:
            ld = sorted(list(children[u]), key=lambda item: u - item)
            for v1 in ld:
                b_hypergraph, derived_items, built_arcs = eager_iter(derived_items, (u, v1), built_arcs, b_hypergraph)

            rd = sorted(list(right_dependents[u]), key=lambda item: u-item)
            for v1 in rd:
                b_hypergraph, derived_items, built_arcs = eager_iter(derived_items, (u, v1), built_arcs, b_hypergraph)
            ld = sorted(list(rest[u]), key=lambda item: u-item)
            for v1 in ld:
                b_hypergraph, derived_items, built_arcs = eager_iter(derived_items, (u, v1), built_arcs, b_hypergraph,True)


    """
    for u in dfs_order:
        rd = sorted(list(children[u]), key=lambda item:-item)
        for v in rd:
            for (i1, j1, h1, b1) in derived_items:
                if i1 == u and j1 == v:
                    item_l = (i1, j1, h1, b1)
                    item_r = (i1, j1, h1, b1)
                    new_item = (j1, j1 + 1, j1, 1)
                    built_arcs.append((i1, j1))
                    if new_item not in derived_items:
                        derived_items.append(new_item)
                    break
                for (i2, j2, h2, b2) in derived_items:
                    if j2 == u and j1 == v and i2 == v and b2 == 0:
                        item_l = (i1, j1, h1, b1)
                        item_r = (i2, j2, h2, b2)
                        new_item = (i1, j2, i1, b1)
                        built_arcs.append((j2,i2))
                        if new_item not in derived_items:
                            derived_items.append(new_item)
                        break
                    if b2 == 1 and j1 == i2:
                        item_l = (i1, j1, h1, b1)
                        item_r = (i2, j2, h2, b2)
                        new_item = (i1, j2, i1, b1)
                        if new_item not in derived_items:
                            derived_items.append(new_item)
                        break
            b_hypergraph.append((item_l, item_r, new_item))
            if item_l in derived_items:
                derived_items.remove(item_l)
            if item_r in derived_items:
                derived_items.remove(item_r)
    ld = sorted(list(right_dependents[u]), key=lambda item: -item)
    for v in ld:
        for (i1, j1, h1, b1) in derived_items:
            if i1 == u and j1 == v:
                item_l = (i1, j1, h1, b1)
                item_r = (i1, j1, h1, b1)
                new_item = (j1, j1 + 1, j1, 1)
                built_arcs.append((i1, j1))
                if new_item not in derived_items:
                    derived_items.append(new_item)
                break
            for (i2, j2, h2, b2) in derived_items:
                if j2 == u and j1 == v and i2 == v and b2 == 0:
                    item_l = (i1, j1, h1, b1)
                    item_r = (i2, j2, h2, b2)
                    new_item = (i1, j2, i1, b1)
                    built_arcs.append((j2,i2))
                    if new_item not in derived_items:
                        derived_items.append(new_item)
                    break
                if b2 == 1 and j1 == i2:
                    item_l = (i1, j1, h1, b1)
                    item_r = (i2, j2, h2, b2)
                    new_item = (i1, j2, i1, b1)
                    if new_item not in derived_items:
                        derived_items.append(new_item)
                    break
        b_hypergraph.append((item_l, item_r, new_item))
        if item_l in derived_items:
            derived_items.remove(item_l)
        if item_r in derived_items:
            derived_items.remove(item_r)
    ld = sorted(list(rest[u]), key=lambda item: u - item)
    for v in ld:
        for (i1, j1, h1, b1) in derived_items:
            if i1 == u and j1 == v:
                item_l = (i1, j1, h1, b1)
                item_r = (i1, j1, h1, b1)
                new_item = (j1, j1 + 1, j1, 1)
                built_arcs.append((i1, j1))
                if new_item not in derived_items:
                    derived_items.append(new_item)
                break
            for (i2, j2, h2, b2) in derived_items:
                if j2 == u and j1 == v and i2 == v and b2 == 0:
                    item_l = (i1, j1, h1, b1)
                    item_r = (i2, j2, h2, b2)
                    new_item = (i1, j2, i1, b1)
                    built_arcs.append((j2, i2))
                    if new_item not in derived_items:
                        derived_items.append(new_item)
                    break
        b_hypergraph.append((item_l, item_r, new_item))
        if item_l in derived_items:
            derived_items.remove(item_l)
        if item_r in derived_items:
            derived_items.remove(item_r)
    """
    return b_hypergraph


def item_eager_oracle(sentence, word2head):
    true_arcs = get_arcs(word2head)
    n = len(sentence)
    ordered_arcs = build_easy_first(sentence, true_arcs)
    b_hypergraph = build_hypergraph_eager(ordered_arcs, n)
    ordered_arcs = true_arcs#list(ordered_arcs)
    if (0,0) in ordered_arcs:
        ordered_arcs.remove((0,0))
    ordered_heads = OrderedDict()
    for (u, v) in ordered_arcs:
        ordered_heads[u] = 1

    agenda = BaseAgenda()
    chart = Chart()
    built_arcs = []
    actions = []
    oracle = []
    for i in range(n):
        agenda[(i, i + 1, i, 0)] = BaseEagerItem(i, i + 1, i, 0, i, i)
        agenda[(i, i + 1, i, 1)] = BaseEagerItem(i, i + 1, i, 1, i, i)

    ordered_agenda = BaseAgendaEager()
    for left, right, derived in b_hypergraph:

        il, jl, hl, bl = left[0], left[1], left[2], left[3]
        ir, jr, hr, br = right[0], right[1], right[2], right[3]

        right_arc = il == ir and jl == jr and hl == hr and bl == br
        if il == derived[0] and bl == derived[3] and jl == ir and jr == derived[1] and br == 0:
            # left arc eager
            actions.append("LEFT")
            h, other = jr, ir
            built_arcs.append((h, other))

        elif il == ir and jl == jr and hl == hr and bl == br and derived[0] + 1 == derived[1] and derived[3] == 1:
            # is right_arc_eager
            h, other = il, jl
            actions.append("RIGHT")
            built_arcs.append((h, other))


        elif il == derived[0] and bl == derived[3] and jl == ir and jr == derived[1] and br == 1:
            # reduce
            actions.append("REDUCE")

        oracle.append((
            (il, jl, hl, bl),
            (ir, jr, hr, br),
            (derived[0], derived[1], derived[2], derived[3])
        ))
    print("UU(O((((((())(()())(())(()")
    print(ordered_arcs)
    print(built_arcs)
    print(true_arcs)
    print("UU(O((((((())(()())(())(()")
    return oracle, set(built_arcs) == set(true_arcs)



