from utils import constants
from utils.data_structures import BaseStack, BaseBuffer
import numpy as np

import itertools


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


# adapted from NLTK (copy pasted out of laziness...will fix)
def is_projective(word2head):
    arc_list = get_arcs(word2head)
    # for i, (u,v) in enumerate(arc_list):
    #    if u == "ROOT":
    #        arc_list[i] = (0,v)
    # print(arc_list)
    for (parentIdx, childIdx) in arc_list:
        # Ensure that childIdx < parentIdx
        if childIdx == parentIdx:
            continue
        if childIdx > parentIdx:
            temp = childIdx
            childIdx = parentIdx
            parentIdx = temp
        for k in range(childIdx + 1, parentIdx):
            for m in range(len(word2head)):
                if (m < childIdx) or (m > parentIdx):
                    if (k, m) in arc_list:
                        return False
                    if (m, k) in arc_list:
                        return False
    return True


def is_good(heads):
    return is_projective(heads) and not contains_cycles(heads)


def test_oracle_arc_standard(action_history, sentence, true_arcs):
    sigma = []
    beta = sentence.copy()
    arcs = []
    for action in action_history:
        if action == constants.shift:
            sigma.append(beta.pop(0))
        elif action == constants.reduce_l:
            arcs.append((sigma[-1], sigma[-2]))
            sigma.pop(-2)
        else:
            arcs.append((sigma[-2], sigma[-1]))
            sigma.pop(-1)
            # beta[0] = item
    arcs.append((0,0))

    return set(arcs) == set(true_arcs)


def find_corresponding_relation(labeled_arcs, arc):
    for (u, v, r) in labeled_arcs:
        if arc == (u, v):
            return r


def arc_standard_oracle(sentence, word2head, relations):
    # (head,tail)
    # heads[a] == b --> (b,a)

    stack = []  # BaseStack()
    buffer = sentence.copy()  # BaseBuffer(sentence)
    # true_arcs_no_label = get_arcs(word2head)
    # true_arcs_labeled = get_labeled_arcs(word2headrels)
    true_arcs = get_arcs(word2head)
    built_arcs = []
    built_labeled_arcs = []
    # labeled_arcs = arcs_with_relations(true_arcs, relations)
    action_history = []

    arcs_sorted = sorted(true_arcs, key=lambda tup: tup[1])[1:]
    labeled_arcs = []
    for i, (u, v) in enumerate(arcs_sorted):
        labeled_arcs.append((u, v, relations[i]))

    relations_in_order = []
    while len(buffer) > 0 or len(stack) > 1:
        # front = buffer[0]
        if len(stack) > 1:
            top = stack[-1]
            second = stack[-2]
            # if (top, top) in true_arcs and have_completed_expected_children(top, true_arcs, built_arcs):
            #    action_history.append(constants.reduce_r)
            #    built_arcs.append((top, top))
            #    relations_in_order.append(find_corresponding_relation(labeled_arcs, (top, top)))
            #    # built_labeled_arcs.append((top, top, relations[0]))
            #    # relations.pop(0)
            #    stack.pop(-1)
            #    continue

            if (top, second) in true_arcs:
                action_history.append(constants.reduce_l)
                built_arcs.append((top, second))
                relations_in_order.append(find_corresponding_relation(labeled_arcs, (top, second)))
                # built_labeled_arcs.append((front, top, relations[0]))
                # relations.pop(0)
                # stack.pop()
                if have_completed_expected_children(second, true_arcs, built_arcs):
                    stack.pop(-2)
                else:
                    action_history.append(constants.shift)
                    stack.append(buffer.pop(0))
                continue
            if (second, top) in true_arcs:
                precondition = have_completed_expected_children(top, true_arcs, built_arcs)
                if precondition:
                    action_history.append(constants.reduce_r)
                    built_arcs.append((second, top))
                    relations_in_order.append(find_corresponding_relation(labeled_arcs, (second, top)))

                    # built_labeled_arcs.append((top, front, relations[0]))
                    item = stack.pop(-1)
                    # buffer[0] = item
                    # relations.pop(0)
                    continue
            action_history.append(constants.shift)
            stack.append(buffer.pop(0))

        else:
            stack.append(buffer.pop(0))
            action_history.append(constants.shift)
            # if len(buffer) == 0:
            #    top = stack.pop(-1)
            #    if (top, top) in true_arcs:
            #        built_arcs.append((top, top))
            #        # built_labeled_arcs.append((top, top, relations[0]))
            #        # relations.pop(0)
            #        action_history.append(None)
            #        # relations_in_order.append(find_corresponding_relation(labeled_arcs, (top, top)))

    built_arcs.append((0, 0))

    cond = test_oracle_arc_standard(action_history, sentence.copy(), true_arcs)
    return action_history, relations_in_order, set(built_arcs) == set(true_arcs) and cond


def arc_standard_oracle2(sentence, word2head, relations):
    # (head,tail)
    # heads[a] == b --> (b,a)

    stack = []  # BaseStack()
    buffer = sentence.copy()  # BaseBuffer(sentence)
    # true_arcs_no_label = get_arcs(word2head)
    # true_arcs_labeled = get_labeled_arcs(word2headrels)
    true_arcs = get_arcs(word2head)
    built_arcs = []
    built_labeled_arcs = []
    # labeled_arcs = arcs_with_relations(true_arcs, relations)
    action_history = []

    arcs_sorted = sorted(true_arcs, key=lambda tup: tup[1])[1:]
    labeled_arcs = []
    for i, (u, v) in enumerate(arcs_sorted):
        labeled_arcs.append((u, v, relations[i]))

    relations_in_order = []
    while len(buffer) > 0:
        front = buffer[0]
        if len(stack) > 0:
            top = stack[-1]
            if (top, top) in true_arcs and have_completed_expected_children(top, true_arcs, built_arcs):
                action_history.append(constants.reduce_r)
                built_arcs.append((top, top))
                relations_in_order.append(find_corresponding_relation(labeled_arcs, (top, top)))

                # built_labeled_arcs.append((top, top, relations[0]))
                # relations.pop(0)
                stack.pop(-1)
                continue

            if (front, top) in true_arcs:
                action_history.append(constants.reduce_l)
                built_arcs.append((front, top))
                relations_in_order.append(find_corresponding_relation(labeled_arcs, (front, top)))
                # built_labeled_arcs.append((front, top, relations[0]))
                # relations.pop(0)
                # stack.pop()
                if have_completed_expected_children(top, true_arcs, built_arcs):
                    stack.pop(-1)
                else:
                    action_history.append(constants.shift)
                    stack.append(buffer.pop(0))
                continue
            if (top, front) in true_arcs:
                precondition = have_completed_expected_children(front, true_arcs, built_arcs)
                if precondition:
                    action_history.append(constants.reduce_r)
                    built_arcs.append((top, front))
                    relations_in_order.append(find_corresponding_relation(labeled_arcs, (top, front)))

                    # built_labeled_arcs.append((top, front, relations[0]))
                    item = stack.pop(-1)
                    buffer[0] = item
                    # relations.pop(0)
                    continue
            action_history.append(constants.shift)
            stack.append(buffer.pop(0))

        else:
            stack.append(buffer.pop(0))
            action_history.append(constants.shift)
            if len(buffer) == 0:
                top = stack.pop(-1)
                if (top, top) in true_arcs:
                    built_arcs.append((top, top))
                    # built_labeled_arcs.append((top, top, relations[0]))
                    # relations.pop(0)
                    action_history.append(None)
                    # relations_in_order.append(find_corresponding_relation(labeled_arcs, (top, top)))

    return action_history, relations_in_order


def test_oracle_arc_eager(action_history, sentence, true_arcs):
    sigma = []
    beta = sentence.copy()
    arcs = []

    for action in action_history:
        if action == constants.shift:
            sigma.append(beta.pop(0))
        elif action == constants.left_arc_eager:
            arcs.append((beta[0], sigma[-1]))
            sigma.pop(-1)
        elif action == constants.right_arc_eager:
            arcs.append((sigma[-1], beta[0]))
            sigma.append(beta.pop(0))
        elif action == constants.reduce:
            sigma.pop(-1)

    # arcs.append((0, 0))
    return set(arcs) == set(true_arcs)


def arc_eager_oracle(sentence, word2head, relations):
    stack = []  # BaseStack()
    buffer = sentence.copy()  # BaseBuffer(sentence)
    # true_arcs_no_label = get_arcs(word2head)
    # true_arcs_labeled = get_labeled_arcs(word2headrels)
    true_arcs = get_arcs(word2head)
    built_arcs = []
    built_labeled_arcs = []
    # labeled_arcs = arcs_with_relations(true_arcs, relations)
    action_history = []

    arcs_sorted = sorted(true_arcs, key=lambda tup: tup[1])[1:]
    labeled_arcs = []
    for i, (u, v) in enumerate(arcs_sorted):
        labeled_arcs.append((u, v, relations[i]))

    relations_in_order = []
    while len(buffer) > 0:
        front = buffer[0]
        if len(stack) > 0:
            top = stack[-1]
            if (top, front) in true_arcs:
                built_arcs.append((top, front))
                relations_in_order.append(find_corresponding_relation(labeled_arcs, (top, front)))
                action_history.append(constants.right_arc_eager)
                buffer.pop(0)
                stack.append(front)
                continue
            if (front, top) in true_arcs:
                built_arcs.append((front, top))
                relations_in_order.append(find_corresponding_relation(labeled_arcs, (front, top)))
                action_history.append(constants.left_arc_eager)
                stack.pop(-1)
                continue

            if has_head(top, built_arcs) and have_completed_expected_children(top, true_arcs, built_arcs) and top != 0:
                action_history.append(constants.reduce)
                stack.pop(-1)
                continue
            stack.append(buffer.pop(0))
            action_history.append(constants.shift)

        else:
            stack.append(buffer.pop(0))
            action_history.append(constants.shift)

    built_arcs.append((0, 0))

    cond1 = set(built_arcs) == set(true_arcs)
    cond2 = test_oracle_arc_eager(action_history, sentence.copy(), true_arcs)
    # print("cond1 {} cond2 {}".format(cond1,cond2))
    return action_history, relations_in_order, cond1 and cond2


def test_oracle_hybrid(action_history, sentence, true_arcs):
    sigma = []
    beta = sentence.copy()
    arcs = []

    for action in action_history:
        if action == constants.shift:
            sigma.append(beta.pop(0))
        elif action == constants.left_arc_eager:
            arcs.append((beta[0], sigma[-1]))
            sigma.pop(-1)
        elif action == constants.reduce_r:
            arcs.append((sigma[-2], sigma[-1]))
            sigma.pop(-1)
    # arcs.append((0,0))
    # print(arcs)
    return set(arcs) == set(true_arcs)


def hybrid_oracle(sentence, word2head, relations):
    stack = []  # BaseStack()
    buffer = sentence.copy()  # BaseBuffer(sentence)
    # true_arcs_no_label = get_arcs(word2head)
    # true_arcs_labeled = get_labeled_arcs(word2headrels)
    true_arcs = get_arcs(word2head)
    built_arcs = []
    built_labeled_arcs = []
    # labeled_arcs = arcs_with_relations(true_arcs, relations)
    action_history = []
    arcs_sorted = sorted(true_arcs, key=lambda tup: tup[1])[1:]
    labeled_arcs = []
    for i, (u, v) in enumerate(arcs_sorted):
        labeled_arcs.append((u, v, relations[i]))
    relations_in_order = []
    while len(buffer) > 0 or len(stack) > 1:
        if len(stack) > 0:
            top = stack[-1]
            if len(stack) == 1 and len(buffer) > 0:
                front = buffer[0]
                if (front, top) in true_arcs and have_completed_expected_children(top, true_arcs, built_arcs):
                    built_arcs.append((front, top))
                    relations_in_order.append(find_corresponding_relation(labeled_arcs, (front, top)))
                    action_history.append(constants.left_arc_eager)
                    stack.pop(-1)
                    continue
                stack.append(buffer.pop(0))
                action_history.append(constants.shift)

            else:
                second = stack[-2]
                if (second, top) in true_arcs and have_completed_expected_children(top, true_arcs, built_arcs):
                    built_arcs.append((second, top))
                    relations_in_order.append(find_corresponding_relation(labeled_arcs, (second, top)))
                    action_history.append(constants.reduce_r)
                    stack.pop(-1)
                    continue
                if len(buffer) > 0:
                    front = buffer[0]
                    if (front, top) in true_arcs and have_completed_expected_children(top, true_arcs, built_arcs):
                        built_arcs.append((front, top))
                        relations_in_order.append(find_corresponding_relation(labeled_arcs, (front, top)))
                        action_history.append(constants.left_arc_eager)
                        stack.pop(-1)
                        continue

                stack.append(buffer.pop(0))
                action_history.append(constants.shift)
        else:
            stack.append(buffer.pop(0))
            action_history.append(constants.shift)

    built_arcs.append((0, 0))
    # action_history.append(None)

    cond1 = set(built_arcs) == set(true_arcs)
    cond2 = test_oracle_hybrid(action_history, sentence.copy(), true_arcs)
    return action_history, relations_in_order, cond1 and cond2

def test_oracle_mh4(action_history, sentence, true_arcs):
    sigma = []
    beta = sentence.copy()
    arcs = []

    for action in action_history:
        if action == constants.shift:
            sigma.append(beta.pop(0))
        elif action == constants.left_arc_eager:
            arcs.append((beta[0], sigma[-1]))
            sigma.pop(-1)
        elif action == constants.reduce_r:
            arcs.append((sigma[-2], sigma[-1]))
            sigma.pop(-1)
        elif action == constants.left_arc_prime:
            arcs.append((sigma[-1],sigma[-2]))
            sigma.pop(-2)
        elif action == constants.right_arc_prime:
            arcs.append((sigma[-3],sigma[-2]))
            sigma.pop(-2)
        elif action == constants.left_arc_2:
            arcs.append((beta[0],sigma[-2]))
            sigma.pop(-2)
        elif action == constants.right_arc_2:
            arcs.append((sigma[-3],sigma[-1]))
            sigma.pop(-1)
    arcs.append((0,0))
    # print(arcs)
    return set(arcs) == set(true_arcs)

def mh4_oracle(sentence, word2head, relations):
    stack = []
    buffer = sentence.copy()

    true_arcs = get_arcs(word2head)
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
    cond2 = test_oracle_mh4(action_history, sentence.copy(), true_arcs)
    return action_history, relations_in_order, cond1 and cond2



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


def test_oracle_easy_first(action_history, sentence, true_arcs):
    pending = sentence.copy()
    arcs = []
    for action in action_history:
        (i,j) = action
        arcs.append((pending[i],pending[j]))
        pending.pop(j)


    return set(arcs) == set(true_arcs)

def easy_first_pending(sentence, word2head, relations):
    stack = []  # BaseStack()
    pending = sentence.copy()  # BaseBuffer(sentence)
    buffer = sentence.copy()  # BaseBuffer(sentence)
    # true_arcs_no_label = get_arcs(word2head)
    # true_arcs_labeled = get_labeled_arcs(word2headrels)
    true_arcs = get_arcs(word2head)
    built_arcs = {}
    counter = 0
    built_labeled_arcs = []
    # labeled_arcs = arcs_with_relations(true_arcs, relations)
    action_history = []
    arcs_sorted = sorted(true_arcs, key=lambda tup: tup[1])[1:]
    labeled_arcs = []
    for i, (u, v) in enumerate(arcs_sorted):
        labeled_arcs.append((u, v, relations[i]))
    relations_in_order = []
    easy_first_actions = []
    true_arcs.remove((0, 0))
    #pending = pending[1:]
    while len(pending) > 1:
        n = len(pending)
        if n==2:
            if (pending[0], pending[1]) in true_arcs:
                built_arcs[counter] = (pending[0], pending[1])
                easy_first_actions.append((0, 1))
                relations_in_order.append(find_corresponding_relation(labeled_arcs, (pending[0], pending[1])))

                counter += 1
                # built_arcs.append((0,pending[0]))
                pending.pop(1)
                continue
            elif (pending[1], pending[0]) in true_arcs:
                built_arcs[counter] = (pending[1], pending[0])
                easy_first_actions.append((1, 0))
                relations_in_order.append(find_corresponding_relation(labeled_arcs, (pending[1], pending[0])))

                counter += 1
                # built_arcs.append((0,pending[0]))
                pending.pop(0)
                continue
        #print(n)
        for i in range(1,n,1):
            this = i
            nxt = min(i+1,n-1)
            precondition_i = has_all_children(true_arcs, built_arcs, pending[nxt])
            precondition_inext = has_all_children(true_arcs, built_arcs, pending[i])
            if (pending[i], pending[nxt]) in true_arcs and precondition_i:
                built_arcs[counter] = (pending[i], pending[nxt])
                relations_in_order.append(find_corresponding_relation(labeled_arcs, (pending[i], pending[i + 1])))
                counter += 1
                # built_arcs.append((pending[i],pending[i+1]))
                #pending[i + 1] = -1
                #easy_first_actions.append((pending[i], pending[i + 1]))
                easy_first_actions.append((i, nxt))
                pending.pop(i+1)

                break
            elif (pending[nxt], pending[i]) in true_arcs and precondition_inext:
                built_arcs[counter] = (pending[nxt], pending[i])
                relations_in_order.append(find_corresponding_relation(labeled_arcs, (pending[nxt], pending[i])))
                counter += 1
                # built_arcs.append((pending[i+1],pending[i]))
                #pending[i] = -1
                #easy_first_actions.append((pending[i + 1], pending[i]))
                easy_first_actions.append((nxt, i))
                pending.pop(i)

                break


    ordered_arcs = built_arcs.values()  # []
    actions = []
    for (i,j) in easy_first_actions:
        if i < j:
            actions.append((constants.left_attach,(i,j)))
        else:
            actions.append((constants.right_attach,(i,j)))
    cond1 = set(ordered_arcs) == set(true_arcs)
    cond2 = test_oracle_easy_first(easy_first_actions,sentence,true_arcs)
    return actions,relations_in_order, cond1 and cond2

def easy_first_prune(sentence, word2head, relations):
    stack = []  # BaseStack()
    pending = sentence.copy()  # BaseBuffer(sentence)
    buffer = sentence.copy()  # BaseBuffer(sentence)
    # true_arcs_no_label = get_arcs(word2head)
    # true_arcs_labeled = get_labeled_arcs(word2headrels)
    true_arcs = get_arcs(word2head)
    built_arcs = {}
    counter = 0
    built_labeled_arcs = []
    # labeled_arcs = arcs_with_relations(true_arcs, relations)
    action_history = []
    arcs_sorted = sorted(true_arcs, key=lambda tup: tup[1])[1:]
    labeled_arcs = []
    for i, (u, v) in enumerate(arcs_sorted):
        labeled_arcs.append((u, v, relations[i]))
    relations_in_order = []
    easy_first_actions = []
    true_arcs.remove((0, 0))
    pending = pending[1:]
    while len(pending) > 0:
        n = len(pending) - 1
        if n == 0:
            built_arcs[counter] = (0, pending[0])
            counter += 1
            # built_arcs.append((0,pending[0]))
            #pending.pop(0)
            continue

        for i in range(n):
            precondition_i = has_all_children(true_arcs, built_arcs, pending[i + 1])
            precondition_inext = has_all_children(true_arcs, built_arcs, pending[i])
            if (pending[i], pending[i + 1]) in true_arcs and precondition_i:
                built_arcs[counter] = (pending[i], pending[i + 1])
                relations_in_order.append(find_corresponding_relation(labeled_arcs, (pending[i], pending[i + 1])))
                counter += 1
                # built_arcs.append((pending[i],pending[i+1]))
                #pending[i + 1] = -1
                pending.pop(i+1)
                easy_first_actions.append(("left-attach", (i, i + 1)))
            elif (pending[i + 1], pending[i]) in true_arcs and precondition_inext:
                built_arcs[counter] = (pending[i + 1], pending[i])
                relations_in_order.append(find_corresponding_relation(labeled_arcs, (pending[i + 1], pending[i])))

                counter += 1
                # built_arcs.append((pending[i+1],pending[i]))
                #pending[i] = -1
                pending.pop(i)
                easy_first_actions.append(("right-attach", (i + 1, i)))

        #for item in pending:
        #    if item == -1:
        #        pending.remove(-1)

    ordered_arcs = built_arcs.values()  # []
    have_inserted = [False] * len(built_arcs)
    # reordering, will try this approach too
    # for key1,(u1,v1) in built_arcs.items():
    #    if not have_inserted[key1]:
    #        ordered_arcs.append((u1,v1))
    #        have_inserted[key1] = True
    #    for key2,(u2,v2) in built_arcs.items():
    #        if u2 == u1 or v2 == u1:
    #            if abs(u2 - u1) <= 1 and abs(v2-v1) <= 1 and u2 != 0:
    #                if not have_inserted[key2]:
    #                    ordered_arcs.append((u2,v2))
    #                    have_inserted[key2] = True

    have_shifted = [False] * len(sentence)

    for (u, v) in ordered_arcs:

        if have_shifted[u] and have_shifted[v]:
            arc_on_top = (stack[-1] == u and stack[-2] == v) or (stack[-1] == v and stack[-2] == u)
            while not arc_on_top and len(stack) >= 2:
                buffer = [stack[-1]] + buffer
                have_shifted[stack[-1]] = False
                stack.pop(-1)
                action_history.append(constants.prune)
                arc_on_top = (stack[-1] == u and stack[-2] == v) or (stack[-1] == v and stack[-2] == u)
        if not have_shifted[u]:
            shifts = 0
            while not have_shifted[u]:
                shifts += 1
                item = buffer.pop(0)
                have_shifted[item] = True
                stack.append(item)
            shifted = [constants.shift] * shifts
            action_history.extend(shifted)
        if not have_shifted[v]:
            shifts = 0
            while not have_shifted[v]:
                shifts += 1
                item = buffer.pop(0)
                have_shifted[item] = True
                stack.append(item)
            shifted = [constants.shift] * shifts
            action_history.extend(shifted)
        if v < u:
            action_history.append(constants.reduce_l)
            stack.pop(-2)
        else:
            action_history.append(constants.reduce_r)
            stack.pop(-1)

    cond1 = test_oracle_easy_first(action_history, sentence.copy(), true_arcs)

    cond2 = set(ordered_arcs) == set(true_arcs)
    return action_history, relations_in_order, cond2 and cond1


def sort_hypergraph(b_hypergraph, axioms):
    sorted_b_hypergraph = []
    stack = []

    have_visited = {item: False for item in list(itertools.chain(*b_hypergraph))}
    have_placed = {item: False for item in b_hypergraph}
    for item in axioms:
        have_visited[item] = True

    for (a1, b1, c1) in b_hypergraph:
        if not have_placed[(a1, b1, c1)]:
            sorted_b_hypergraph.append((a1, b1, c1))
            have_placed[(a1, b1, c1)] = True
        stack.append(c1)
        have_visited[a1] = True
        if b1 != (-1, -1, -1):
            have_visited[b1] = True
        # have_visited[c1] = True
        while len(stack) > 0:
            item = stack.pop(-1)
            if not have_visited[item]:
                have_visited[item] = True
            for (a2, b2, c2) in b_hypergraph:
                if b2 != (-1, -1, -1):
                    if (a2 == item and have_visited[b2]) or (b2 == item and have_visited[a2]):
                        have_visited[a2] = True
                        have_visited[b2] = True
                        stack.append(c2)
                        if not have_placed[(a2, b2, c2)]:
                            sorted_b_hypergraph.append((a2, b2, c2))
                            have_placed[(a2, b2, c2)] = True
                else:
                    if a2 == item:
                        have_visited[a2] = True
                        stack.append(c2)
                        if not have_placed[(a2, b2, c2)]:
                            sorted_b_hypergraph.append((a2, b2, c2))
                            have_placed[(a2, b2, c2)] = True

    return sorted_b_hypergraph


def build_easy_first(sentence, true_arcs):
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


def build_hypergraph(ordered_arcs, n):
    derived_items = [(i, i, i) for i in range(n)]
    b_hypergraph = []
    print(ordered_arcs)
    jhjh
    for (u, v) in ordered_arcs:
        derived_items = sorted(derived_items, key=lambda x: x[2])
        item_l, item_r, new_item = None, None, None
        for (i1, j1, h1) in derived_items:
            for (i2, j2, h2) in derived_items:
                if h1 == u and h2 == v:
                    bigger = max(u, v)
                    if bigger == v:
                        if j1 + 1 == i2:
                            new_item = (i1, j2, u)
                            derived_items.append(new_item)
                            item_l = (i1, j1, h1)
                            item_r = (i2, j2, h2)
                            break
                    else:
                        if j2 + 1 == i1:
                            new_item = (i2, j1, u)
                            derived_items.append(new_item)
                            item_l = (i2, j2, h2)
                            item_r = (i1, j1, h1)

        b_hypergraph.append((item_l, item_r, new_item))
        derived_items.remove(item_l)
        derived_items.remove(item_r)
    return b_hypergraph


def build_hypergraph_hybrid(ordered_arcs, n):
    derived_items = [(i, i + 1, i + 1) for i in range(n)]
    # derived_items.remove((n-1,n,n))
    b_hypergraph = []
    for (u, v) in ordered_arcs:
        derived_items = sorted(derived_items, key=lambda x: x[1])
        item_l, item_r, new_item = None, None, None
        # [_,v] [v,u] -- > [u,v] left-eager
        # [u,v] [v,_] -- > [v,u] right_reduce
        for (i1, j1, h1) in derived_items:
            for (i2, j2, h2) in derived_items:
                if j1 == v and i2 == v and j2 == u and j2 < n:
                    # left-arc-eager
                    new_item = (i1, j2, u)
                    item_l = (i1, j1, h1)
                    item_r = (i2, j2, h2)
                    break
                elif i1 == u and j1 == v and i2 == v:
                    # right-reduce
                    new_item = (i1, j2, u)
                    item_l = (i1, j1, h1)
                    item_r = (i2, j2, h2)
                    break
        b_hypergraph.append((item_l, item_r, new_item))

        derived_items.append(new_item)
        derived_items.remove(item_l)
        derived_items.remove(item_r)
    return b_hypergraph


def build_hypergraph_mh4(ordered_arcs, n):
    derived_items = [{i, i + 1} for i in range(n)]

    # derived_items.remove((n-1,n,n))
    b_hypergraph = []
    heads = {u for (u,v) in ordered_arcs}

    while len(derived_items) > 0:
        item = derived_items.pop(0)




    return b_hypergraph


def build_hypergraph_eager(ordered_arcs, n):
    # the item is [i,j,h,b] === [i,j,h^b]
    derived_items = [(i, i + 1, 0) for i in range(n)]
    derived_items = {(i, i + 1,0):False for i in range(n)}
    #for i in range(n):
    #    derived_items[(i,i+1,1)] = False
    print(ordered_arcs)
    # derived_items.remove((n-1,n,n))
    b_hypergraph = []
    for (u, v) in ordered_arcs:
        for (i1, j1,b1) in list(derived_items.keys()):
            #if derived_items[(i1, j1,b1)]:
            #    continue
            for (i2, j2,b2) in list(derived_items.keys()):
                # if derived_items[(i2, j2,b2)]:
                #     continue
                if u == j2 and v == j1 and v == i2 and b2 == 0 and j2 < n:
                    derived_items[(i1, j2,b1)] = False
                    derived_items[(i1, j1,b1)] = True
                    derived_items[(i2, j2,b2)] = True
                    #break
                elif i1 == u and j1 == v:
                    derived_items[(j1, j1 + 1,1)] = False
                    derived_items[(i1, j1,b1)] = True
                    #break
                elif i2 == u and j2 == v:
                    derived_items[(j2, j2 + 1,1)] = False
                    derived_items[(i2, j2,b2)] = True
                    #break
                elif j1 == i2 and b2 == 1 and u != i1 and u != j1 and v != j1 and v != j2 and u != j2:
                    derived_items[(i1, j2,b1)] =  False
                    derived_items[(i1, j1,b1)] =  True
                    derived_items[(i2, j2,b2)] =  True
                    #break


    return b_hypergraph


def prepare_easy_first(sentence, word2head, relations, mode):
    labeled_arcs = []
    n = len(sentence)
    true_arcs = get_arcs(word2head)
    arcs_sorted = sorted(true_arcs, key=lambda tup: tup[1])[1:]
    for i, (u, v) in enumerate(arcs_sorted):
        labeled_arcs.append((u, v, relations[i]))
    ordered_arcs = build_easy_first(sentence, true_arcs)
    if mode == 'arc-standard':
        b_hypergraph = build_hypergraph(ordered_arcs, n)
        axioms = [(i, i, i) for i in range(n)]
    elif mode == 'hybrid':
        axioms = [(i, i + 1, i + 1) for i in range(n)]
        b_hypergraph = build_hypergraph_hybrid(ordered_arcs, n)
    elif mode == 'eager':
        # axioms = [(i,i,i) for i in range(n)]
        axioms = [(i, i + 1, 0) for i in range(n)]
        # for i in range(n):
        #    axioms.append((i, i + 1, 1))
        b_hypergraph = build_hypergraph_eager(ordered_arcs, n)
    elif mode == 'mh4':
        axioms = [(i,i+1,i+1) for i in range(n)]
        b_hypergraph = build_hypergraph_mh4(ordered_arcs,n)

    sorted_b_hypergraph = sort_hypergraph(b_hypergraph, axioms)
    return sorted_b_hypergraph, labeled_arcs, ordered_arcs, true_arcs


def easy_first_arc_standard(sentence, word2head, relations):
    stack = []  # BaseStack()
    buffer = sentence.copy()  # BaseBuffer(sentence)
    action_history = []
    relations_in_order = []
    sorted_b_hypergraph, labeled_arcs, ordered_arcs, true_arcs = prepare_easy_first(sentence, word2head, relations,
                                                                                    mode='arc-standard')
    have_shifted = [False] * len(sentence)
    for (a, b, c) in sorted_b_hypergraph:
        if not have_shifted[a[2]]:
            while not have_shifted[a[2]]:
                item = buffer.pop(0)
                have_shifted[item] = True
                stack.append(item)
                action_history.append(constants.shift)
        if not have_shifted[b[2]]:
            while not have_shifted[b[2]]:
                item = buffer.pop(0)
                have_shifted[item] = True
                stack.append(item)
                action_history.append(constants.shift)
        (lower, upper, head) = c

        if head == a[2]:
            relations_in_order.append(find_corresponding_relation(labeled_arcs, (stack[-2], stack[-1])))
            stack.pop(-1)
            action_history.append(constants.reduce_r)
        elif head == b[2]:
            relations_in_order.append(find_corresponding_relation(labeled_arcs, (stack[-1], stack[-2])))
            stack.pop(-2)
            action_history.append(constants.reduce_l)

    # cond1 = test_oracle_easy_first(action_history,sentence.copy(),true_arcs)
    # cond2 = set(ordered_arcs) == set(true_arcs)
    return action_history, relations_in_order, True  # , cond1 and cond2


def easy_first_hybrid(sentence, word2head, relations):
    stack = []  # BaseStack()
    buffer = sentence.copy()  # BaseBuffer(sentence)
    action_history = []
    relations_in_order = []
    sorted_b_hypergraph, labeled_arcs, ordered_arcs, true_arcs = prepare_easy_first(sentence, word2head, relations,
                                                                                    mode='hybrid')
    have_shifted = [False] * (len(sentence))
    n = len(sentence)
    hypergraph = []
    for (a, b, c) in sorted_b_hypergraph:
        i1, j1, h1 = a
        i2, j2, h2 = b
        i3, j3, h3 = c
        hypergraph.append(((min(i1, n - 1), min(j1, n - 1), min(h1, n - 1)),
                           (min(i2, n - 1), min(j2, n - 1), min(h2, n - 1)),
                           (min(i3, n - 1), min(j3, n - 1), min(h3, n - 1))))
    for (a, b, c) in hypergraph:

        (lower, upper, head) = c
        if not have_shifted[a[1]]:
            while not have_shifted[a[1]]:
                item = buffer.pop(0)
                have_shifted[item] = True
                stack.append(item)
                action_history.append(constants.shift)

        if head == b[1]:

            relations_in_order.append(find_corresponding_relation(labeled_arcs, (buffer[0], stack[-1])))
            stack.pop(-1)
            action_history.append(constants.left_arc_eager)
        elif head == a[0]:

            relations_in_order.append(find_corresponding_relation(labeled_arcs, (stack[-2], stack[-1])))
            stack.pop(-1)
            action_history.append(constants.reduce_r)
        elif not have_shifted[b[1]]:
            while not have_shifted[b[1]]:
                item = buffer.pop(0)
                have_shifted[item] = True
                stack.append(item)
                action_history.append(constants.shift)

    cond1 = test_oracle_hybrid(action_history, sentence.copy(), true_arcs)
    cond2 = set(ordered_arcs) == set(true_arcs)

    return action_history, relations_in_order, cond1 and cond2


def easy_first_mh4(sentence, word2head, relations):
    stack = []  # BaseStack()
    buffer = sentence.copy()  # BaseBuffer(sentence)
    action_history = []
    relations_in_order = []
    sorted_b_hypergraph, labeled_arcs, ordered_arcs, true_arcs = prepare_easy_first(sentence, word2head, relations,
                                                                                    mode='mh4')
    have_shifted = [False] * (len(sentence))
    n = len(sentence)

    hypergraph = []
    pass




def easy_first_arc_eager(sentence, word2head, relations):
    stack = []  # BaseStack()
    buffer = sentence.copy()  # BaseBuffer(sentence)
    action_history = []
    relations_in_order = []
    sorted_b_hypergraph, labeled_arcs, ordered_arcs, true_arcs = prepare_easy_first(sentence, word2head, relations,
                                                                                    mode='eager')
    have_shifted = [False] * (len(sentence))
    n = len(sentence)
    hypergraph = []
    print(sorted_b_hypergraph)
    for (a, b, c) in sorted_b_hypergraph:
        i1, j1, h1 = a
        try:
            i2, j2, h2 = b
        except:
            i2, j2, h2 = -1, -1, -1

        i3, j3, h3 = c
        hypergraph.append(((min(i1, n - 1), min(j1, n - 1), min(h1, n - 1)),
                           (min(i2, n - 1), min(j2, n - 1), min(h2, n - 1)),
                           (min(i3, n - 1), min(j3, n - 1), min(h3, n - 1))))
    # print()
    print(n)
    print(hypergraph)
    print(true_arcs)
    for (a, b, c) in hypergraph:

        (lower, upper, has_head) = c
        head = lower
        if not have_shifted[a[0]]:
            while not have_shifted[a[0]]:
                item = buffer.pop(0)
                have_shifted[item] = True
                stack.append(item)
                action_history.append(constants.shift)

        if head == b[0] and has_head == 0:
            print(stack)
            print(buffer)

            relations_in_order.append(find_corresponding_relation(labeled_arcs, (buffer[0], stack[-1])))
            stack.pop(-1)
            action_history.append(constants.left_arc_eager)
        elif head == a[0]:
            print(stack)
            print(buffer)

            stack.append(buffer.pop(0))
            relations_in_order.append(find_corresponding_relation(labeled_arcs, (stack[-2], stack[-1])))
            action_history.append(constants.right_arc_eager)
        elif b[2] == 1 and c == (a[0], b[1], a[2]):
            print(stack)
            print(buffer)

            action_history.append(constants.reduce)
            stack.pop(-1)

        elif not have_shifted[b[0]]:
            while not have_shifted[b[0]]:
                item = buffer.pop(0)
                have_shifted[item] = True
                stack.append(item)
                action_history.append(constants.shift)
    # print(action_history)
    # print(true_arcs)

    cond1 = test_oracle_arc_eager(action_history, sentence.copy(), true_arcs)
    cond2 = set(ordered_arcs) == set(true_arcs)
    print(cond1)
    return action_history, relations_in_order, cond1 and cond2

