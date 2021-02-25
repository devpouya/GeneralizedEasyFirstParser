from utils import constants
from utils.data_structures import BaseStack, BaseBuffer
import numpy as np


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
            arcs.append((beta[0], sigma[-1]))
            sigma.pop(-1)
        elif action is None:
            arcs.append((sigma[-1], sigma[-1]))
        else:
            arcs.append((sigma[-1], beta[0]))
            item = sigma.pop()
            beta[0] = item

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
        else:
            item = sigma.pop(-1)
            arcs.append((item,item))
    arcs.append((0,0))
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
            if len(buffer) == 0:
                top = stack.pop(-1)
                if (top, top) in true_arcs:
                    built_arcs.append((top, top))
                    # built_labeled_arcs.append((top, top, relations[0]))
                    # relations.pop(0)
                    action_history.append(None)
    built_arcs.append((0,0))
    action_history.append(None)

    #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    #print(built_arcs)
    #print(true_arcs)
    #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

    #cond1 = set(built_arcs) == set(true_arcs)
    #cond2 = test_oracle_arc_eager(action_history,sentence.copy(),true_arcs)
    return action_history, relations_in_order#, cond1 and cond2
