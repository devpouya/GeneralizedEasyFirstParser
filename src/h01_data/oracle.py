from utils import constants
from utils.data_structures import BaseStack, BaseBuffer
import numpy as np


def get_arcs(heads):
    arcs = []
    for i in range(len(heads)):
        arcs.append((heads[i], i))
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
        if a == item:
            needed_arcs.append((a, b))
    for arc in needed_arcs:
        if arc in built_arcs:
            continue
        else:
            return False
    return True

def neighbors(edges, node):
    neighbors = []
    for (u,v) in edges:
        if u == node:
            neighbors.append(u)
        elif v == node:
            neighbors.append(v)
    return neighbors
def from_node(edges, node,visited,rec_stack):
    visited[node] = True
    rec_stack[node] = True
    for neighbor in neighbors(edges, node):
        if not visited[neighbor]:
            if from_node(edges,neighbor,visited,rec_stack):
                return True
            elif rec_stack[neighbor]:
                return True

    rec_stack[node] = False
    return False




def contains_cycles(heads):
    arc_list = get_arcs(heads)
    visited = [False]*len(heads)
    rec_stack = [False]*len(heads)
    for node in range(len(heads)):
        if not visited[node]:
            if from_node(arc_list,node,visited,rec_stack):
                return True
    return False

# adapted from NLTK (copy pasted out of laziness...will fix)
def is_projective(heads):
    arc_list = get_arcs(heads)

    for (parentIdx, childIdx) in arc_list:
        # Ensure that childIdx < parentIdx
        if childIdx > parentIdx:
            temp = childIdx
            childIdx = parentIdx
            parentIdx = temp
        for k in range(childIdx + 1, parentIdx):
            for m in range(len(heads)):
                if (m < childIdx) or (m > parentIdx):
                    if (k, m) in arc_list:
                        return False
                    if (m, k) in arc_list:
                        return False
    return True

def is_good(heads):
    return is_projective(heads) and not contains_cycles(heads)

def arc_standard_oracle(heads):
    # (head,tail)
    # heads[a] == b --> (b,a)
    # sentence = [range(len(heads))]
    # PROJECTIVES 2101
    # SUCCESSES 1547
    # Fails 554
    sentence = list(range(len(heads)))

    stack = BaseStack()
    buffer = BaseBuffer(sentence)

    true_arcs = get_arcs(heads)
    built_arcs = []

    action_history = []

    # initialize stack with root
    #stack.push("ROOT")

    while buffer.get_len() > 0:
        built_arcs = list(set(built_arcs))
        front = buffer.left()
        if stack.get_len() > 0:
            top = stack.top()


            if (front,top) in true_arcs:
                action_history.append(constants.reduce_l)
                built_arcs.append((front,top))
                stack.pop()
                continue
            if (top,front) in true_arcs:
                precondition = have_completed_expected_children(front,true_arcs,built_arcs)
                if precondition:
                    action_history.append(constants.reduce_r)
                    built_arcs.append((top,front))
                    item = stack.pop()
                    buffer.put_left(item)
                    continue
            action_history.append(constants.shift)
            stack.push(buffer.pop_left())

        else:
            action_history.append(constants.shift)
            stack.push(buffer.pop_left())
    #if stack.get_len() > 0:
    #    action_history.append(constants.reduce_l)
    for (a,b) in true_arcs:
        if (b,a) in true_arcs:
            if (a,b) in built_arcs:
                built_arcs.append((b,a))
            elif (b,a) in built_arcs:
                built_arcs.append((a,b))

    while stack.get_len() > 0:
        if set(true_arcs) == set(built_arcs):
            break
        #print(stack.stack)
        #print(true_arcs)
        #print(built_arcs)
        if stack.get_len() >= 2:
            top = stack.top()
            second = stack.second()
            if (top,second) in true_arcs:
                action_history.append(constants.reduce_l)
                built_arcs.append((top,second))
                stack.pop_second()
            elif (second,top) in true_arcs and have_completed_expected_children(top,true_arcs,built_arcs):
                action_history.append(constants.reduce_r)
                built_arcs.append((second,top))
                stack.pop()
            elif (top,top) in true_arcs:
                built_arcs.append((top,top))
                action_history.append(constants.reduce_l)
                stack.pop()
            elif (second,second) in true_arcs:
                built_arcs.append((second,second))
                action_history.append(constants.reduce_r)
                stack.pop_second()
            elif have_completed_expected_children(top,true_arcs,built_arcs):
                stack.pop()
            elif have_completed_expected_children(second,true_arcs,built_arcs):
                stack.pop_second()
            else:

                stack.pop()
        else:
            # stack_len == 1
            top = stack.pop()
            if (top,top) in true_arcs:
                built_arcs.append((top,top))
                action_history.append(constants.reduce_l)
            break
            #else:
            #    left_tings = []
            #    for (a,b) in true_arcs:
            #        if a == top or b == top:
            #            left_tings.append((a,b))


    #print("--------------------")
    #print(true_arcs)
    #print(built_arcs)
    #print("--------------------")

    succ = set(true_arcs) == set(built_arcs)
    return action_history, succ, built_arcs, true_arcs




def arc_eager_oracle(heads):
    pass
