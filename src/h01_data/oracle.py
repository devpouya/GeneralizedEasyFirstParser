from utils import constants
from utils.data_structures import BaseStack, BaseBuffer


def is_head(a, b, heads):

    return heads[b] == a


def children(a, heads):
    children = []
    for item in heads:

        if is_head(a, item, heads):
            children.append(item)

    return children


def has_as_head(a, heads):
    for item in heads:
        if item != a:
            if is_head(a, item, heads):
                return True
    return False


def ready_to_right(a, heads, arcs):
    # c = children(a,heads)
    for item in children(a, heads):
        if (a, item) in arcs:
            continue
        else:
            return False
    return True


def arc_standard_oracle(sentence, heads):
    stack = BaseStack()  # []
    buffer = BaseBuffer(sentence)
    # print(heads)
    # print(sentence)

    action_history = []
    # stack.push("SIGMA_0")
    stack.push(buffer.pop_left())
    #stack.push("ROOT")
    stack.push(buffer.pop_left())
    # print(constants.shift)
    # print(constants.shift)
    # print(constants.shift)
    # print(stack.stack)
    # print(buffer.buffer)
    relations = []
    while buffer.get_len() > 0 and stack.get_len() != 1:

        if len(stack.stack) >= 2:
            if is_head(stack.top(), stack.second(), heads):
                #print("A")
                action_history.append(constants.reduce_l)
                relations.append((stack.top(), stack.second()))
                stack.pop_second()
            elif is_head(stack.second(), stack.top(), heads) and ready_to_right(stack.top(), heads, relations):
                #print("B")
                action_history.append(constants.reduce_r)
                relations.append((stack.second(), stack.top()))
                stack.pop()
            elif buffer.get_len() > 0:
                #print("C")
                action_history.append(constants.shift)
                buffer.pop_left()
        elif buffer.get_len() > 0:
            #print("D")
            action_history.append(constants.shift)
            buffer.pop_left()

    return action_history


def arc_eager_oracle(sentence, heads):
    pass


def hybrid_oracle(sentence, heads):
    pass


def non_projective_oracle(sentence, heads):
    pass


def easy_first_oracle(sentence, heads):
    pass
