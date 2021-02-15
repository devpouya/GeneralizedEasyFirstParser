from utils import constants


def children(heads, sentence):
    children = []
    for i in range(len(sentence)):
        children_i = []
        for j in range(len(sentence)):
            if heads[j] == i:
                children_i.append(j)
        children.append(children_i)
    return children


def ready_to_right(a, children, arcs):
    if len(children) == 0:
        return True
    for c in children:
        if (a, c) not in arcs:
            return False
    return True


def arc_standard_oracle(parser, heads):
    s = parser.stack
    b = parser.buffer
    arcs = parser.arcs.arcs
    sentence = [i for i in range(len(heads))]
    all_children = children(heads, sentence)
    if s.get_len() < 2:
        if b.get_len() > 0:
            #print(constants.shift)
            return constants.shift
        else:
            #print("DONE")
            return "DONE"
    elif heads[s.top()[1]] == s.second()[1]:
        #print(constants.reduce_l)
        return constants.reduce_l
    elif heads[s.second()[1]] == s.top()[1]:
        if ready_to_right(s.top()[1], all_children[s.top()[1]], arcs):
            #print(constants.reduce_r)
            return constants.reduce_r
        else:
            #print(constants.shift)
            return constants.shift
    else:
        #print(constants.shift)
        return constants.shift


def arc_eager_oracle(sentence, heads):
    pass


def hybrid_oracle(sentence, heads):
    pass


def non_projective_oracle(sentence, heads):
    pass


def easy_first_oracle(sentence, heads):
    pass
