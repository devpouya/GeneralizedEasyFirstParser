
class BaseItem(object):

    def __init__(self, i, j, h,item_l, item_r):
        self.i, self.j, self.h = i, j, h
        self.l = item_l
        self.r = item_r
        self.subtrees = {}
        self.arcs = []

    def __str__(self):
        return "BaseItem:\t" + str((self.i, self.j, self.h))

    def __repr__(self):
        return "BaseItem:\t" + str((self.i, self.j, self.h))

    def __hash__(self):
        return hash((self.i, self.j, self.h))


class Chart(object):

    def __init__(self):
        self.chart = {}

    def __getitem__(self, key):
        if isinstance(key, BaseItem):
            return self.chart[(key.i, key.j, key.h)]
        else:
            return self.chart[key]

    def __setitem__(self, key, item):
        if isinstance(key, BaseItem):
            self.chart[(key.i, key.j, key.h)] = item
        else:
            self.chart[key] = item

    def __iter__(self):
        return iter(self.chart)


class BaseAgenda(object):

    def __init__(self):
        self.locator = {}
        self.Q = []

    def pop(self):
        """ super slow ! """
        tmp = self.Q[0]
        self.Q = self.Q[1:]
        return tmp

    def empty(self):
        return True if len(self.Q) == 0 else False

    def locate(self, item):
        pass

    def percolate(self):
        pass

    def __len__(self):
        return len(self.Q)

    def __iter__(self):
        return iter(self.locator)

    def __getitem__(self, key):
        return self.locator[key]

    def __setitem__(self, key, item):
        """ super slow ! """
        if (item.i, item.j, item.h) not in self:
            self.locator[(item.i, item.j, item.h)] = item
            self.Q.append(item)
            #self.Q.sort(key=lambda x: -x.w)
        else:
            self.locator[key] = item
            #self.Q.sort(key=lambda x: -x.w)




class BaseStack:

    def __init__(self):
        # stack is implemented as a list
        # each element is a tuple (TENSOR A, INT INDEX)
        self.stack = []

    def push(self, x):
        # add to the end of the list (slow)
        self.stack.append(x)

    def pop(self):
        # remove and return last element
        return self.stack.pop(-1)

    def pop_second(self):
        # remove and return second to last element
        return self.stack.pop(-2)
    def pop_third(self):
        return self.stack.pop(-3)

    def second(self):
        # return the second element
        return self.stack[-2]
    def third(self):
        return self.stack[-3]

    def set_second(self, x):
        # set the second element's tensor to x, keeping the index
        (_, ind) = self.stack[-2]
        self.stack[-2] = (x, ind)

    def top(self):
        # return the top of the stack
        return self.stack[-1]

    def set_top(self, x):
        # set the top element's tensor to x, keeping the index
        (_, ind) = self.stack[-1]
        self.stack[-1] = (x, ind)

    def get_len(self):
        # lol get the length not necessary lol
        return len(self.stack)

class BaseBuffer:

    def __init__(self, sentence):
        # initialized with the words in the sentence
        # each element is a tuple (TENSOR WORD, INT INDEX)
        self.buffer = sentence #[(word, i) for i, word in enumerate(sentence)]

    def pop_left(self):
        # pop left (first) element
        return self.buffer.pop(0)

    def left(self):
        # return first element
        return self.buffer[0]

    def put_left(self, x):
        self.buffer[0] = x

    def get_len(self):
        # lol why did I implement this?
        return len(self.buffer)

    def last(self):
        return self.buffer[-1]

class BaseArcs:

    def __init__(self):
        # a list of tuples (u,v) indicating an arc u --> v
        self.arcs = []

    def add_arc(self, i, j):
        # adds arc i --> j
        self.arcs.append((i, j))

    def has_incoming(self, v):
        # check if node v has an incoming arc
        for (src, dst) in self.arcs:
            if dst == v:
                return True
        return False
