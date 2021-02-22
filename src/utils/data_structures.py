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
