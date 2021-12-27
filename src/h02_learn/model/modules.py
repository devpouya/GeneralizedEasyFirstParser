import torch
import torch.nn as nn

from utils import constants
import torch.nn.functional as F



class Item(object):

    def __init__(self, i, j, h, item_l, item_r):
        self.i, self.j, self.h = i, j, h
        self.l = item_l
        self.r = item_r
        self.subtrees = {}
        self.arcs = []
        self.vector_rep = None
        self.score = torch.tensor([0]).to(device=constants.device)
        self.rel = None
        self.key = (self.i,self.j,self.h)

    def set_vector_rep(self, x):
        self.vector_rep = x
        return self
    def update_score(self, s):
        self.score = s
    def add_rel(self, rel):
        self.rel = rel
        return self
    def __eq__(self, other):
        return self.key == other.key
    def __str__(self):
        return "Item:\t" + str(self.key)

    def __repr__(self):
        return "Item:\t" + str(self.key)

    def __hash__(self):
        return hash(self.key)



class ItemMH4(object):
    def __init__(self, heads,l,r):
        self.heads = heads#[-1]*4

        self.l = l
        self.r =r
        self.rep = None
        self.pre_computed = False
        #for i, item in en
        #umerate(sorted(heads)):
        #    self.heads[i] = heads
        if len(heads) == 4:
            self.key = (self.heads[0],self.heads[1],self.heads[2],self.heads[3])
        elif len(heads) == 3:
            self.key = (self.heads[0], self.heads[1], self.heads[2], -1)
        elif len(heads) == 2:
            self.key = (self.heads[0], self.heads[1], -1, -1)
        elif len(heads) == 1:
            self.key = (self.heads[0], -1, -1, -1)
        #elif len(heads) == 5:
        #    self.key = (self.heads[0], self.heads[1], self.heads[2], self.heads[3],heads[4])


        #self.degree = len([i for i in self.key if i != -1])

    def __str__(self):
        return "Item:\t" + str(self.key)

    def __repr__(self):
        return "Item:\t" + str(self.key)

    def __hash__(self):
        return hash(self.key)

    def set_rep(self, rep):
        self.rep = rep
        self.pre_computed = True




class Bilinear(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, dim_left, dim_right, dim_out):
        super().__init__()
        self.dim_left = dim_left
        self.dim_right = dim_right
        self.dim_out = dim_out

        self.bilinear = nn.Bilinear(dim_left, dim_right, dim_out)
        self.linear_l = nn.Linear(dim_left, dim_out)
        self.linear_r = nn.Linear(dim_right, dim_out)

    def forward(self, x_l, x_r):
        # x shape [batch, length, dim_out]
        x = self.bilinear(x_l, x_r)

        # x shape [batch, length, dim_out] and [batch, length, dim_out]
        x += self.linear_l(x_l) + self.linear_r(x_r)
        return x



