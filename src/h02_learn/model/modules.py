import torch
import torch.nn as nn

from utils import constants


class ItemMH4(object):
    def __init__(self, heads, l, r):
        self.heads = heads
        self.l = l
        self.r = r
        if len(heads) == 4:
            self.key = (self.heads[0], self.heads[1], self.heads[2], self.heads[3])
        elif len(heads) == 3:
            self.key = (self.heads[0], self.heads[1], self.heads[2], -1)
        elif len(heads) == 2:
            self.key = (self.heads[0], self.heads[1], -1, -1)
        elif len(heads) == 1:
            self.key = (self.heads[0], -1, -1, -1)


    def __str__(self):
        return "Item:\t" + str(self.key)

    def __repr__(self):
        return "Item:\t" + str(self.key)

    def __hash__(self):
        return hash(self.key)


class TreeLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear_tree = nn.Linear(hidden_size * 2, hidden_size).to(device=constants.device)

    def forward(self, words, head_index, mod_index):
        clown_tensor = torch.zeros_like(words).to(device=constants.device)
        clown_tensor[head_index, :] = words[mod_index, :].clone().detach()
        rep = torch.cat([words, clown_tensor], dim=-1)
        rep = nn.Tanh()(self.linear_tree(rep))
        return rep


class Bilinear(nn.Module):
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
