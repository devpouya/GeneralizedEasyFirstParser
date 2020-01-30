import torch
import torch.nn as nn


class Biaffine(nn.Module):
    # pylint: disable=arguments-differ
    def __init__(self, dim_left, dim_right):
        super().__init__()
        self.dim_left = dim_left
        self.dim_right = dim_right

        self.matrix = nn.Parameter(torch.Tensor(dim_left, dim_right))
        self.bias = nn.Parameter(torch.Tensor(1))

        self.linear_l = nn.Linear(dim_left, 1)
        self.linear_r = nn.Linear(dim_right, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.bias, 0.)
        nn.init.xavier_uniform_(self.matrix)

    def forward(self, x_l, x_r):
        # x shape [batch, length_l, length_r]
        x = torch.matmul(x_l, self.matrix)
        x = torch.bmm(x, x_r.transpose(1, 2)) + self.bias

        # x shape [batch, length_l, 1] and [batch, 1, length_r]
        x += self.linear_l(x_l) + self.linear_r(x_r).transpose(1, 2)
        return x


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
