import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# kernel functions

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []
    for _ in range(nb_full_blocks):
        unstructured_block = torch.randn((nb_columns, nb_columns))
        q, _ = torch.qr(unstructured_block)
        q = q.t()
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        unstructured_block = torch.randn((nb_columns, nb_columns))
        q, _ = torch.qr(unstructured_block)
        q = q.t()
        block_list.append(q[:remaining_rows])

    final_matrix = torch.stack(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns)).norm(dim = 1)
        print(multiplier.shape)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones(nb_rows)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

# classes

class FastAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class FastSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class Performer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x
