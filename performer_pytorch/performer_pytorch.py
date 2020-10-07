import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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
