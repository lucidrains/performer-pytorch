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

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_self_attention/fast_self_attention.py

def softmax_kernel(data, projection_matrix, *, is_query, normalize_data=True, eps=0.0001):
    if normalize_data:
        data_normalizer = 1.0 / (data.shape[-1] ** 0.25)
    else:
        data_normalizer = 1.0

    ratio = 1.0 / (projection_matrix.shape[0] ** 0.5)

    data_mod_shape = data.shape[:(len(data.shape) - 2)] + projection_matrix.shape
    data_thick_random_matrix = torch.zeros(data_mod_shape) + projection_matrix

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), data_thick_random_matrix)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data -
                    torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash

def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []
    for _ in range(nb_full_blocks):
        unstructured_block = torch.randn((nb_columns, nb_columns), device = device)
        q, _ = torch.qr(unstructured_block)
        q = q.t()
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        unstructured_block = torch.randn((nb_columns, nb_columns), device = device)
        q, _ = torch.qr(unstructured_block)
        q = q.t()
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns)).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,))
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        return self.fn(self.norm(x))

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)

class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features = 256, redraw_projection = True, ortho_scaling = 1):
        super().__init__()
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.redraw_projection = redraw_projection
        if not redraw_projection:
            projection_matrix = gaussian_orthogonal_random_matrix(nb_features, dim_heads, scaling = ortho_scaling)
            self.register_buffer('projection_matrix', projection_matrix)

    def forward(self, q, k, v):
        device = q.device

        if self.redraw_projection:
            projection_matrix = gaussian_orthogonal_random_matrix(self.nb_features, self.dim_heads, scaling = self.ortho_scaling, device = device)
        else:
            projection_matrix = self.projection_matrix

        q_kernel = softmax_kernel(q, projection_matrix, is_query = True)
        k_kernel = softmax_kernel(k, projection_matrix, is_query = False)

        context = torch.einsum('...nd,...ne->...de', k_kernel, v)
        out = torch.einsum('...de,...nd->...ne', context, q_kernel)

        return out

class FastSelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, nb_features = 256, redraw_projection = True):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        self.fast_attention = FastAttention(dim // heads, nb_features, redraw_projection)

        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h)

        out = self.fast_attention(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Performer(nn.Module):
    def __init__(self, dim, depth, heads, ff_mult = 4):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([
                Residual(PreNorm(dim, FastSelfAttention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, ff_mult)))
            ])
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
