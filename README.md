<img src="./favor+.png" width="500px"></img>

## Performer - Pytorch (wip)

[![PyPI version](https://badge.fury.io/py/performer-pytorch.svg)](https://badge.fury.io/py/performer-pytorch)

An implementation of <a href="https://arxiv.org/abs/2009.14794">Performer</a>, a linear attention-based transformer variant with a **F**ast **A**ttention **V**ia positive **O**rthogonal **R**andom features approach (FAVOR+).

## Install

```bash
$ pip install performer-pytorch
```

## Usage

Performer Language Model

```python
import torch
from performer_pytorch import PerformerLM

model = PerformerLM(
    num_tokens = 20000,
    max_seq_len = 2048,             # max sequence length
    dim = 512,                      # dimension
    depth = 6,                      # layers
    heads = 8,                      # heads
    causal = False,                 # auto-regressive or not
    nb_features = 256,              # random features dimension, set to 256 as default in original repository
    generalized_attention = False,  # defaults to softmax approximation, but can be set to True for generalized attention
    kernel_fn = nn.ReLU(),          # the kernel function to be used, if generalized attention is turned on, defaults to Relu
    reversible = True,              # reversible layers, from Reformer paper
    ff_chunks = 10,                 # chunk feedforward layer, from Reformer paper
)

x = torch.randint(0, 20000, (1, 2048))
model(x) # (1, 2048, 20000)
```

Plain Performer, if you are working with say images or other modalities

```python
import torch
from performer_pytorch import Performer

model = Performer(
    dim = 512,
    depth = 1,
    heads = 8,
    causal = True
)

x = torch.randn(1, 2048, 512)
model(x) # (1, 2048, 512)
```

### Todo

1. ~~causal variant~~
2. renormalizations
3. ~~full language model~~
4. masking
5. make causal variant efficient memory-wise
6. ~~add enwik8 training~~
7. try to implement gating in https://openreview.net/forum?id=QtTKTdVrFBB, contingent on (5)
8. ~~Email authors about how low nb_features can go. Running out of memory easily.~~
9. Benchmark against elu(x) + 1 kernel and measure impact of kernel choice

## Citations

```bibtex
@misc{choromanski2020rethinking,
    title={Rethinking Attention with Performers}, 
    author={Krzysztof Choromanski and Valerii Likhosherstov and David Dohan and Xingyou Song and Andreea Gane and Tamas Sarlos and Peter Hawkins and Jared Davis and Afroz Mohiuddin and Lukasz Kaiser and David Belanger and Lucy Colwell and Adrian Weller},
    year={2020},
    eprint={2009.14794},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

```bibtex
@inproceedings{kitaev2020reformer,
    title       = {Reformer: The Efficient Transformer},
    author      = {Nikita Kitaev and Lukasz Kaiser and Anselm Levskaya},
    booktitle   = {International Conference on Learning Representations},
    year        = {2020},
    url         = {https://openreview.net/forum?id=rkgNKkHtvB}
}
```
