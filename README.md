## Performer - Pytorch (wip)

[![PyPI version](https://badge.fury.io/py/performer-pytorch.svg)](https://badge.fury.io/py/performer-pytorch)

An implementation of Performer, a linear attention-based transformer, in Pytorch.

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
    max_seq_len = 2048,     # max sequence length
    dim = 512,              # dimension
    depth = 6,              # layers
    heads = 8,              # heads
    causal = False          # auto-regressive or not
)

x = torch.randint(0, 20000, (1, 2048))
model(x) # (1, 2048, 20000)
```

Performer model, if you are working with say images

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
6. document fast linear attention class. give better name than 'fast'
7. add enwik8 training

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
