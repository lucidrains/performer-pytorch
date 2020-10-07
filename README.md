## Performer - Pytorch (wip)

An implementation of Performer, a linear attention-based transformer, in Pytorch.

## Install

```bash
$ pip install performer-pytorch
```

## Usage

```python
import torch
from performer_pytorch import Performer

model = Performer(
    dim = 512,
    depth = 1,
    heads = 8,
    causal = True   # auto-regressive or not
)

x = torch.randn(1, 1024, 512)
model(x) # (1, 1024, 512)
```

### Todo

1. ~~causal variant~~
2. renormalizations
3. full language model
4. masking
5. make causal variant efficient memory-wise
6. document fast linear attention class. give better name than 'fast'

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
