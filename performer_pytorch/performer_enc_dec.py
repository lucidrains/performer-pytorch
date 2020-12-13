import re
import torch
from torch import nn
from performer_pytorch.performer_pytorch import PerformerLM
from performer_pytorch.autoregressive_wrapper import AutoregressiveWrapper

ENC_PREFIX = 'enc_'
DEC_PREFIX = 'dec_'

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return bool(re.match(f'^{prefix}', str))

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(lambda x: string_begins_with(prefix, x), d)

def group_by_key_prefix_and_remove_prefix(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(lambda x: string_begins_with(prefix, x), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def extract_enc_dec_kwargs(kwargs):
    enc_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(ENC_PREFIX, kwargs)
    dec_kwargs, kwargs = group_by_key_prefix_and_remove_prefix(DEC_PREFIX, kwargs)
    return enc_kwargs, dec_kwargs, kwargs

def extract_and_set_enc_dec_kwargs(kwargs):
    enc_kwargs, dec_kwargs, kwargs = extract_enc_dec_kwargs(kwargs)
    if 'mask' in enc_kwargs:
        dec_kwargs.setdefault('context_mask', enc_kwargs['mask'])
    return enc_kwargs, dec_kwargs, kwargs

class PerformerEncDec(nn.Module):
    def __init__(
        self,
        dim,
        ignore_index = 0,
        pad_value = 0,
        tie_token_embeds = False,
        no_projection = False,
        **kwargs
    ):
        super().__init__()
        enc_kwargs, dec_kwargs, _ = extract_enc_dec_kwargs(kwargs)
        
        assert 'dim' not in dec_kwargs and 'dim' not in enc_kwargs, 'you must set the dim for both encoder and decoder'

        enc_kwargs['dim'] = dec_kwargs['dim'] = dim
        enc_kwargs['no_projection'] = dec_kwargs['no_projection'] = no_projection

        dec_kwargs['causal'] = True
        dec_kwargs['cross_attend'] = True

        enc = PerformerLM(**enc_kwargs)
        dec = PerformerLM(**dec_kwargs)

        if tie_token_embeds:
            enc.token_emb = dec.token_emb

        self.enc = enc
        self.dec = AutoregressiveWrapper(dec, ignore_index = ignore_index, pad_value = pad_value)

    @torch.no_grad()
    def generate(self, seq_in, seq_out_start, seq_len, **kwargs):
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        encodings = self.enc(seq_in, return_encodings = True, **enc_kwargs)
        return self.dec.generate(seq_out_start, seq_len, context = encodings, **{**dec_kwargs, **kwargs})

    def forward(self, seq_in, seq_out, enc_mask = None, **kwargs):
        enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
        encodings = self.enc(seq_in, mask = enc_mask, return_encodings = True, **enc_kwargs)
        return self.dec(seq_out, context = encodings, context_mask = enc_mask, **dec_kwargs)