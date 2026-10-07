"""Positive-feature attention must exclude padding from normalization."""

import math

import pytest
import torch
import torch.nn.functional as F

from performer_pytorch import CrossAttention, SelfAttention


def feature_maps(query, key, attention):
    """Explicit FAVOR feature definition evaluated only on valid key rows."""
    if attention.no_projection:
        return query.softmax(-1), key.softmax(0)
    projection = attention.projection_matrix
    normalization = query.shape[-1] ** -.25
    q_projected = (query * normalization) @ projection.T
    k_projected = (key * normalization) @ projection.T
    if attention.generalized_attention:
        return F.relu(q_projected) + .001, F.relu(k_projected) + .001
    ratio = 1. / math.sqrt(projection.shape[0])
    q_diag = query.square().sum(-1, keepdim=True) * normalization ** 2 / 2.
    k_diag = key.square().sum(-1, keepdim=True) * normalization ** 2 / 2.
    q_features = ratio * ((q_projected - q_diag - q_projected.max(-1, keepdim=True).values.detach()).exp() + .0001)
    k_features = ratio * ((k_projected - k_diag - k_projected.max().detach()).exp() + .0001)
    return q_features, k_features


def dense_feature_oracle(module, x, context, mask):
    heads, width = module.heads, module.to_q.out_features // module.heads
    q = module.to_q(x).reshape(x.shape[0], x.shape[1], heads, width).transpose(1, 2)
    k = module.to_k(context).reshape(context.shape[0], context.shape[1], heads, width).transpose(1, 2)
    v = module.to_v(context).reshape(context.shape[0], context.shape[1], heads, width).transpose(1, 2)
    batch_outputs = []
    for batch_index in range(x.shape[0]):
        head_outputs = []
        for head in range(heads):
            valid_key = k[batch_index, head, mask[batch_index]]
            valid_value = v[batch_index, head, mask[batch_index]]
            if valid_key.shape[0] == 0:
                # Retain the genuine differentiable zero contract for each input.
                zero = (q[batch_index, head].sum() + k[batch_index, head].sum() + v[batch_index, head].sum()) * 0.
                result = x.new_zeros(x.shape[1], width) + zero
            else:
                q_features, k_features = feature_maps(q[batch_index, head], valid_key, module.fast_attention)
                scores = q_features @ k_features.T
                result = (scores / scores.sum(-1, keepdim=True)) @ valid_value
            head_outputs.append(result)
        batch_outputs.append(torch.stack(head_outputs))
    output = torch.stack(batch_outputs).transpose(1, 2).reshape(x.shape[0], x.shape[1], heads * width)
    return module.to_out(output)


@pytest.mark.parametrize("kind", ["softmax", "generalized", "no_projection"])
@pytest.mark.parametrize("attention_type", ["self", "cross"])
@pytest.mark.parametrize("pattern", ["padding", "holes", "all_masked", "all_valid"])
def test_native_attention_masks_keys_in_the_feature_denominator(kind, attention_type, pattern):
    torch.manual_seed(11)
    cls = SelfAttention if attention_type == "self" else CrossAttention
    module = cls(dim=8, heads=2, dim_head=4, nb_features=5, dropout=0.,
                 generalized_attention=kind == "generalized", no_projection=kind == "no_projection").double()
    x = torch.linspace(-.7, .9, 2 * 7 * 8, dtype=torch.float64).reshape(2, 7, 8)
    context = torch.linspace(.8, -.6, 2 * 7 * 8, dtype=torch.float64).reshape(2, 7, 8)
    mask = torch.ones(2, 7, dtype=torch.bool)
    if pattern == "padding":
        mask[0, 3:] = False
        mask[1, 5:] = False
    elif pattern == "holes":
        mask[:, [1, 3, 5]] = False
    elif pattern == "all_masked":
        mask[0] = False
    if attention_type == "self":
        x = x.masked_fill(~mask[..., None], 30.).requires_grad_()
        actual = module(x, mask=mask)
        expected = dense_feature_oracle(module, x, x, mask)
        inputs = (x,)
    else:
        x.requires_grad_()
        context = context.masked_fill(~mask[..., None], 30.).requires_grad_()
        actual = module(x, context=context, context_mask=mask)
        expected = dense_feature_oracle(module, x, context, mask)
        inputs = (x, context)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    coefficients = torch.linspace(.1, 1.2, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    parameters = inputs + (module.to_q.weight, module.to_k.weight, module.to_v.weight, module.to_out.weight)
    actual_gradients = torch.autograd.grad((actual * coefficients).sum(), parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad((expected * coefficients).sum(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)
        assert torch.isfinite(actual_gradient).all()
    if attention_type == "cross":
        assert torch.count_nonzero(actual_gradients[1][~mask]) == 0


@pytest.mark.parametrize("kind", ["softmax", "generalized", "no_projection"])
def test_padding_does_not_change_real_graph_node_outputs(kind):
    torch.manual_seed(17)
    module = SelfAttention(dim=8, heads=2, dim_head=4, nb_features=5, dropout=0.,
                           generalized_attention=kind == "generalized", no_projection=kind == "no_projection").double()
    real_nodes = torch.linspace(-.7, .9, 3 * 8, dtype=torch.float64).reshape(1, 3, 8)
    expected = module(real_nodes)
    for padding in (1, 4):
        for pad_value in (0., 30.):
            padded = torch.cat([real_nodes, torch.full((1, padding, 8), pad_value, dtype=torch.float64)], 1)
            mask = torch.arange(3 + padding)[None, :] < 3
            actual = module(padded, mask=mask)[:, :3]
            torch.testing.assert_close(actual, expected)



@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('pattern', ['padding', 'holes'])
@pytest.mark.parametrize('include_all_masked', [False, True])
def test_masked_softmax_features_do_not_overflow_in_an_inactive_exponential(dtype, pattern, include_all_masked):
    module = CrossAttention(dim=4, heads=1, dim_head=4, nb_features=4, dropout=0.).to(dtype=dtype)
    with torch.no_grad():
        for layer in (module.to_q, module.to_k, module.to_v, module.to_out):
            layer.weight.copy_(torch.eye(4, dtype=dtype))
            if layer.bias is not None:
                layer.bias.zero_()
        module.fast_attention.projection_matrix.copy_(torch.eye(4, dtype=dtype))
    # This public fixed projection can also participate in differentiation.
    module.fast_attention.projection_matrix.requires_grad_()
    mask = torch.tensor([[True, True, False, False], [True, True, False, False]])
    if pattern == 'holes':
        mask = torch.tensor([[True, False, True, False], [True, False, True, False]])
    if include_all_masked:
        mask[1] = False
    magnitude = -150. if dtype == torch.float32 else -1100.
    # Keep values bounded independently of the key-feature overflow trigger.
    with torch.no_grad():
        module.to_v.weight.div_(abs(magnitude))
    context = torch.zeros(2, 4, 4, dtype=dtype).masked_fill(mask[..., None], magnitude).requires_grad_()
    x = torch.linspace(-.3, .4, 2 * 3 * 4, dtype=dtype).reshape(2, 3, 4).requires_grad_()
    actual = module(x, context=context, context_mask=mask)
    expected = dense_feature_oracle(module, x, context, mask)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=3e-6, atol=3e-6)
    # A positive cotangent avoids cancellation of the extreme value-weight gradient.
    weights = torch.linspace(.1, 1., actual.numel(), dtype=dtype).reshape_as(actual)
    targets = (x, context, *tuple(module.parameters()), module.fast_attention.projection_matrix)
    actual_gradients = torch.autograd.grad((actual * weights).sum(), targets, retain_graph=True)
    expected_gradients = torch.autograd.grad((expected * weights).sum(), targets)
    for observed, reference in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, reference, rtol=3e-6, atol=3e-6)
    assert torch.count_nonzero(actual_gradients[1][~mask]) == 0
