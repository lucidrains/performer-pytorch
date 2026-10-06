"""Redrawing random features must not invalidate pending forward graphs."""

import math

import pytest
import torch
import torch.nn.functional as F

from performer_pytorch import SelfAttention
from performer_pytorch.performer_pytorch import FastAttention, ProjectionUpdater


def dense_feature_attention(q, k, v, projection, generalized):
    """Explicit positive-feature matrices with immutable per-forward projections."""
    outputs = []
    normalization = q.shape[-1] ** -.25
    for batch in range(q.shape[0]):
        heads = []
        for head in range(q.shape[1]):
            query, key = q[batch, head], k[batch, head]
            q_projected = (query * normalization) @ projection.T
            k_projected = (key * normalization) @ projection.T
            if generalized:
                q_features = F.relu(q_projected) + .001
                k_features = F.relu(k_projected) + .001
            else:
                ratio = 1. / math.sqrt(projection.shape[0])
                q_diag = query.square().sum(-1, keepdim=True) * normalization ** 2 / 2.
                k_diag = key.square().sum(-1, keepdim=True) * normalization ** 2 / 2.
                q_features = ratio * ((q_projected - q_diag - q_projected.max(-1, keepdim=True).values.detach()).exp() + .0001)
                k_features = ratio * ((k_projected - k_diag - k_projected.max().detach()).exp() + .0001)
            scores = q_features @ k_features.T
            heads.append((scores / scores.sum(-1, keepdim=True)) @ v[batch, head])
        outputs.append(torch.stack(heads))
    return torch.stack(outputs)


def dense_module_output(module, x, projection):
    batch, length = x.shape[:2]
    heads, width = module.heads, module.to_q.out_features // module.heads
    q, k, v = [layer(x).reshape(batch, length, heads, width).transpose(1, 2)
               for layer in (module.to_q, module.to_k, module.to_v)]
    output = dense_feature_attention(q, k, v, projection, module.fast_attention.generalized_attention)
    return module.to_out(output.transpose(1, 2).reshape(batch, length, heads * width))


@pytest.mark.parametrize("generalized", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_manual_redraw_preserves_pending_input_gradients(generalized, dtype, shape):
    torch.manual_seed(13)
    batch, heads = shape
    module = FastAttention(dim_heads=4, nb_features=6, generalized_attention=generalized).to(dtype)
    values = [torch.randn(batch, heads, 5, 4, dtype=dtype) * .3 for _ in range(3)]
    actual_inputs = [value.clone().requires_grad_() for value in values]
    expected_inputs = [value.clone().requires_grad_() for value in values]
    projection = module.projection_matrix.clone()
    expected = dense_feature_attention(*expected_inputs, projection, generalized)
    actual = module(*actual_inputs)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-10
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    weights = torch.linspace(-.7, .9, actual.numel(), dtype=dtype).reshape_as(actual)
    expected_gradients = torch.autograd.grad((expected * weights).sum(), expected_inputs)

    module.redraw_projection_matrix(actual.device)
    assert module.projection_matrix.dtype == dtype
    assert not torch.equal(module.projection_matrix, projection)
    actual_gradients = torch.autograd.grad((actual * weights).sum(), actual_inputs)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("generalized", [False, True])
def test_automatic_redraw_allows_two_forwards_before_backward(generalized):
    torch.manual_seed(19)
    module = SelfAttention(dim=8, heads=1, dim_head=8, nb_features=6,
                           generalized_attention=generalized, dropout=0.).double()
    updater = ProjectionUpdater(module, feature_redraw_interval=1)
    first = (torch.randn(1, 5, 8, dtype=torch.float64) * .3).requires_grad_()
    second = (torch.randn(1, 5, 8, dtype=torch.float64) * .3).requires_grad_()
    expected_first = first.detach().clone().requires_grad_()
    expected_second = second.detach().clone().requires_grad_()

    updater.redraw_projections()
    first_projection = module.fast_attention.projection_matrix.clone()
    first_output = module(first)
    updater.redraw_projections()
    second_projection = module.fast_attention.projection_matrix.clone()
    second_output = module(second)
    assert not torch.equal(first_projection, second_projection)
    first_expected = dense_module_output(module, expected_first, first_projection)
    second_expected = dense_module_output(module, expected_second, second_projection)
    torch.testing.assert_close(first_output, first_expected, rtol=2e-10, atol=2e-10)
    torch.testing.assert_close(second_output, second_expected, rtol=2e-10, atol=2e-10)
    parameters = tuple(module.parameters())
    expected_loss = first_expected.square().sum() + .3 * second_expected.sin().sum()
    expected_gradients = torch.autograd.grad(expected_loss, (expected_first, expected_second, *parameters))
    actual_loss = first_output.square().sum() + .3 * second_output.sin().sum()
    actual_gradients = torch.autograd.grad(actual_loss, (first, second, *parameters))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-9, atol=3e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_redrawn_projection_remains_a_serialized_buffer(dtype):
    torch.manual_seed(29)
    module = FastAttention(dim_heads=4, nb_features=6).to(dtype)
    module.redraw_projection_matrix(torch.device("cpu"))
    assert "projection_matrix" in dict(module.named_buffers())
    assert "projection_matrix" not in dict(module.named_parameters())
    assert module.projection_matrix.dtype == dtype
    restored = FastAttention(dim_heads=4, nb_features=6).to(dtype)
    restored.load_state_dict(module.state_dict())
    torch.testing.assert_close(restored.projection_matrix, module.projection_matrix, rtol=0., atol=0.)
