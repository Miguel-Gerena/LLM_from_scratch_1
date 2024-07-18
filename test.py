import numpy as np
from collections import Counter
import torch
import math
import numpy

from tests.adapters import (
    run_gelu,
    run_multihead_self_attention,
    run_positionwise_feedforward,
    run_rmsnorm,
    run_scaled_dot_product_attention,
    run_transformer_block,
    run_transformer_lm, run_cross_entropy, run_gradient_clipping, run_softmax, run_get_batch
)
import pytest

from tests.common import FIXTURES_PATH


def test_multihead_self_attention():
    reference_weights = torch.load(
        FIXTURES_PATH / "unbatched_multihead_self_attention_weights.pt"
    )
    in_features = torch.load(FIXTURES_PATH / "in_features.pt")
    expected_output = torch.load(
        FIXTURES_PATH / "unbatched_multihead_self_attention_expected_output.pt"
    )
    d_model = 64
    num_heads = 2
    attn_pdrop = 0.0
    # mha = MHSelfAttention(d_model, num_heads, 0, attn_pdrop, False)
    # mha.set_weights_from_dict(reference_weights)
    # mha(in_features)
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=num_heads,
        attn_pdrop=attn_pdrop,
        weights=reference_weights,
        in_features=in_features,
    )
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-6
    )





def test_transformer_lm():
    torch.manual_seed(42)
    vocab_size = 100
    context_length = 64
    d_model = 128
    num_layers = 2
    num_heads = 2
    d_ff = d_model * 4
    attn_pdrop = 0.0
    residual_pdrop = 0.0

    reference_weights = torch.load(FIXTURES_PATH / "transformer_lm_weights.pt")
    in_indices = torch.load(FIXTURES_PATH / "in_indices.pt")
    expected_output = torch.load(FIXTURES_PATH / "transformer_lm_expected_output.pt")
    actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        attn_pdrop=attn_pdrop,
        residual_pdrop=residual_pdrop,
        weights=reference_weights,
        in_indices=in_indices,
    )
    numpy.testing.assert_allclose(
        actual_output.detach().numpy(), expected_output.detach().numpy(), atol=1e-4
    )


def test_gradient_clipping():
    tensors = [torch.randn((5, 5)) for _ in range(6)]
    max_norm = 1e-2

    t1 = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    # Test freezing one parameter.
    t1[-1].requires_grad_(False)

    loss = torch.cat(t1).sum()
    loss.backward()
    torch.nn.utils.clip_grad.clip_grad_norm_(t1, max_norm)
    t1_grads = [torch.clone(t.grad) for t in t1 if t.grad is not None]

    t1_c = tuple(torch.nn.Parameter(torch.clone(t)) for t in tensors)
    t1_c[-1].requires_grad_(False)
    loss_c = torch.cat(t1_c).sum()
    loss_c.backward()
    run_gradient_clipping(t1_c, max_norm)
    t1_c_grads = [torch.clone(t.grad) for t in t1_c if t.grad is not None]

    assert len(t1_grads) == len(t1_c_grads)

    for t1_grad, t1_c_grad in zip(t1_grads, t1_c_grads):
        numpy.testing.assert_allclose(
            t1_grad.detach().numpy(),
            t1_c_grad.detach().numpy(),
            atol=1e-6,
        )



def test_get_batch():
    dataset = np.arange(0, 100)
    context_length = 7
    batch_size = 32
    device = "cpu"

    # Sanity check to make sure that the random samples are indeed somewhat random.
    starting_indices = Counter()
    num_iters = 1000
    for _ in range(num_iters):
        x, y = run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=device,
        )

        # Make sure the shape is correct
        assert x.shape == (batch_size, context_length)
        assert y.shape == (batch_size, context_length)

        # Make sure the y's are always offset by 1
        np.testing.assert_allclose((x + 1).detach().numpy(), y.detach().numpy())

        starting_indices.update(x[:, 0].tolist())

    # Make sure we never sample an invalid start index
    num_possible_starting_indices = len(dataset) - context_length
    assert max(starting_indices) == num_possible_starting_indices - 1
    assert min(starting_indices) == 0
    # Expected # of times that we see each starting index
    expected_count = (num_iters * batch_size) / num_possible_starting_indices
    standard_deviation = math.sqrt(
        (num_iters * batch_size)
        * (1 / num_possible_starting_indices)
        * (1 - (1 / num_possible_starting_indices))
    )
    # Range for expected outcomes (mu +/- 5sigma). For a given index,
    # this should happen 99.99994% of the time of the time.
    # So, in the case where we have 93 possible start indices,
    # the entire test should pass with 99.9944202% of the time
    occurrences_lower_bound = expected_count - 5 * standard_deviation
    occurrences_upper_bound = expected_count + 5 * standard_deviation

    for starting_index, count in starting_indices.items():
        if count < occurrences_lower_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at least {occurrences_lower_bound}"
            )
        if count > occurrences_upper_bound:
            raise ValueError(
                f"Starting index {starting_index} occurs {count} times, but expected at most {occurrences_upper_bound}"
            )

    with pytest.raises((RuntimeError, AssertionError)) as excinfo:
        # We're assuming that cuda:99 is an invalid device ordinal.
        # Just adding this here to make sure that the device flag is
        # being handled.
        run_get_batch(
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device="cuda:99",
        )
        assert "CUDA error" in str(
            excinfo.value
        ) or "Torch not compiled with CUDA enabled" in str(excinfo.value)
test_get_batch()