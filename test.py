import numpy
import torch


from tests.adapters import (
    run_gelu,
    run_multihead_self_attention,
    run_positionwise_feedforward,
    run_rmsnorm,
    run_scaled_dot_product_attention,
    run_transformer_block,
    run_transformer_lm,
)
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


test_transformer_lm()