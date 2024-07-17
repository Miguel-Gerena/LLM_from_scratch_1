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

test_multihead_self_attention()



