import pytest
import torch
from torch import nn

from vllm_fl.patches.deepseek_v4_quantized_o_proj import grouped_linear_o_proj


@pytest.mark.parametrize('groups', [1, 2, 8])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_grouped_projection_matches_independent_complex_rope_reference(groups, dtype):
    torch.manual_seed(42)
    tokens, heads, dim, rope_dim, rank = 5, 2, 16, 8, 4
    inputs = torch.randn(tokens, groups * heads, dim, dtype=dtype)
    original = inputs.clone()
    angles = torch.randn(20, rope_dim // 2)
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    positions = torch.tensor([0, 3, 7, 8, 12])
    wa = nn.Linear(heads*dim, groups*rank, bias=False, dtype=dtype)
    wb = nn.Linear(groups*rank, 7, bias=False, dtype=dtype)
    actual = grouped_linear_o_proj(inputs, positions, cache, wa, wb,
                                  n_groups=groups, heads_per_group=heads,
                                  nope_dim=dim-rope_dim, rope_dim=rope_dim,
                                  o_lora_rank=rank)
    complex_input = torch.view_as_complex(inputs[..., -rope_dim:].float().reshape(tokens, groups*heads, -1, 2))
    phase = torch.polar(torch.ones_like(angles[positions]), -angles[positions])
    rotated = torch.view_as_real(complex_input * phase[:, None]).flatten(-2).to(dtype)
    reference_inputs = torch.cat((inputs[..., :-rope_dim], rotated), dim=-1).reshape(tokens, groups, heads*dim)
    expected = wb(torch.cat([torch.nn.functional.linear(reference_inputs[:, g], wa.weight[g*rank:(g+1)*rank])
                            for g in range(groups)], dim=-1))
    torch.testing.assert_close(actual, expected, rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
                               atol=0.005 if dtype == torch.bfloat16 else 1e-6)
    assert torch.equal(inputs, original)


def test_installer_is_idempotent():
    from vllm_fl.patches.deepseek_v4_quantized_o_proj import install_deepseek_v4_quantized_o_proj
    from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention
    assert install_deepseek_v4_quantized_o_proj()
    first = DeepseekV4FlashMLAAttention._o_proj
    assert install_deepseek_v4_quantized_o_proj()
    assert DeepseekV4FlashMLAAttention._o_proj is first
