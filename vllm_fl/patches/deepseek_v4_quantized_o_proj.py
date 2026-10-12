"""Use loaded integer linear kernels for DeepSeek V4 output projections."""
from functools import wraps

import torch


def grouped_linear_o_proj(o, positions, cos_sin_cache, wo_a, wo_b, *,
                          n_groups, heads_per_group, nope_dim, rope_dim,
                          o_lora_rank):
    """Inverse RoPE in A16 followed by the configured packed linear kernels.

    ColumnParallelLinear owns all local groups in its output dimension. Apply
    it to each input group and retain the matching output group. This generic
    path supports TP < o_groups without unpacking/re-quantizing the weights;
    it computes extra cross-group products when more than one group is local.
    """
    if o.shape[1:] != (n_groups * heads_per_group, nope_dim + rope_dim):
        raise ValueError('Unexpected DeepSeek V4 output-projection input shape')
    cache = cos_sin_cache[positions.long()]
    cos, sin = cache.chunk(2, dim=-1)
    cos, sin = cos[:, None, :].float(), sin[:, None, :].float()
    rope = o[..., nope_dim:].float().reshape(o.shape[0], o.shape[1], rope_dim // 2, 2)
    even, odd = rope[..., 0], rope[..., 1]
    rotated = torch.stack((even * cos + odd * sin, odd * cos - even * sin), dim=-1)
    inputs = torch.cat((o[..., :nope_dim], rotated.flatten(-2).to(o.dtype)), dim=-1)
    inputs = inputs.reshape(o.shape[0] * n_groups, heads_per_group * (nope_dim + rope_dim))
    projected = wo_a(inputs)
    if isinstance(projected, tuple):
        projected = projected[0]
    projected = projected.reshape(o.shape[0], n_groups, n_groups, o_lora_rank)
    groups = torch.arange(n_groups, device=o.device)
    selected = projected[:, groups, groups, :].reshape(o.shape[0], n_groups * o_lora_rank)
    return wo_b(selected)


def install_deepseek_v4_quantized_o_proj():
    from vllm.platforms import current_platform
    if (not current_platform.is_cuda()
            or getattr(current_platform, 'vendor_name', 'nvidia') != 'nvidia'):
        return False
    from vllm.models.deepseek_v4.nvidia.flashmla import DeepseekV4FlashMLAAttention

    classes = [DeepseekV4FlashMLAAttention]
    for cls in classes:
        original = cls._o_proj
        if getattr(original, '_fl_integer_o_proj', False):
            continue

        @wraps(original)
        def forward(self, o, positions, _original=original):
            if hasattr(self.wo_a, 'weight_scale_inv'):
                return _original(self, o, positions)
            return grouped_linear_o_proj(
                o, positions, self.rotary_emb.cos_sin_cache, self.wo_a, self.wo_b,
                n_groups=self.n_local_groups,
                heads_per_group=self.n_local_heads // self.n_local_groups,
                nope_dim=self.nope_head_dim, rope_dim=self.rope_head_dim,
                o_lora_rank=self.o_lora_rank)

        forward._fl_integer_o_proj = True
        cls._o_proj = forward
    return True
