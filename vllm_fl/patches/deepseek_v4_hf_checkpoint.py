"""Accept Transformers' DeepSeek V4 names in calibrated GPTQ/AWQ exports."""

import re


def hf_to_original_name(name: str) -> str:
    """Undo the Transformers namespace conversion without touching tensors."""
    name = name.removeprefix("model.")
    if name == "embed_tokens" or name.startswith("embed_tokens."):
        return name.replace("embed_tokens", "embed", 1)
    if name == "lm_head" or name.startswith("lm_head."):
        return name.replace("lm_head", "head", 1)
    if name.startswith("hc_head.hc_"):
        return name.replace("hc_head.hc_", "hc_head_", 1)
    if not name.startswith("layers."):
        return name
    name = re.sub(r"\.attn_hc\.(fn|base|scale)$", r".hc_attn_\1", name)
    name = re.sub(r"\.ffn_hc\.(fn|base|scale)$", r".hc_ffn_\1", name)
    name = name.replace(".input_layernorm", ".attn_norm")
    name = name.replace(".post_attention_layernorm", ".ffn_norm")
    if ".self_attn." in name:
        name = name.replace(".self_attn.", ".attn.")
        name = name.replace(".attn.sinks", ".attn.attn_sink")
        name = name.replace(".attn.q_a_norm", ".attn.q_norm")
        # HF flattens the indexer's compressor into the indexer module.
        name = name.replace(
            ".attn.compressor.indexer.scorer.weights_proj", ".attn.indexer.weights_proj"
        )
        name = name.replace(".attn.compressor.indexer.q_b_proj", ".attn.indexer.wq_b")
        name = name.replace(".attn.compressor.indexer.", ".attn.indexer.compressor.")
        name = name.replace(".compressor.position_bias", ".compressor.ape")
        name = name.replace(".compressor.kv_norm", ".compressor.norm")
        for old, new in [
            ("q_a_proj", "wq_a"),
            ("q_b_proj", "wq_b"),
            ("kv_proj", "wkv"),
            ("o_a_proj", "wo_a"),
            ("o_b_proj", "wo_b"),
            ("gate_proj", "wgate"),
        ]:
            name = re.sub(r"\." + old + r"(?=\.|$)", "." + new, name)
    if ".mlp." in name:
        name = name.replace(".mlp.", ".ffn.")
        for old, new in [("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")]:
            name = re.sub(r"\." + old + r"(?=\.|$)", "." + new, name)
        name = name.replace(".gate.e_score_correction_bias", ".gate.bias")
    return name


def install_deepseek_v4_hf_checkpoint():
    from functools import wraps
    from vllm.platforms import current_platform

    if (
        not current_platform.is_cuda()
        or getattr(current_platform, "vendor_name", "nvidia") != "nvidia"
    ):
        return False
    from vllm.model_executor.models.utils import WeightsMapper
    from vllm.models.deepseek_v4.nvidia import model as implementation

    factory = implementation._make_deepseek_v4_weights_mapper
    if getattr(factory, "_fl_hf_checkpoint", False):
        return True

    class HFCheckpointMapper(WeightsMapper):
        def __init__(self, original):
            super().__init__()
            self.original = original

        def _map_name(self, key):
            return self.original._map_name(hf_to_original_name(key))

    @wraps(factory)
    def make_mapper(expert_dtype):
        return HFCheckpointMapper(factory(expert_dtype))

    make_mapper._fl_hf_checkpoint = True
    implementation._make_deepseek_v4_weights_mapper = make_mapper
    cls = implementation.DeepseekV4ForCausalLM
    cls.hf_to_vllm_mapper = make_mapper("fp4")
    cls.packed_modules_mapping = {
        **getattr(cls, "packed_modules_mapping", {}),
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }
    return True
