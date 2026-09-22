"""Checkpoint names must map consistently for both tensors and selectors."""

import importlib.util
from pathlib import Path
import pytest

path = (
    Path(__file__).resolve().parents[3] / "vllm_fl/patches/deepseek_v4_hf_checkpoint.py"
)
spec = importlib.util.spec_from_file_location("deepseek_v4_hf_checkpoint", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize(
    "source,target",
    [
        ("model.layers.0.attn_hc.base", "layers.0.hc_attn_base"),
        ("model.layers.2.ffn_hc.scale", "layers.2.hc_ffn_scale"),
        ("model.hc_head.hc_fn", "hc_head_fn"),
        ("model.layers.2.self_attn.kv_proj.qweight", "layers.2.attn.wkv.qweight"),
        ("model.layers.2.self_attn.o_a_proj.weight", "layers.2.attn.wo_a.weight"),
        ("model.layers.2.self_attn.q_a_norm.weight", "layers.2.attn.q_norm.weight"),
        (
            "model.layers.2.self_attn.compressor.indexer.kv_proj.weight",
            "layers.2.attn.indexer.compressor.wkv.weight",
        ),
        (
            "model.layers.2.self_attn.compressor.indexer.q_b_proj.weight",
            "layers.2.attn.indexer.wq_b.weight",
        ),
        (
            "model.layers.2.self_attn.compressor.indexer.scorer.weights_proj.weight",
            "layers.2.attn.indexer.weights_proj.weight",
        ),
        (
            "model.layers.2.self_attn.compressor.indexer.position_bias",
            "layers.2.attn.indexer.compressor.ape",
        ),
        (
            "model.layers.2.self_attn.compressor.kv_norm.weight",
            "layers.2.attn.compressor.norm.weight",
        ),
        (
            "model.layers.2.mlp.experts.10.up_proj.scales",
            "layers.2.ffn.experts.10.w3.scales",
        ),
        (
            "model.layers.2.mlp.shared_experts.down_proj",
            "layers.2.ffn.shared_experts.w2",
        ),
        ("model.layers.3.mlp.gate.e_score_correction_bias", "layers.3.ffn.gate.bias"),
        ("model.embed_tokens.weight", "embed.weight"),
        ("lm_head.weight", "head.weight"),
    ],
)
def test_hf_mapping(source, target):
    assert module.hf_to_original_name(source) == target


@pytest.mark.parametrize(
    "name",
    [
        "layers.2.attn.wkv.weight_packed",
        "layers.2.attn.indexer.compressor.wkv.weight",
        "layers.2.ffn.shared_experts.w2.weight_packed",
        "hc_head_base",
        "embed.weight",
    ],
)
def test_original_checkpoint_names_are_unchanged(name):
    assert module.hf_to_original_name(name) == name


def test_runtime_mapper_preserves_payload_and_selects_fused_quantized_layers():
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.quantization.utils.gptq_utils import (
        is_layer_gptq_quantized,
    )
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM

    assert module.install_deepseek_v4_hf_checkpoint()
    mapper = DeepseekV4ForCausalLM.hf_to_vllm_mapper
    payload = object()
    assert list(
        mapper.apply([("model.layers.2.self_attn.kv_proj.qweight", payload)])
    ) == [("model.layers.2.attn.wkv.qweight", payload)]
    assert mapper.apply_list(["layers.2.attn.wq_a.scale"]) == [
        "model.layers.2.attn.wq_a.weight_scale_inv"
    ]
    assert mapper.apply_list(["model.layers.2.attn_hc.scale"]) == [
        "model.layers.2.hc_attn_scale"
    ]
    selected = mapper.apply_list(
        [
            "model.layers.2.self_attn.q_a_proj",
            "model.layers.2.self_attn.kv_proj",
            "model.layers.2.mlp.shared_experts.gate_proj",
            "model.layers.2.mlp.shared_experts.up_proj",
        ]
    )
    packed = DeepseekV4ForCausalLM.packed_modules_mapping
    assert is_layer_gptq_quantized(
        "model.layers.2.attn.fused_wqa_wkv", selected, packed
    )
    assert is_layer_gptq_quantized(
        "model.layers.2.ffn.shared_experts.gate_up_proj", selected, packed
    )
    assert not is_layer_gptq_quantized("model.layers.2.attn.wo_a", selected, packed)
    assert not is_layer_gptq_quantized(
        "model.layers.2.attn.compressor.fused_wkv_wgate", selected, packed
    )
    assert module.install_deepseek_v4_hf_checkpoint()
    assert DeepseekV4ForCausalLM.hf_to_vllm_mapper is mapper


def test_runtime_mapper_survives_fp8_cache_scale_composition():
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.models.utils import WeightsMapper
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM

    assert module.install_deepseek_v4_hf_checkpoint()
    mapper = DeepseekV4ForCausalLM.hf_to_vllm_mapper
    # AutoWeightsLoader combines its model mapper with a quantization-specific
    # cache mapper. This used to erase the subclass's entire name conversion.
    cache = Fp8Config().get_cache_scale_mapper()
    combined = mapper | cache
    names = [
        "embed.weight",
        "layers.0.attn.wq_a.scale",
        "model.layers.0.self_attn.kv_proj.qweight",
        "layers.0.k_proj.output_scale",
    ]
    expected = [
        "model.embed_tokens.weight",
        "model.layers.0.attn.wq_a.weight_scale_inv",
        "model.layers.0.attn.wkv.qweight",
        "model.layers.0.attn.k_scale",
    ]
    payload = object()
    assert list(combined.apply((name, payload) for name in names)) == [
        (name, payload) for name in expected
    ]
    assert combined.apply_list(names) == expected
    assert combined.apply_dict(dict.fromkeys(names, payload)) == dict.fromkeys(
        expected, payload
    )
    assert (combined | WeightsMapper()).apply_list(names) == expected
    assert mapper.apply_list(["embed.weight"]) == ["model.embed_tokens.weight"]
