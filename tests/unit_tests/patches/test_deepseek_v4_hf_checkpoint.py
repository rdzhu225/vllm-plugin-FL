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


def _mixed_schemes():
    values = {
        'model.layers.0.self_attn.q_a_proj': {'bits': 8, 'group_size': 128},
        'model.layers.0.self_attn.kv_proj': {'bits': 8, 'group_size': 128},
    }
    for projection in ['gate_proj', 'up_proj', 'down_proj']:
        values[f'model.layers.0.mlp.shared_experts.{projection}'] = {'bits': 8, 'group_size': 128}
        for expert in range(2):
            values[f'model.layers.0.mlp.experts.{expert}.{projection}'] = {'bits': 4, 'group_size': 128}
    return values


class _Mapper:
    def _map_name(self, name):
        return 'model.' + module.hf_to_original_name(name)

    def apply_list(self, names):
        return [self._map_name(name) for name in names]


def test_mixed_gptq_rules_cover_fused_attention_and_separate_shared_experts():
    schemes, ignored = module.map_gptq_module_schemes(_mixed_schemes(), [], _Mapper(), {
        'fused_wqa_wkv': ['wq_a', 'wkv'], 'fused_wkv_wgate': ['wkv', 'wgate'],
        'gate_up_proj': ['w1', 'w3'],
    })
    assert schemes['model.layers.0.attn.fused_wqa_wkv']['bits'] == 8
    assert schemes['model.layers.0.ffn.shared_experts.gate_up_proj']['bits'] == 8
    assert schemes['model.layers.0.ffn.shared_experts.w2']['bits'] == 8
    assert schemes['model.layers.0.ffn.experts']['bits'] == 4
    assert not ignored


def test_mixed_gptq_mapper_rejects_inconsistent_fused_precision():
    schemes = _mixed_schemes()
    schemes['model.layers.0.self_attn.kv_proj'] = {'bits': 4, 'group_size': 128}
    with pytest.raises(ValueError, match='Different GPTQ schemes within fused unit'):
        module.map_gptq_module_schemes(schemes, [], _Mapper(), {'fused_wqa_wkv': ['wq_a', 'wkv']})


def test_mixed_gptq_mapper_rejects_partially_selected_fused_projection():
    schemes = _mixed_schemes()
    name = 'model.layers.0.self_attn.kv_proj'
    del schemes[name]
    with pytest.raises(ValueError, match='Partially quantized'):
        module.map_gptq_module_schemes(schemes, [name], _Mapper(), {'fused_wqa_wkv': ['wq_a', 'wkv']})


def test_actual_auto_gptq_config_applies_hf_and_fused_bit_overrides():
    pytest.importorskip('vllm')
    import re
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
    from vllm.model_executor.layers.quantization.utils.gptq_utils import get_dynamic_override, override_config
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4ForCausalLM
    assert module.install_deepseek_v4_hf_checkpoint()
    mapper = DeepseekV4ForCausalLM.hf_to_vllm_mapper
    schemes = _mixed_schemes()
    original = {'bits': 4, 'group_size': 128, 'desc_act': False, 'sym': True,
                'modules_in_block_to_quantize': list(schemes),
                'flagos_module_quantization': schemes,
                'dynamic': {f'+:^{re.escape(name)}$': value for name, value in schemes.items() if value['bits'] == 8}}
    config = AutoGPTQConfig.from_config(original)
    config.apply_vllm_mapper(mapper)
    for prefix in ['model.layers.0.attn.fused_wqa_wkv', 'model.layers.0.ffn.shared_experts.gate_up_proj', 'model.layers.0.ffn.shared_experts.down_proj']:
        assert get_dynamic_override(config, prefix, 'bits', 4) == 8
    assert get_dynamic_override(config, 'model.layers.0.ffn.experts', 'bits', 4) == 4
    override_config(config, 'model.layers.0.attn.fused_wqa_wkv')
    assert config.weight_bits == 8 and config.pack_factor == 4
