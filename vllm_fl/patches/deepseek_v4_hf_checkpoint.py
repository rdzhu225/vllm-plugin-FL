"""Accept Transformers' DeepSeek V4 names in calibrated GPTQ/AWQ exports."""

import re


def map_gptq_module_schemes(schemes, unquantized_names, mapper, packed_mapping):
    """Translate exact module schemes and close native fused execution units."""
    mapped = {}
    for source, settings in schemes.items():
        name = mapper._map_name(source)
        if name is None:
            raise ValueError(f"Quantized GPTQ module was dropped by the weight mapper: {source}")
        if name in mapped and mapped[name] != settings:
            raise ValueError(f"Conflicting GPTQ schemes map to {name}")
        mapped[name] = dict(settings)
    ignored = set(mapper.apply_list(unquantized_names))
    if ignored & mapped.keys():
        raise ValueError("A GPTQ module is both quantized and excluded after mapping")
    all_names = set(mapped) | ignored
    parents = {name.rsplit('.', 1)[0] for name in all_names if '.' in name}
    units = {}
    for parent in parents:
        for fused, shards in packed_mapping.items():
            members = {f"{parent}.{shard}" for shard in shards}
            if members <= all_names:
                units[f"{parent}.{fused}"] = members
    # The routed expert bank has one runtime quantization method. Shared
    # experts remain separate Linear methods and may use another bit width.
    for name in all_names:
        match = re.match(r"^(.*\.experts)\.\d+\.(?:w1|w2|w3|gate_proj|up_proj|down_proj)$", name)
        if match:
            units.setdefault(match.group(1), set()).add(name)
    for target, members in units.items():
        chosen = members & mapped.keys()
        if chosen and chosen != members:
            raise ValueError(f"Partially quantized GPTQ fused unit: {target}")
        if not chosen:
            ignored.add(target)
            continue
        values = [mapped[name] for name in chosen]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"Different GPTQ schemes within fused unit: {target}")
        mapped[target] = dict(values[0])
    return mapped, sorted(ignored)


def _install_gptq_module_mapper(packed_mapping):
    from functools import wraps
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig

    original = AutoGPTQConfig.apply_vllm_mapper
    if getattr(original, '_fl_module_schemes', False):
        return

    @wraps(original)
    def apply_mapper(self, mapper):
        original(self, mapper)
        schemes = self.full_config.get('flagos_module_quantization')
        if not schemes or not getattr(mapper, '_fl_deepseek_v4_mapper', False):
            return
        # Compressor writes canonical exact-match negatives. Decode those
        # names, rather than running a tensor-name mapper on regex syntax.
        ignored = []
        for pattern in self.dynamic:
            if not pattern.startswith('-:'):
                continue
            regex = pattern[2:]
            name = re.sub(r'\\(.)', r'\1', regex[1:-1])
            if regex != '^' + re.escape(name) + '$':
                raise ValueError('FlagOS module-scheme export requires exact GPTQ exclusions')
            ignored.append(name)
        resolved, exclusions = map_gptq_module_schemes(schemes, ignored, mapper, packed_mapping)
        baseline = {'bits': self.weight_bits, 'group_size': self.group_size}
        self.dynamic = {f'-:^{re.escape(name)}$': {} for name in exclusions}
        self.dynamic.update({f'+:^{re.escape(name)}$': value for name, value in resolved.items()
                             if value != baseline})
        self.full_config = {**self.full_config, 'dynamic': self.dynamic}

    apply_mapper._fl_module_schemes = True
    AutoGPTQConfig.apply_vllm_mapper = apply_mapper


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
        _fl_deepseek_v4_mapper = True

        def __init__(self, original):
            super().__init__()
            self.original = original

        def _map_name(self, key):
            return self.original._map_name(hf_to_original_name(key))

        def __or__(self, other):
            # AutoWeightsLoader merges the FP8 KV-cache scale mapper here.
            # The base implementation would return a plain WeightsMapper and
            # discard both our HF conversion and the wrapped native mappings.
            return HFCheckpointMapper(self.original | other)

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
    _install_gptq_module_mapper(cls.packed_modules_mapping)
    return True
