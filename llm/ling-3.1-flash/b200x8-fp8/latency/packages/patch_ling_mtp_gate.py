"""Version-guarded vLLM 0.30.0 workaround for Ling 3.1's FP8 MTP gate.

Only draft layer 72's 64x4096 gate is dequantized. Target weights are untouched.
Remove this shim when upstream supports the narrow block-FP8 draft gate.
"""

import hashlib
import importlib.metadata
import inspect
from pathlib import Path


def prepare_mtp_gate_quant_config(quant_config, config):
    """Exclude only the known narrow MTP gate from block-FP8 allocation."""
    if quant_config is None or not getattr(quant_config, "weight_block_size", None):
        return []
    if (
        tuple(quant_config.weight_block_size) != (128, 128)
        or config.num_hidden_layers != 72
        or config.num_nextn_predict_layers != 1
        or config.num_attention_heads != 64
        or config.hidden_size != 4096
        or config.gated_attention_proj_granularity_type != "head_wise"
    ):
        raise RuntimeError("Ling MTP gate shim only supports the pinned Ling 3.1 config")
    prefixes = ["model.layers.72.self_attn.g_proj"]
    # vLLM's HF-to-runtime mapper has already run; these are runtime names.
    quant_config.ignored_layers = list(quant_config.ignored_layers)
    for prefix in prefixes:
        if prefix not in quant_config.ignored_layers:
            quant_config.ignored_layers.append(prefix)
    return prefixes


def dequantize_mtp_gate_weights(weights, gate_prefixes):
    """Buffer at most one tiny gate/scale pair; preserve all other tensors."""
    import torch

    watched = {}
    for prefix in gate_prefixes:
        watched[prefix + ".weight"] = (prefix, "weight")
        watched[prefix + ".weight_scale_inv"] = (prefix, "scale")
    pending = {}
    completed = set()
    for name, tensor in weights:
        if name not in watched:
            yield name, tensor
            continue
        prefix, kind = watched[name]
        if prefix in completed or kind in pending.get(prefix, {}):
            raise RuntimeError(f"Duplicate MTP gate tensor: {name}")
        pair = pending.setdefault(prefix, {})
        pair[kind] = tensor
        if len(pair) != 2:
            continue
        weight, scale = pair["weight"], pair["scale"]
        if tuple(weight.shape) != (64, 4096) or tuple(scale.shape) != (1, 32):
            raise RuntimeError("Unexpected Ling MTP gate/scale shape")
        if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32:
            raise RuntimeError("Unexpected Ling MTP gate/scale dtype")
        # weight_scale_inv is the stored dequantization multiplier, not 1/scale.
        # 128x128 blocks: one partial row block and 32 column blocks.
        expanded = scale.to(device=weight.device).repeat_interleave(128, dim=1)
        restored = (weight.float() * expanded).to(torch.bfloat16)
        yield prefix + ".weight", restored
        del pending[prefix]
        completed.add(prefix)
    if pending:
        raise RuntimeError("Incomplete MTP gate/scale pair in checkpoint stream")
    if gate_prefixes and completed != set(gate_prefixes):
        raise RuntimeError("Pinned checkpoint did not supply the expected MTP gate")


def patch_file(path):
    """Fail closed on source drift; never import vLLM or initialize CUDA."""
    original = path.read_text()
    marker = "# LING31_MTP_GATE_BF16_SHIM"
    if marker in original:
        return False
    expected = "898c25789e62ce58ebdf62ddc57309498cdf3d482142669ea00000edc7cdb888"
    if hashlib.sha256(original.encode()).hexdigest() != expected:
        raise RuntimeError("Bailing MTP source differs from vLLM v0.30.0; review shim")
    ctor = """        _configure_ling_fp8_quant_config(self.quant_config, self.config)
        self.model = BailingMoeV3MultiTokenPredictor("""
    ctor_new = """        _configure_ling_fp8_quant_config(self.quant_config, self.config)
        self._bf16_mtp_gate_prefixes = prepare_mtp_gate_quant_config(
            self.quant_config, self.config
        )
        self.model = BailingMoeV3MultiTokenPredictor("""
    loader = """        weights = _maybe_remap_ling_mxfp4_weight_names(weights, self.quant_config)
        stacked_params_mapping = ["""
    loader_new = """        weights = _maybe_remap_ling_mxfp4_weight_names(weights, self.quant_config)
        weights = dequantize_mtp_gate_weights(weights, self._bf16_mtp_gate_prefixes)
        stacked_params_mapping = ["""
    if original.count(ctor) != 1 or original.count(loader) != 1:
        raise RuntimeError("Bailing MTP patch anchors are not unique")
    helpers = (
        "\n\n" + marker + "\n"
        + inspect.getsource(prepare_mtp_gate_quant_config)
        + "\n" + inspect.getsource(dequantize_mtp_gate_weights)
    )
    patched = original.replace(ctor, ctor_new).replace(loader, loader_new) + helpers
    compile(patched, str(path), "exec")
    path.write_text(patched)
    return True


if __name__ == "__main__":
    dist = importlib.metadata.distribution("vllm")
    if dist.version != "0.30.0":
        raise RuntimeError("Ling MTP gate shim is pinned to vLLM 0.30.0")
    path = Path(dist.locate_file("vllm/model_executor/models/bailing_moe_v3_mtp.py"))
    changed = patch_file(path)
    print("Ling 3.1 MTP narrow-gate BF16 shim installed" if changed else "Shim already installed")
