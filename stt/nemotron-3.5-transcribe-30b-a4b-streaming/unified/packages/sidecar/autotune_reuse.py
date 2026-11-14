"""Decide which FlashInfer autotune ops a warm start can skip (NEMOTRON_AUTOTUNE_REUSE=1, the default).

Vendored from the H100 streaming preset (stt/nemotron-3.5-transcribe-30b-a4b-streaming/h100); this
image runs the same locked vLLM 0.32.0 / FlashInfer 0.7.0.post1 (NeMo Speech a837f4e uv.lock), so the
vLLM and FlashInfer lines cited below are this image's too. RTX PRO 6000: FlashInfer records
_metadata.gpu with torch.cuda.get_device_name() (flashinfer/autotuner/autotuner.py:358), the same
call the hook passes as ``gpu``, so a file tuned on another GPU type is refused by name.

Runs inside vLLM's worker process: autotune_hook/sitecustomize.py wraps vLLM 0.32's
_flashinfer_autotune_skip_ops (vllm/model_executor/warmup/kernel_warmup.py:295-312), which
flashinfer_autotune() calls at kernel_warmup.py:445, right before it resolves and loads the
autotune file (kernel_warmup.py:452-468). The wrapper computes the file path the same way
(resolve_flashinfer_autotune_file plus the _dp<rank>_rank<rank> suffix, kernel_warmup.py:452-457)
and passes it to decide(). So reuse is keyed on the exact file this config loads:
<VLLM_CACHE_ROOT>/flashinfer_autotune_cache/<flashinfer version>/<arch>/<config hash>/
autotune_configs_dp0_rank0.json, where the version/arch levels are FlashInfer's workspace dir
(flashinfer/jit/env.py:355-368) and <config hash> is
sha256(VllmConfig.compute_hash(include_version=False)) with mamba_block_size normalized
(vllm/model_executor/warmup/flashinfer_autotune_cache.py:20-35, 38-55; vllm/config/vllm.py:449).
Any change to the vLLM config therefore points at a different file.

Why: with a restored autotune file, vLLM 0.32 loads it before tuning (kernel_warmup.py:467-468),
so the MoE GEMMs hit it, but FlashInfer 0.7.0.post1 ignores file entries while tuning for ops
profiled under a non-default policy (flashinfer/autotuner/autotuner.py:1969-1972;
checkpointing_ssu profiles with cold L2), so checkpointing_ssu re-profiles on every start. A
skipped op returns the fallback tactic during the tuning pass only (autotuner.py:2183-2190, before
any cache lookup); outside that context the loaded file still serves its tuned tactic
(autotuner.py:1975-1993, "Config cache hit ... source=config file"). The file stays loaded
whatever is skipped: the load happens before and independent of skip_ops
(kernel_warmup.py:445-468), and since nothing new is tuned it is not rewritten
(kernel_warmup.py:503).

Reuse needs all of: the expected file exists and is the only autotune file in the cache (the BDN
startup cache is per deployment, so it holds one); it parses; its recorded FlashInfer version and
GPU match; it has a checkpointing_ssu entry. Otherwise nothing is skipped and the start tunes as
before, with the reason logged (no file: a cold start; a file under another config hash: a stale
file from a different config). Whether serving used the file is confirmed afterwards by vLLM's
"Config cache hit for checkpointing_ssu (source=config file)" line.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

PREFIX = "autotune_reuse:"
FILE_NAME = "autotune_configs_dp0_rank0.json"
REQUIRED_OP = "checkpointing_ssu"
# FlashInfer 0.7.0.post1 writes each entry under
# ProfilingCacheKey.file_key = str((custom_op, runner_class_name, nearest_profile, extras))
# (flashinfer/autotuner/autotuner.py:1184-1206). ``extras`` can hold non-literal reprs such as
# torch.bfloat16 (checkpointing_ssu's module args), so the key is not a Python literal; the op
# is the leading quoted string. str() of a str quotes with ' unless the text contains '.
_OP_RE = re.compile(r"""^\(\s*(?:'([^'\\]*)'|"([^"\\]*)")\s*,""")


def op_of(key: str) -> str | None:
    """The custom op of a FlashInfer autotune file key, or None if it is not one."""
    m = _OP_RE.match(key)
    if not m:
        return None
    return m.group(1) if m.group(1) is not None else m.group(2)


def log(msg: str) -> None:
    print(f"{PREFIX} {msg}", file=sys.stderr, flush=True)


def decide(expected: str, fi_version: str, gpu: str | None = None) -> list[str]:
    """Ops to skip given the autotune file vLLM will load for this config (``expected``)."""
    # Siblings: every hash dir next to this config's, i.e. the files other configs left behind.
    pattern = os.path.join(os.path.dirname(os.path.dirname(expected)), "*", os.path.basename(expected))
    files = sorted(glob.glob(pattern))
    if not files:
        log(f"no restored autotune file ({pattern}); cold start, tuning as usual")
        return []
    if expected not in files:
        log(f"restored autotune file(s) {files} are for another vLLM config; this config loads "
            f"{expected}; tuning as usual")
        return []
    if len(files) > 1:
        log(f"{len(files)} autotune files in the cache ({files}); expected one, tuning as usual")
        return []
    path = expected
    try:
        with open(path) as f:
            configs = json.load(f)
        metadata = configs.pop("_metadata", None) or {}
    except (OSError, ValueError, AttributeError) as e:
        log(f"unreadable autotune file {path} ({e}); tuning as usual")
        return []
    if metadata.get("flashinfer_version") not in (None, fi_version):
        log(f"{path} was tuned with FlashInfer {metadata.get('flashinfer_version')}, "
            f"installed {fi_version}; tuning as usual")
        return []
    saved_gpu = metadata.get("gpu")
    if gpu and saved_gpu and saved_gpu != "unknown" and saved_gpu != gpu:
        log(f"{path} was tuned on {saved_gpu}, this GPU is {gpu}; tuning as usual")
        return []
    counts: dict[str, int] = {}
    for key in configs:
        if key.startswith("_"):  # _records, _generation and other file bookkeeping
            continue
        op = op_of(key)
        if op:
            counts[op] = counts.get(op, 0) + 1
    if REQUIRED_OP not in counts:
        log(f"{path} has no {REQUIRED_OP} entry (ops: {counts}); tuning as usual")
        return []
    ops = sorted(counts)
    log(f"reusing {path}, the file this config loads ({sum(counts.values())} entries, "
        f"gpu={saved_gpu}, per op {counts}); skipping tuning for {ops}")
    return ops
