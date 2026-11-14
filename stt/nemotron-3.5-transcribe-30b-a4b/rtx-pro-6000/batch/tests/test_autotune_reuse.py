"""autotune_reuse.py (data/start.sh's NEMOTRON_AUTOTUNE_REUSE): which ops a restored FlashInfer autotune file lets the start skip.

Standard library only. In production autotune_hook/sitecustomize.py calls decide() inside vLLM
with the path vLLM resolves for its config hash; here that path is built by hand:

    python -m pytest stt/nemotron-3.5-transcribe-30b-a4b/rtx-pro-6000/batch/tests/test_autotune_reuse.py

Keys are built the way FlashInfer 0.7.0.post1 builds them: ProfilingCacheKey.file_key is
``str((custom_op, runner_class_name, nearest_profile, extras))`` (autotuner.py:1184-1206).
checkpointing_ssu's extras carry torch dtypes (checkpointing_ssu.py:409-437, 615-631), whose
str() is ``torch.bfloat16`` and so on: not a Python literal. exec-7 (wp8541z, 2026-11-12) logged
``has no checkpointing_ssu entry (ops: {})`` for a 51-entry H100 file because of that; the RTX
presets vendor the same parser.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "data" / "autotune_reuse.py"
_spec = importlib.util.spec_from_file_location("autotune_reuse", _PATH)
autotune_reuse = importlib.util.module_from_spec(_spec)
sys.modules["autotune_reuse"] = autotune_reuse
_spec.loader.exec_module(autotune_reuse)

FI_VERSION = "0.7.0.post1"


class _Dtype:
    """Stands in for torch.dtype: str()/repr() is the same text, e.g. ``torch.bfloat16``."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"torch.{self.name}"


F32, BF16, I32 = _Dtype("float32"), _Dtype("bfloat16"), _Dtype("int32")


def file_key(custom_op, runner_class_name, nearest_profile, extras) -> str:
    # ProfilingCacheKey.file_key: str(self.key_fields), key_fields drops runner_hash.
    return str((custom_op, runner_class_name, nearest_profile, extras))


# checkpointing_ssu_s_f32_i_bf16_dt_bf16_w_f32_a_f32_si_i32_d_64_ds_128_np_9_mw_16_hpg_8_ng_8
SSU_EXTRAS = (
    (F32, BF16, BF16, F32, F32, I32, None, 64, 128, 9, 16, 8, 8, 10, True),  # module_base_args
    True,  # dt_softplus
    -1,  # pad_slot_id
    None,  # requested_d_split
    (True, False, True),  # optional_tensor_presence
    (((0, 0, 0, 1), (0, 0, 0, 2)), ((1, 2), (1, 2, 4), (8, 4, 2, 1), (1, 2))),
    ("cuda", 12, 0, 188),  # _device_tuning_signature on RTX PRO 6000 (sm120, 188 SMs)
)
SSU_KEY = file_key(
    "checkpointing_ssu",
    "CheckpointingSSURunner",
    ((256, 64, 64, 128), (8, 64, 64), (8, 64, 64)),
    SSU_EXTRAS,
)
MOE_KEYS = [
    file_key(op, "MoERunner", ((8, 3072), (8, 128)), ("trtllm", "FP8", "FP8", "BF16", 8, 128, 128, 0, 1856))
    for op in ("trtllm::fused_moe::gemm1", "trtllm::fused_moe::gemm2")
]


def test_ssu_key_is_not_a_python_literal():
    import ast

    assert "torch.float32" in SSU_KEY
    with pytest.raises(ValueError):
        ast.literal_eval(SSU_KEY)


@pytest.mark.parametrize(
    ("key", "op"),
    [
        (SSU_KEY, "checkpointing_ssu"),
        (MOE_KEYS[0], "trtllm::fused_moe::gemm1"),
        (file_key("it's", "R", (), ()), "it's"),  # str() switches to double quotes
        ("_generation", None),
        ("not a key", None),
    ],
)
def test_op_of(key, op):
    assert autotune_reuse.op_of(key) == op


HASH = "e9762d58" * 8  # this config's sha256(VllmConfig.compute_hash(include_version=False))
OTHER_HASH = "1f0c3a77" * 8  # another config's
# torch.cuda.get_device_name(); FlashInfer records the same call as _metadata.gpu (autotuner.py:358).
GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"


def _path(tmp_path: Path, config_hash: str = HASH) -> Path:
    return tmp_path / FI_VERSION / "120f" / config_hash / autotune_reuse.FILE_NAME


def _write(tmp_path: Path, entries: dict, metadata: dict | None = None, config_hash: str = HASH) -> Path:
    path = _path(tmp_path, config_hash)
    path.parent.mkdir(parents=True)
    body = {"_metadata": metadata or {"flashinfer_version": FI_VERSION, "gpu": GPU}}
    body["_records"] = {}
    body.update(entries)
    body["_generation"] = "0" * 64
    path.write_text(json.dumps(body, indent=2))
    return path


def _decide(tmp_path: Path, gpu: str | None = GPU) -> list[str]:
    return autotune_reuse.decide(str(_path(tmp_path)), FI_VERSION, gpu)


GOOD = {SSU_KEY: ["CheckpointingSSURunner", [0, 0, 0, 1]], MOE_KEYS[0]: ["MoERunner", 3], MOE_KEYS[1]: ["MoERunner", 7]}


def test_decide_matching_hash_reuses(tmp_path, capsys):
    _write(tmp_path, GOOD)
    assert _decide(tmp_path) == ["checkpointing_ssu", "trtllm::fused_moe::gemm1", "trtllm::fused_moe::gemm2"]
    assert "reusing" in capsys.readouterr().err


def test_decide_stale_file_from_other_config_tunes(tmp_path, capsys):
    _write(tmp_path, GOOD, config_hash=OTHER_HASH)
    _path(tmp_path).parent.mkdir(parents=True)  # vLLM's resolve() creates this config's dir
    assert _decide(tmp_path) == []
    assert "for another vLLM config" in capsys.readouterr().err


def test_decide_multiple_files_tunes(tmp_path, capsys):
    _write(tmp_path, GOOD)
    _write(tmp_path, GOOD, config_hash=OTHER_HASH)
    assert _decide(tmp_path) == []
    assert "2 autotune files" in capsys.readouterr().err


def test_decide_without_ssu_entry_tunes(tmp_path, capsys):
    _write(tmp_path, {MOE_KEYS[0]: ["MoERunner", 3]})
    assert _decide(tmp_path) == []
    assert "has no checkpointing_ssu entry" in capsys.readouterr().err


def test_decide_version_mismatch_tunes(tmp_path):
    _write(tmp_path, {SSU_KEY: ["CheckpointingSSURunner", 0]}, {"flashinfer_version": "0.6.0"})
    assert _decide(tmp_path) == []


def test_decide_gpu_mismatch_tunes(tmp_path):
    _write(tmp_path, GOOD)
    assert _decide(tmp_path, gpu="NVIDIA H100 80GB HBM3") == []


def test_decide_cold_start(tmp_path, capsys):
    assert _decide(tmp_path) == []
    assert "cold start" in capsys.readouterr().err


_FAKES = {
    # Stand-ins for the vLLM 0.32 / FlashInfer / torch names the hook touches.
    "vllm/__init__.py": "",
    "vllm/model_executor/__init__.py": "",
    "vllm/model_executor/warmup/__init__.py": "",
    "vllm/model_executor/warmup/kernel_warmup.py": (
        "def _flashinfer_autotune_skip_ops(runner):\n    return None\n"
    ),
    "vllm/model_executor/warmup/flashinfer_autotune_cache.py": (
        "import os\nfrom pathlib import Path\n"
        "def resolve_flashinfer_autotune_file(runner):\n"
        "    return Path(os.environ['FAKE_CACHE_DIR']) / 'autotune_configs.json'\n"
    ),
    "vllm/distributed/__init__.py": "",
    "vllm/distributed/parallel_state.py": (
        "class _G:\n    rank_in_group = 0\ndef get_world_group():\n    return _G()\n"
    ),
    "flashinfer/__init__.py": f"__version__ = {FI_VERSION!r}\n",
    "torch/__init__.py": "class cuda:\n    @staticmethod\n    def get_device_name():\n        return 'NVIDIA RTX PRO 6000 Blackwell Server Edition'\n",
}

_HOOK_DIR = _PATH.parent / "autotune_hook"
_PROBE = (
    "import types, vllm.model_executor.warmup.kernel_warmup as kw\n"
    "r = types.SimpleNamespace(vllm_config=types.SimpleNamespace("
    "parallel_config=types.SimpleNamespace(data_parallel_rank=0)))\n"
    "print(sorted(kw._flashinfer_autotune_skip_ops(r) or []))\n"
)


@pytest.mark.parametrize(("config_hash", "expected"), [(HASH, 3), (OTHER_HASH, 0)])
def test_hook_uses_vllm_resolved_path(tmp_path, config_hash, expected):
    import os
    import subprocess

    fakes = tmp_path / "fakes"
    for rel, text in _FAKES.items():
        (fakes / rel).parent.mkdir(parents=True, exist_ok=True)
        (fakes / rel).write_text(text)
    cache = tmp_path / "cache"
    _write(cache, GOOD, config_hash=config_hash)
    env = dict(os.environ, PYTHONPATH=f"{_HOOK_DIR}:{fakes}", NEMOTRON_AUTOTUNE_REUSE_HOOK="1",
               FAKE_CACHE_DIR=str(_path(cache).parent))
    out = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True,
                         cwd=tmp_path, check=True)
    assert len(eval(out.stdout)) == expected, out.stderr
