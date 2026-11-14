"""Run autotune_reuse.decide() inside vLLM's worker, on the file vLLM is about to load.

start_vllm.sh puts this directory on PYTHONPATH and sets NEMOTRON_AUTOTUNE_REUSE_HOOK=1 for
`vllm serve` when NEMOTRON_AUTOTUNE_REUSE=1. Python imports sitecustomize at start-up in every
process, including vLLM's spawned engine core. Nothing is imported eagerly: an import hook waits
for vllm.model_executor.warmup.kernel_warmup and then wraps its _flashinfer_autotune_skip_ops
(kernel_warmup.py:295-312), which flashinfer_autotune() calls at kernel_warmup.py:445 and looks up
as a module global. The wrapper resolves the cache file exactly as kernel_warmup.py:452-457 does,
so the config hash is vLLM's own, and adds the ops decide() returns. Any error: tune as usual.
Another sitecustomize further down sys.path (e.g. the venv's) still runs.

Under NEMOTRON_DIAG_WARMUP_PROFILE=1 it also registers a stack dump on SIGUSR2 in every process
(see _register_stack_dumps); start_vllm.sh then puts this directory on PYTHONPATH even with reuse off.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = "vllm.model_executor.warmup.kernel_warmup"


def _load_decider():
    path = os.path.join(os.path.dirname(_HERE), "autotune_reuse.py")
    spec = importlib.util.spec_from_file_location("_nemotron_autotune_reuse", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def expected_cache_file(runner) -> str:
    """The path flashinfer_autotune() loads for ``runner`` (kernel_warmup.py:452-457)."""
    from vllm.distributed.parallel_state import get_world_group
    from vllm.model_executor.warmup.flashinfer_autotune_cache import (
        resolve_flashinfer_autotune_file,
    )

    cache_path = resolve_flashinfer_autotune_file(runner)
    dp_rank = runner.vllm_config.parallel_config.data_parallel_rank
    rank = get_world_group().rank_in_group
    return str(cache_path.with_name(f"{cache_path.stem}_dp{dp_rank}_rank{rank}{cache_path.suffix}"))


def _patch(module) -> None:
    original = module._flashinfer_autotune_skip_ops

    def skip_ops(runner):
        base = original(runner)
        decider = None
        try:
            decider = _load_decider()
            import flashinfer
            import torch

            ops = decider.decide(
                expected_cache_file(runner), flashinfer.__version__, torch.cuda.get_device_name()
            )
        except Exception as e:  # never fail a start over reuse
            msg = f"reuse check failed ({type(e).__name__}: {e}); tuning as usual"
            (decider.log if decider else lambda m: print(f"autotune_reuse: {m}", file=sys.stderr))(msg)
            ops = []
        if not ops:
            return base
        return set(ops) | (base or set())

    module._flashinfer_autotune_skip_ops = skip_ops


class _Finder:
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def _chain() -> None:
    others = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", others)
    if spec is not None and spec.loader is not None:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)


def _register_stack_dumps() -> None:
    """NEMOTRON_DIAG_WARMUP_PROFILE=1 (diagnostic): dump every thread's Python stack to stderr on
    SIGUSR2, so nemotron_stt's warmup profiler can name what a slow warmup decode is doing.

    faulthandler's handler is C-level, so it answers while a thread is blocked in native code
    (a compile, a driver call). SIGUSR2's default action ends the process, so the profiler signals
    only processes that left a marker here: the file is named by pid and holds the process start
    time, which the profiler checks so a reused pid is never signalled.
    """
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR2, all_threads=True, chain=False)
    marker_dir = os.environ.get("NEMOTRON_DIAG_STACK_DIR", "/tmp/nemotron-diag-stacks")
    os.makedirs(marker_dir, exist_ok=True)
    with open("/proc/self/stat") as f:
        start_time = f.read().rsplit(")", 1)[1].split()[19]
    with open(os.path.join(marker_dir, str(os.getpid())), "w") as f:
        f.write(start_time)


if os.environ.get("NEMOTRON_DIAG_WARMUP_PROFILE") == "1":
    try:
        _register_stack_dumps()
    except Exception as e:  # a diagnostic never fails a start
        print(f"diag warmup profile: stack dumps unavailable ({type(e).__name__}: {e})", file=sys.stderr)

if os.environ.get("NEMOTRON_AUTOTUNE_REUSE_HOOK") == "1" and _TARGET not in sys.modules:
    sys.meta_path.insert(0, _Finder())
_chain()
