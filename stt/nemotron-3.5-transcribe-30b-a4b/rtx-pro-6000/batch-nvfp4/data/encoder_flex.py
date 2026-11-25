"""Force FlexAttention's main Triton kernel in the compiled audio encoder.

With ``encoder_compile`` on, inductor lowers the encoder's FlexAttention calls to its
flex_decoding kernel when a clip has fewer than 128 encoder tokens, and that kernel broke
short-clip transcripts (mu-bench WER ~70% on B200). Passing
``kernel_options={"BACKEND": "TRITON"}`` keeps every call on the main flex_attention kernel.

start.sh writes a ``.pth`` file into the venv when NEMOTRON_ENCODER_COMPILE=1, so every vLLM
process runs ``install()`` at startup. ``install()`` waits for NeMo Speech's
``encoder_graphs`` module to be imported, then wraps
``transformer_encoder_utils._get_flex_attention``. NeMo's ``MultiHeadAttention`` looks that
selector up at every call, so the wrapper covers eager blocks, compiled blocks (dynamo inlines
it) and the streaming path. Nothing else about the encoder changes.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import logging
import sys

TARGET = "nemo.collections.speechlm2.vllm.salm.encoder_graphs"
FLEX_UTILS = "nemo.collections.asr.modules.transformer_encoder_utils"
BACKEND = "TRITON"

log = logging.getLogger("encoder_flex")


def force_triton() -> None:
    """Make NeMo's encoder FlexAttention calls pass kernel_options={"BACKEND": "TRITON"}."""
    utils = importlib.import_module(FLEX_UTILS)
    if getattr(utils, "_encoder_flex_wrapped", False):
        return
    select = utils._get_flex_attention

    def _get_flex_attention(x):
        attn_fn = select(x)

        def flex(*args, kernel_options=None, **kwargs):
            return attn_fn(*args, kernel_options={**(kernel_options or {}), "BACKEND": BACKEND}, **kwargs)

        return flex

    utils._get_flex_attention = _get_flex_attention
    utils._encoder_flex_wrapped = True
    log.debug("encoder FlexAttention backend forced to %s", BACKEND)


class _Finder(importlib.abc.MetaPathFinder):
    """Run force_triton() right after TARGET executes."""

    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(mod):
            exec_module(mod)
            force_triton()

        spec.loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    """Called from the .pth line."""
    if TARGET in sys.modules:
        force_triton()
    elif not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
