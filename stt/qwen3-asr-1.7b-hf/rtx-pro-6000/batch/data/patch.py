"""Edit the installed vLLM v0.29.0 in place at container start, before ``vllm serve``.

The ``stt/vibevoice-asr/latency`` pattern: idempotent (marker check), and every anchor is
asserted so a vLLM upgrade fails the deploy at startup instead of serving unpatched.

Refuses to run on any other vLLM version. Three edits, each a measured bottleneck of the transcriptions route on fresh audio:

  multimodal/media/audio.py            ``load_audio`` decodes through ``b10_decode.py``
                                       (torchcodec in one native call, long files in
                                       parallel ranges) instead of the PyAV per-frame loop
  model_executor/models/qwen3_asr.py   audio-encoder forward from ``b10_encoder.py``: the
                                       same kernels without the five GPU->CPU readbacks
                                       per batch that stalled the engine thread
  v1/attention/backends/flashinfer.py  metadata build from the CPU sequence-length upper
                                       bound instead of a blocking GPU copy per prefill step

The decoder edit cuts the CPU decode stage of long files (a 1 hr m4a: 2.8 s -> 0.4 s), which
is the bottleneck at low concurrency. The two no-sync edits remove the engine-thread stalls
that bound high concurrency: GPU busy ~86% -> ~97% at 1 hr c16, WER unchanged.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import shutil
import sys

MARKER = "[b10 patch]"
HERE = os.path.dirname(os.path.abspath(__file__))


SUPPORTED_VLLM_VERSION = "0.29.0"
# The file edited last; its marker means every edit above it landed.
COMPLETION_FILE = "v1/attention/backends/flashinfer.py"
FIRST_FILE = "multimodal/media/audio.py"


def _vllm_root() -> str:
    # The image reports a local build tag ("0.29.0+cu129"); the public version is what the patch targets.
    installed = importlib.metadata.version("vllm").split("+", 1)[0]
    assert installed == SUPPORTED_VLLM_VERSION, (
        f"patch.py was verified against vllm {SUPPORTED_VLLM_VERSION}, found {installed}: re-verify before bumping"
    )
    spec = importlib.util.find_spec("vllm")
    assert spec is not None and spec.submodule_search_locations, "vllm is not installed"
    return list(spec.submodule_search_locations)[0]


def _has_marker(root: str, rel_path: str) -> bool:
    with open(os.path.join(root, rel_path)) as f:
        return MARKER in f.read()


def _edit(root: str, rel_path: str, edits: list[tuple[str, str]]) -> None:
    path = os.path.join(root, rel_path)
    with open(path) as f:
        src = f.read()
    for anchor, replacement in edits:
        n = src.count(anchor)
        assert n == 1, f"{rel_path}: expected exactly one match for anchor, found {n}:\n{anchor}"
        src = src.replace(anchor, replacement)
    with open(path, "w") as f:
        f.write(src)
    print(f"[patch.py] {rel_path}: {len(edits)} edit(s)")


def _install_decoder(root: str) -> None:
    # Fail the deploy here, not on the first upload, if the image stops shipping torchcodec.
    import torchcodec  # noqa: F401
    from torchcodec.decoders import AudioDecoder  # noqa: F401

    print(f"[patch.py] torchcodec {torchcodec.__version__} importable")
    shutil.copyfile(os.path.join(HERE, "b10_decode.py"), os.path.join(root, "b10_decode.py"))
    # The original becomes _load_audio_vllm (kept importable); the name every caller
    # imports now dispatches to torchcodec.
    _edit(
        root,
        "multimodal/media/audio.py",
        [
            (
                "def load_audio(\n    path: BytesIO | Path | str,\n",
                f"def load_audio(  # {MARKER}\n"
                "    path: BytesIO | Path | str,\n"
                "    *,\n"
                "    sr: float | None = 22050,\n"
                "    mono: bool = True,\n"
                "    max_duration_s: float | None = None,\n"
                "    max_decode_bytes: int | None = None,\n"
                "):\n"
                "    if sr is None:  # chat-route loaders keep the source rate: original path, unchanged\n"
                "        return _load_audio_vllm(\n"
                "            path, sr=sr, mono=mono, max_duration_s=max_duration_s, max_decode_bytes=max_decode_bytes\n"
                "        )\n"
                "    from vllm.b10_decode import load_audio_b10\n\n"
                "    return load_audio_b10(\n"
                "        path,\n"
                "        sr=sr,\n"
                "        mono=mono,\n"
                "        max_duration_s=max_duration_s,\n"
                "        max_decode_bytes=max_decode_bytes,\n"
                "    )\n\n\n"
                "def _load_audio_vllm(\n    path: BytesIO | Path | str,\n",
            )
        ],
    )


def _install_encoder(root: str) -> None:
    # Appended after the class bodies so the decorated class definitions stay untouched;
    # the original forward/_process_audio_input stay importable as *_vllm.
    shutil.copyfile(os.path.join(HERE, "b10_encoder.py"), os.path.join(root, "b10_encoder.py"))
    asr_path = os.path.join(root, "model_executor/models/qwen3_asr.py")
    with open(asr_path) as f:
        assert "class Qwen3ASRForConditionalGeneration(" in f.read(), asr_path
    with open(asr_path, "a") as f:
        f.write(
            "\n\n# " + MARKER + "\n"
            "from vllm import b10_encoder as _b10_encoder  # noqa: E402\n\n"
            "_b10_encoder.install(Qwen3OmniMoeAudioEncoder, Qwen3ASRForConditionalGeneration)\n"
        )
    print("[patch.py] model_executor/models/qwen3_asr.py: encoder no-sync installed")


def _install_flashinfer_nosync(root: str) -> None:
    # FlashInfer's metadata build reads seq_lens through the deprecated CommonAttentionMetadata
    # .seq_lens_cpu accessor, which copies from the GPU when the runner did not pass a CPU copy.
    # The V2 runner (v1/worker/gpu) never does, so every prefill-admitting step blocks the engine
    # thread until the previous step's kernels finish (27-58 ms per step at 1 hr c16, GPU then
    # idles through the launch phase). Without speculative decoding the CPU upper bound the
    # runner does pass is exact (num_computed + num_scheduled), so use it.
    _edit(
        root,
        "v1/attention/backends/flashinfer.py",
        [
            (
                "        if needs_seq_lens_cpu:\n"
                "            with gpu_sync_allowed():\n"
                "                seq_lens_cpu = common_attn_metadata.seq_lens_cpu\n",
                "        if needs_seq_lens_cpu:  # " + MARKER + "\n"
                "            seq_lens_cpu = common_attn_metadata._seq_lens_cpu\n"
                "            if (\n"
                "                seq_lens_cpu is None\n"
                "                and self.vllm_config.speculative_config is None\n"
                "                and common_attn_metadata.seq_lens_cpu_upper_bound is not None\n"
                "            ):\n"
                "                seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound.clone()\n"
                "            if seq_lens_cpu is None:\n"
                "                with gpu_sync_allowed():\n"
                "                    seq_lens_cpu = common_attn_metadata.seq_lens_cpu\n",
            )
        ],
    )


def main() -> int:
    root = _vllm_root()
    if _has_marker(root, COMPLETION_FILE):
        print(f"[patch.py] {root} already patched, skipping")
        return 0
    if _has_marker(root, FIRST_FILE):
        raise RuntimeError(f"{root}: partially patched (an earlier run failed mid-way); rebuild the container")
    _install_decoder(root)
    _install_encoder(root)
    _install_flashinfer_nosync(root)
    print(f"[patch.py] done: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
