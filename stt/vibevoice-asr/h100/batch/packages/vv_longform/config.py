"""Environment knobs for the VibeVoice long-form proxy.

Every knob is an environment variable so /batch-bench can sweep it with
`truss_overrides.environment_variables`. Defaults are the shipped values (the
same as config.yaml and the data/start.sh fallbacks): 64k context, 300 s chunks
with speaker linking at 0.30, the upstream prompt recipe, the 12/s + 256 loop
guard and parallel audio prep. So a trimmed or misspelled env block still serves
the tested configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return default if raw == "" else int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw == "":
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # Where the proxy listens and where vLLM listens.
    listen_host: str = "0.0.0.0"
    listen_port: int = 8000
    upstream: str = "http://127.0.0.1:8001"
    served_model: str = "vibevoice"
    # vLLM's context window; used only to clamp max_tokens on routes the proxy
    # builds itself (transcriptions, chunks), so they never exceed the window.
    max_model_len: int = 65536
    # Fetch http(s) audio and decode non-WAV audio (m4a etc.) with ffmpeg into
    # 24 kHz mono WAV before vLLM sees it. The VibeVoice plugin decodes through
    # an ffmpeg stdin pipe, which cannot seek to an AAC moov atom at the end.
    decode_remote_audio: bool = True
    # Lever 1: server-side chunking. 0 = single pass.
    chunk_s: float = 300.0
    chunk_trigger_s: float | None = None  # default: chunk_s * 1.1
    cut_search_s: float = 20.0  # look this far either side of a boundary for the quietest frame
    max_inflight_chunks: int = 32  # bound on chunk sub-requests in flight across all requests
    # Audio preparation for chunked requests (README "Audio preparation"). Blocking prep runs on
    # a pool of prep_threads threads, off the event loop. prep_parallel decodes long audio
    # as prep_threads integer-second ffmpeg segments in parallel; a seam whose overlap differs from
    # the next segment's head falls back to the sequential decode (prep_seam_fallback=False keeps the
    # parallel samples instead). Sample-identical except for PNS-coded AAC, whose noise bands get a
    # different noise realization the seam check can miss. Single-pass requests never take this path.
    prep_threads: int = 4
    prep_parallel: bool = True
    prep_seam_fallback: bool = True
    prep_min_seg_s: float = 120.0
    # Cross-chunk speaker linking (only when a request is chunked). "embed" = WeSpeaker
    # ResNet34-LM embeddings + constrained agglomerative clustering (linking.py); "none" =
    # chunk-local labels c<i>s<k> (the bdb0271 stub).
    speaker_link: str = "embed"
    link_threshold: float = 0.30  # cosine similarity at which clusters stop merging
    link_max_s: float = 30.0  # seconds of a speaker's own speech embedded per chunk
    link_min_s: float = 2.0  # below this, an item joins its nearest cluster instead of seeding one
    link_model_path: str = "/models/wespeaker-resnet34-lm"
    link_device: str = "auto"  # auto | cuda | cpu
    link_gpu_mem_fraction: float = 0.02  # torch allocator cap in the proxy process (~1.6 GB on H100)
    link_cpu_threads: int = 2
    # Lever 2: max_tokens loop guard, max_tokens_per_s * s + max_tokens_base. It only lowers a
    # client's max_tokens on single-pass chat requests; chunks of a chunked request each get it.
    # None (VV_MAX_TOKENS_PER_S=0) = keep the client's max_tokens on chat completions and 20 * s +
    # 200 on routes the proxy builds itself.
    max_tokens_per_s: float | None = 12.0
    max_tokens_base: int = 256
    max_tokens_cap: int = 65536
    # Lever 3: canonical prompt (system prompt + "This is a N seconds audio ...", hotwords kept).
    canonical_prompt: bool = True
    # Logging.
    log_timing: bool = True

    @property
    def link_embed(self) -> bool:
        """Whether the proxy loads the speaker-embedding model (chunking on and VV_LINK=embed)."""
        return self.chunk_s > 0 and self.speaker_link == "embed"

    @property
    def effective_chunk_trigger_s(self) -> float:
        if self.chunk_trigger_s is not None:
            return self.chunk_trigger_s
        return self.chunk_s * 1.1


def load() -> Settings:
    link = (os.environ.get("VV_LINK", "").strip() or "embed").lower()
    if link not in ("embed", "none"):
        raise ValueError(f"VV_LINK must be embed or none, got {link!r}")
    return Settings(
        upstream=os.environ.get("VV_UPSTREAM", "http://127.0.0.1:8001").rstrip("/"),
        served_model=os.environ.get("VV_SERVED_MODEL", "vibevoice"),
        max_model_len=_env_int("VV_MAX_MODEL_LEN", 65536),
        decode_remote_audio=_env_bool("VV_DECODE_REMOTE_AUDIO", True),
        chunk_s=_env_float("VV_CHUNK_S", 300.0) or 0.0,
        chunk_trigger_s=_env_float("VV_CHUNK_TRIGGER_S", None),
        cut_search_s=_env_float("VV_CUT_SEARCH_S", 20.0) or 0.0,
        max_inflight_chunks=_env_int("VV_MAX_INFLIGHT_CHUNKS", 32),
        prep_threads=max(1, _env_int("VV_PREP_THREADS", 4)),
        prep_parallel=_env_bool("VV_PREP_PARALLEL", True),
        prep_seam_fallback=_env_bool("VV_PREP_SEAM_FALLBACK", True),
        prep_min_seg_s=_env_float("VV_PREP_MIN_SEG_S", 120.0),
        speaker_link=link,
        link_threshold=_env_float("VV_LINK_THRESHOLD", 0.30),
        link_max_s=_env_float("VV_LINK_MAX_S", 30.0),
        link_min_s=_env_float("VV_LINK_MIN_S", 2.0),
        link_model_path=os.environ.get("VV_LINK_MODEL_PATH", "/models/wespeaker-resnet34-lm"),
        link_device=os.environ.get("VV_LINK_DEVICE", "auto").strip() or "auto",
        link_gpu_mem_fraction=_env_float("VV_LINK_GPU_MEM_FRACTION", 0.02),
        link_cpu_threads=_env_int("VV_LINK_CPU_THREADS", 2),
        max_tokens_per_s=_env_float("VV_MAX_TOKENS_PER_S", 12.0) or None,  # 0 = guard off
        max_tokens_base=_env_int("VV_MAX_TOKENS_BASE", 256),
        max_tokens_cap=_env_int("VV_MAX_TOKENS_CAP", 65536),
        canonical_prompt=_env_bool("VV_CANONICAL_PROMPT", True),
        log_timing=_env_bool("VV_LOG_TIMING", True),
    )
