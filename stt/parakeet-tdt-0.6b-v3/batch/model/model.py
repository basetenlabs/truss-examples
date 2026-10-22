"""Custom Python Truss model for nvidia/parakeet-tdt-0.6b-v3 (STT).

FastConformer-TDT transducer served through NVIDIA NeMo — no vLLM/SGLang
implementation exists for this architecture, so this follows the registry's
custom `model/model.py` path (b10-bench protocol: `baseten_predict`).

Contract:
- in:  {"audio_url": "https://..."} or {"audio_b64": "..."}
       optional: {"timestamps": true}
- out: {"transcript": "...", "text": "...", ["timestamps": {"word": [...], "segment": [...]}]}
"""

import base64
import collections
import contextlib
import copy
import difflib
import gc
import io
import logging
import math
import os
import queue
import subprocess
import tempfile
import threading
import time

from prometheus_client import Counter, Gauge, Histogram

LOGGER = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/parakeet-tdt-0.6b-v3")
NEMO_CHECKPOINT = os.environ.get(
    "NEMO_CHECKPOINT", os.path.join(MODEL_DIR, "parakeet-tdt-0.6b-v3.nemo")
)
AUDIO_SAMPLE_RATE_HZ = 16_000
FFMPEG_ERROR_CONTEXT_CHARS = 2_000
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "1"))
BATCH_WINDOW_SECONDS = float(os.environ.get("BATCH_WINDOW_MS", "2")) / 1000.0
PREDICT_TIMEOUT_SECONDS = float(os.environ.get("PREDICT_TIMEOUT_SECONDS", "300"))


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float_list(name: str, default: str = "") -> tuple[float, ...]:
    value = os.environ.get(name, default).strip()
    if not value:
        return ()
    values = tuple(sorted({float(item.strip()) for item in value.split(",")}))
    if any(item <= 0 for item in values):
        raise ValueError(f"{name} values must be positive: {value}")
    return values


if MAX_BATCH_SIZE < 1:
    raise ValueError("MAX_BATCH_SIZE must be at least 1")
if BATCH_WINDOW_SECONDS < 0:
    raise ValueError("BATCH_WINDOW_MS must be non-negative")
if PREDICT_TIMEOUT_SECONDS <= 0:
    raise ValueError("PREDICT_TIMEOUT_SECONDS must be positive")

BATCH_BUCKET_SECONDS = _env_float_list("BATCH_BUCKET_SECONDS", "2,4,8")
DIRECT_FORWARD = _env_flag("PARAKEET_DIRECT_FORWARD")
# Run the timestamps path once at startup and fail if NeMo leaves the model in
# training mode (see _restore_eval_after_transcribe).
STARTUP_MODE_CHECK = _env_flag("STARTUP_MODE_CHECK", True)
_TRANSCRIBE_MODE_LOGGED = []
REQUIRE_FULL_CUDA_GRAPH = _env_flag("REQUIRE_FULL_CUDA_GRAPH")
WARMUP_AUDIO_SECONDS = _env_float_list("WARMUP_AUDIO_SECONDS")
WARMUP_BATCH_SIZE = int(os.environ.get("WARMUP_BATCH_SIZE", "1"))
FREEZE_GC_AFTER_WARMUP = _env_flag("FREEZE_GC_AFTER_WARMUP")

if WARMUP_BATCH_SIZE < 1:
    raise ValueError("WARMUP_BATCH_SIZE must be at least 1")

# cuDNN for every GPU pass (encoder convolutions and the LSTM prediction network),
# off by default since Model Scout round 2 (#865, run 39667523725: c64 throughput
# 49.9 req/s with cuDNN off vs 16.1 with it on). "false" sets
# torch.backends.cudnn.enabled=False once at load, before warmup and CUDA-graph
# capture, so every later pass on the single GPU worker thread runs without it.
# Mirrors stt/orukeet/batch's SHORT_PATH_CUDNN: on RTX PRO 6000 (sm_120) with torch
# 2.8 / cuDNN 9.10, cuDNN spent 0.4-0.7 s in the encoder the first time it saw each
# new input length, while PyTorch's own kernels were as fast or faster once warm.
ENCODER_CUDNN = _env_flag("ENCODER_CUDNN", False)

# Length bucketing, off by default (the coarse BATCH_BUCKET_SECONDS batcher above is
# unchanged). When on, the batcher holds up to BATCH_BUCKET_LOOKAHEAD pending clips
# (default 4 x MAX_BATCH_SIZE) and builds each batch from the oldest clip plus the
# clips closest to it in length, so the coarse buckets no longer gate batching and
# batches fill more often. Timestamp requests still batch separately.
BATCH_LENGTH_BUCKETING = _env_flag("BATCH_LENGTH_BUCKETING", False)
BATCH_BUCKET_LOOKAHEAD = int(
    os.environ.get("BATCH_BUCKET_LOOKAHEAD", str(4 * MAX_BATCH_SIZE))
)
# Padding cap for length bucketing, off at 0. A clip joins a length-bucketed batch
# only if the batch's padding efficiency (summed clip length / (clips x longest
# clip)) stays at or above this value; the oldest clip always runs.
BATCH_MIN_PADDING_EFFICIENCY = float(
    os.environ.get("BATCH_MIN_PADDING_EFFICIENCY", "0")
)
# Batch window early exit, off by default (every batcher waits out BATCH_WINDOW_MS
# for more clips as before). When on, the worker stops waiting as soon as the job
# queue is empty and no predict() call is still decoding audio it will enqueue:
# nothing already inside the pod can join the batch, so the rest of the window is
# idle GPU time. #865 round 6 timing logs (woo59pn) showed that wait as a ~5.5 ms
# gap before every 2-8 clip batch and a 5.1 ms mean "batching" phase at c=1 (a
# quarter of the 21 ms pod time). Requests still in flight upstream of predict()
# are not seen, so batches can be smaller than with the full window.
BATCH_WINDOW_EARLY_EXIT = _env_flag("BATCH_WINDOW_EARLY_EXIT", False)
if BATCH_BUCKET_LOOKAHEAD < 1:
    raise ValueError("BATCH_BUCKET_LOOKAHEAD must be at least 1")
if not 0 <= BATCH_MIN_PADDING_EFFICIENCY <= 1:
    raise ValueError("BATCH_MIN_PADDING_EFFICIENCY must be between 0 and 1")

# How the model runs in reduced precision. The default is `full` bfloat16 since Model
# Scout round 2 (#865, run 39667523725): same throughput and WER as fp16 autocast,
# lower encoder time (17.8 vs 22.6 ms at c1), and no per-call autocast casts.
# `autocast` (the round-1 behavior): weights stay in the checkpoint's float32
# and every GPU pass runs under torch.autocast(INFERENCE_DTYPE), which casts weights
# and activations per op on every call. float32 means no autocast.
# `full`: load() casts the encoder, the prediction network (decoder) and the joint to
# INFERENCE_DTYPE once, before warmup captures the TDT decoder's CUDA graph, and no
# pass uses autocast. The mel preprocessor still computes in float32 from the float32
# waveform (NeMo's AudioPreprocessor casts its input to float32); only its output is
# cast to INFERENCE_DTYPE, via its dtype sentinel buffer, so the filterbank and
# window keep float32 values.
INFERENCE_DTYPE = os.environ.get("INFERENCE_DTYPE", "bfloat16").strip().lower()
if INFERENCE_DTYPE not in {"float16", "bfloat16", "float32"}:
    raise ValueError("INFERENCE_DTYPE must be float16, bfloat16 or float32")
PRECISION_MODE = os.environ.get("PRECISION_MODE", "full").strip().lower()
if PRECISION_MODE not in {"autocast", "full"}:
    raise ValueError("PRECISION_MODE must be autocast or full")

# Encoder CUDA graphs, off by default so one deployment can A/B them. Ported from
# stt/orukeet/batch-short (#604). When on, load() captures the preprocessor + encoder
# (model.forward) as one CUDA graph per (batch size, padded length): batch sizes from
# ENCODER_GRAPH_BATCH_SIZES up to the first one >= MAX_BATCH_SIZE, lengths in
# ENCODER_GRAPH_STEP_SECONDS steps up to ENCODER_GRAPH_MAX_SECONDS. A direct-path batch
# is zero-padded to the next captured batch size and length step and replayed; eager
# batching already pads every clip to the batch's longest and NeMo masks the padding.
# Any batch outside the captured range runs eagerly as before -- there is no maximum
# input length. Capture happens after the PRECISION_MODE cast and the cuDNN setting,
# in the serving dtype and never under autocast, and before NeMo captures the TDT
# decoder's own full graph. A capture failure fails load(); there is no silent eager
# fallback. Timestamp requests (NeMo transcribe()) never use the encoder graphs.
ENCODER_CUDA_GRAPHS = _env_flag("ENCODER_CUDA_GRAPHS", False)
ENCODER_GRAPH_BATCH_SIZES = tuple(
    sorted(
        {
            int(item)
            for item in os.environ.get("ENCODER_GRAPH_BATCH_SIZES", "1,2,4,8,16").split(",")
            if item.strip()
        }
    )
)
ENCODER_GRAPH_MAX_SECONDS = float(os.environ.get("ENCODER_GRAPH_MAX_SECONDS", "30"))
ENCODER_GRAPH_STEP_SECONDS = float(os.environ.get("ENCODER_GRAPH_STEP_SECONDS", "1"))
# NeMo rebuilds the encoder's relative positional table (a new tensor) whenever an
# input is longer than any seen so far, and a captured graph would keep reading the
# old, freed table. With graphs on, load() pre-builds the table for this much audio
# (default 24 min, the model's full-attention limit; about 0.6 GB in bfloat16). An
# even longer input still runs (eagerly), but rebuilds the table, and every later
# batch then runs eagerly too: each replay checks the table has not moved.
ENCODER_POS_TABLE_SECONDS = float(os.environ.get("ENCODER_POS_TABLE_SECONDS", "1440"))
# Debug, off at 0: for the first GRAPH_PARITY_BATCHES graphed batches, also run the
# unpadded batch eagerly and log how many transcripts differ. Doubles that GPU work.
GRAPH_PARITY_BATCHES = int(os.environ.get("GRAPH_PARITY_BATCHES", "0"))
if not ENCODER_GRAPH_BATCH_SIZES or ENCODER_GRAPH_BATCH_SIZES[0] < 1:
    raise ValueError("ENCODER_GRAPH_BATCH_SIZES must be positive integers")
if ENCODER_GRAPH_MAX_SECONDS <= 0 or ENCODER_GRAPH_STEP_SECONDS <= 0:
    raise ValueError("ENCODER_GRAPH_MAX_SECONDS and ENCODER_GRAPH_STEP_SECONDS must be positive")
if ENCODER_POS_TABLE_SECONDS < 0:
    raise ValueError("ENCODER_POS_TABLE_SECONDS must be non-negative")
if GRAPH_PARITY_BATCHES < 0:
    raise ValueError("GRAPH_PARITY_BATCHES must be non-negative")
if ENCODER_CUDA_GRAPHS and not DIRECT_FORWARD:
    raise ValueError("ENCODER_CUDA_GRAPHS requires PARAKEET_DIRECT_FORWARD=true")
if ENCODER_CUDA_GRAPHS and PRECISION_MODE == "autocast" and INFERENCE_DTYPE != "float32":
    # Graphs are captured in the serving dtype only; autocast is not supported.
    raise ValueError(
        "ENCODER_CUDA_GRAPHS requires PRECISION_MODE=full (or INFERENCE_DTYPE=float32)"
    )

# Encoder/decoder overlap (experiment), off by default. When on, the batch worker
# replays batch N+1's encoder graph on one CUDA stream while a second thread decodes
# batch N on another; a CUDA event hands each encoded batch over through a queue of
# depth 1, and only one decode runs at a time (NeMo's decoding object owns a single
# set of decoder state tensors). Timestamps batches, decoding rebuilds, batches
# outside the captured encoder graphs and GRAPH_PARITY_BATCHES batches first wait
# for the pipeline to drain, then run serially as before. Requires
# ENCODER_CUDA_GRAPHS: warmup sizes NeMo's decoder graph for every graphed shape, so
# an overlapped batch never makes NeMo re-capture that graph mid-pipeline.
_PIPELINE_OVERLAP_VALUE = os.environ.get("PIPELINE_OVERLAP", "false").strip().lower()
if _PIPELINE_OVERLAP_VALUE not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
    raise ValueError(f"PIPELINE_OVERLAP must be true or false, got {_PIPELINE_OVERLAP_VALUE!r}")
PIPELINE_OVERLAP = _PIPELINE_OVERLAP_VALUE in {"1", "true", "yes", "on"}
if PIPELINE_OVERLAP and not ENCODER_CUDA_GRAPHS:
    raise ValueError("PIPELINE_OVERLAP requires ENCODER_CUDA_GRAPHS=true")

# At load, transcribe one public LibriSpeech clip in the serving dtype and fail the
# deploy if the encoder output is non-finite or the WER against the known reference
# is above STARTUP_CHECK_MAX_WER, so a broken precision or decode setup fails in
# seconds instead of passing a perf bench with wrong text. A clip that cannot be
# fetched also fails startup: the check never silently skips.
STARTUP_TRANSCRIPT_CHECK = _env_flag("STARTUP_TRANSCRIPT_CHECK", True)
STARTUP_CHECK_AUDIO_URL = os.environ.get(
    "STARTUP_CHECK_AUDIO_URL",
    "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav",
)
STARTUP_CHECK_REFERENCE = (
    "well i don't wish to see it any more observed phoebe turning away her eyes "
    "it is certainly very like the old portrait"
)
STARTUP_CHECK_MAX_WER = float(os.environ.get("STARTUP_CHECK_MAX_WER", "0.25"))

# Debug, off at 0: log batch size, padding and encoder/decoder milliseconds for the
# first TIMING_LOG_BATCHES direct-path batches served after load. Each logged batch
# synchronizes the GPU between encoder and decoder, so keep it off for perf runs.
# With PIPELINE_OVERLAP, overlapped batches are timed with CUDA events on each
# stream instead (no device sync) and also log each stage's start on a shared GPU
# timeline, so overlap between batches can be measured.
TIMING_LOG_BATCHES = int(os.environ.get("TIMING_LOG_BATCHES", "0"))
if TIMING_LOG_BATCHES < 0:
    raise ValueError("TIMING_LOG_BATCHES must be non-negative")
_TIMING_STATE = {"serving": False, "batches": 0}


class ParakeetMetrics:
    """Prometheus metrics using the standard b10 model identity labels."""

    def __init__(self):
        identity_label_names = ("model_id", "model_version_id")
        self._identity_labels = {
            "model_id": os.environ.get("BT_MODEL_ID", ""),
            "model_version_id": os.environ.get("BT_MODEL_DEPLOYMENT_ID", ""),
        }
        # Match power-of-two seconds buckets used by b10 runtime metrics, with
        # enough range for long-form audio outliers.
        latency_buckets_seconds = tuple(0.001 * (2**index) for index in range(17))
        request_latency_seconds = Histogram(
            "parakeet_request_latency_seconds",
            "Parakeet request critical-path latency in seconds by phase",
            (*identity_label_names, "phase"),
            buckets=latency_buckets_seconds,
        )
        self._request_latency_by_phase = {
            phase: request_latency_seconds.labels(**self._identity_labels, phase=phase)
            for phase in (
                "preprocessing",
                "queueing",
                "batching",
                "inference",
                "postprocessing",
                "total",
            )
        }
        self.queue_depth = Gauge(
            "parakeet_queue_depth",
            "Requests waiting to be assigned to a Parakeet inference batch",
            identity_label_names,
        ).labels(**self._identity_labels)
        self.batch_size = Histogram(
            "parakeet_batch_size",
            "Number of requests in each Parakeet inference batch",
            identity_label_names,
            buckets=(1, 2, 4, 8, 16),
        ).labels(**self._identity_labels)
        # Direct-path batches by how the encoder ran: "graph" (ENCODER_CUDA_GRAPHS
        # replay) or "eager". Timestamp requests (transcribe()) are not counted.
        encoder_batches = Counter(
            "parakeet_encoder_batches",
            "Direct-path Parakeet batches by encoder execution path",
            (*identity_label_names, "path"),
        )
        self.encoder_batches = {
            path: encoder_batches.labels(**self._identity_labels, path=path)
            for path in ("graph", "eager")
        }

    def observe_latency(self, phase: str, duration_seconds: float):
        self._request_latency_by_phase[phase].observe(duration_seconds)


# Prometheus uses a process-global registry, so construct each metric once.
PARAKEET_METRICS = ParakeetMetrics()


def _autocast():
    """The precision context for every GPU pass.

    Under PRECISION_MODE=full the weights were cast once at load, so this is a
    no-op; under autocast, float32 means no autocast.
    """
    if PRECISION_MODE == "full" or INFERENCE_DTYPE == "float32":
        return contextlib.nullcontext()
    import torch

    return torch.autocast(device_type="cuda", dtype=getattr(torch, INFERENCE_DTYPE))


def _normalized_words(text) -> list:
    if not isinstance(text, str):
        text = str(text)
    kept = "".join(c if c.isalnum() or c in " '" else " " for c in text.lower())
    return kept.split()


def _word_error_rate(reference: str, hypothesis) -> float:
    ref, hyp = _normalized_words(reference), _normalized_words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        current = [i]
        for j, hyp_word in enumerate(hyp, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1] / max(1, len(ref))


def _hypothesis_text(hypothesis):
    return hypothesis if isinstance(hypothesis, str) else hypothesis.text


def _modules_in_training_mode(model) -> list:
    return [name or "<root>" for name, module in model.named_modules() if module.training]


def _closest_length_batch(lengths, max_batch_size_for, min_padding_efficiency=0.0) -> list:
    """Choose one length-bucketed batch from pending clips (BATCH_LENGTH_BUCKETING).

    `lengths` are the pending clips' durations in arrival order. lengths[0], the
    oldest, is always chosen, so a clip waits at most len(lengths) batches once it
    is pending. The others are added closest in length to it first (ties in arrival
    order) while the batch stays within max_batch_size_for(its longest clip) and its
    padding efficiency, sum(lengths) / (clips x longest), stays at or above
    min_padding_efficiency (BATCH_MIN_PADDING_EFFICIENCY; 0 turns the check off).
    A clip that fails either check is skipped and farther clips are still tried.
    Returns the chosen indices in arrival order.
    """
    chosen = [0]
    longest = lengths[0]
    total = lengths[0]
    ranked = sorted(
        range(1, len(lengths)),
        key=lambda index: (abs(lengths[index] - lengths[0]), index),
    )
    for index in ranked:
        # The cap only shrinks as the longest clip grows, so a full batch stays full.
        if len(chosen) >= max_batch_size_for(longest):
            break
        candidate_longest = max(longest, lengths[index])
        if len(chosen) >= max_batch_size_for(candidate_longest):
            continue
        candidate_total = total + lengths[index]
        if (
            min_padding_efficiency > 0
            and candidate_longest > 0
            and candidate_total
            < min_padding_efficiency * (len(chosen) + 1) * candidate_longest
        ):
            continue
        chosen.append(index)
        longest = candidate_longest
        total = candidate_total
    return sorted(chosen)


def _graph_batch_sizes(batch_sizes, max_batch_size: int) -> tuple:
    """Batch sizes to capture: those below MAX_BATCH_SIZE plus the first one that
    covers it. Sizes above that would never be used. When none covers it, batches
    larger than the largest size run eagerly."""
    sizes = sorted(set(batch_sizes))
    kept = [size for size in sizes if size < max_batch_size]
    covering = [size for size in sizes if size >= max_batch_size]
    return tuple(kept + covering[:1])


def _round_up_samples(samples: int, step: int) -> int:
    """Pad a length up to the next whole step (at least one step)."""
    return max(1, math.ceil(samples / step)) * step


def _graph_shapes(batch_sizes, step: int, max_padded: int) -> list:
    """Every (batch, padded samples) shape to capture, largest first, so the shared
    memory pool reaches its peak on the first capture."""
    return [
        (batch, padded)
        for batch in sorted(batch_sizes, reverse=True)
        for padded in range(max_padded, 0, -step)
    ]


def _graph_key(batch: int, samples: int, batch_sizes, step: int, max_padded: int):
    """The captured shape for `batch` clips whose longest has `samples` samples, or
    None when the batch is outside the captured range (it then runs eagerly)."""
    if batch < 1 or samples < 1:
        return None
    padded_batch = next((size for size in sorted(batch_sizes) if size >= batch), None)
    padded = _round_up_samples(samples, step)
    if padded_batch is None or padded > max_padded:
        return None
    return padded_batch, padded


def _stack_waveforms(waveforms, length: int, rows=None):
    """Zero-pad waveforms into a (rows, length) float32 array plus int64 lengths.

    Rows past len(waveforms) (graph batch padding) are all-zero clips given the
    full length: a valid input whose output is dropped. A zero length would feed
    an empty clip to NeMo's per-feature normalization.
    """
    import numpy as np

    rows = len(waveforms) if rows is None else rows
    signals = np.empty((rows, length), dtype=np.float32)
    lengths = np.empty(rows, dtype=np.int64)
    _fill_waveforms(signals, lengths, waveforms)
    return signals, lengths


def _fill_waveforms(signals, lengths, waveforms):
    """_stack_waveforms into preallocated (rows, length) and (rows,) arrays."""
    rows, length = signals.shape
    if rows < len(waveforms):
        raise ValueError(f"{len(waveforms)} waveforms do not fit {rows} rows")
    signals.fill(0)
    lengths.fill(length)
    for index, waveform in enumerate(waveforms):
        if waveform.shape[0] > length:
            raise ValueError(f"waveform of {waveform.shape[0]} samples exceeds {length}")
        signals[index, : waveform.shape[0]] = waveform
        lengths[index] = waveform.shape[0]


def _batch_timing_fields(waveforms, path: str, encoder_frames: int) -> str:
    """The shape fields of a "Parakeet batch timing" log line."""
    lengths = [waveform.shape[0] for waveform in waveforms]
    max_length = max(lengths)
    return (
        f"batch={len(waveforms)}, "
        f"max_audio_s={max_length / AUDIO_SAMPLE_RATE_HZ:.2f}, "
        f"audio_s={sum(lengths) / AUDIO_SAMPLE_RATE_HZ:.3f}, "
        # Share of the padded batch that is real audio (1.0 = no padding).
        f"padding_efficiency={sum(lengths) / (len(waveforms) * max_length):.3f}, "
        f"encoder_path={path}, "
        f"encoder_frames={encoder_frames}"
    )


def _word_edits(a: str, b: str) -> int:
    matcher = difflib.SequenceMatcher(a=a.split(), b=b.split(), autojunk=False)
    return sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )


class EncoderGraphs:
    """CUDA graphs of model.forward() (preprocessor + encoder), keyed by (batch, samples).

    Ported from stt/orukeet/batch-short (#604). All graphs share one memory pool (a
    pool of their own, not the TDT decoder's), one static input buffer and one static
    output buffer, each graph using a view of the first batch x length elements. That
    is safe because one thread replays one graph at a time and run() clones the
    outputs before returning.
    """

    def __init__(self, model, batch_sizes, step_samples: int, max_samples: int):
        self.model = model
        self.batch_sizes = tuple(sorted(batch_sizes))
        self.step = int(step_samples)
        self.max_padded = _round_up_samples(int(max_samples), self.step)
        self.graphs = {}
        self.disabled_reason = None
        self.replays = 0
        self.pos_table_seconds = 0.0
        self.capture_seconds = 0.0
        self._pe_ptr = None
        self._signal = self._lengths = self._out = self._out_len = None
        self._pool = None

    def key(self, batch: int, samples: int):
        return _graph_key(batch, samples, self.batch_sizes, self.step, self.max_padded)

    def capture_all(self, pos_table_samples: int = 0):
        """Capture every shape. Any failure raises: the caller fails load()."""
        import torch

        encoder = self.model.encoder
        hop = int(self.model.preprocessor.featurizer.hop_length)
        # Fix the positional table before any capture. NeMo compares the feature
        # length with encoder.max_audio_length on every forward and replaces
        # pos_enc.pe with a new tensor when it is longer; a graph captured before
        # that would keep reading the old, freed table. Built in the encoder's
        # current (serving) dtype, as NeMo would.
        started_at = time.monotonic()
        max_frames = max(self.max_padded, int(pos_table_samples)) // hop + 1
        encoder.set_max_audio_length(max(encoder.max_audio_length, max_frames))
        self._pe_ptr = encoder.pos_enc.pe.data_ptr()
        torch.cuda.synchronize()
        self.pos_table_seconds = time.monotonic() - started_at

        started_at = time.monotonic()
        max_batch = self.batch_sizes[-1]
        with torch.inference_mode():
            self._signal = torch.zeros(
                max_batch * self.max_padded, dtype=torch.float32, device="cuda"
            )
            self._lengths = torch.zeros(max_batch, dtype=torch.int64, device="cuda")
        self._pool = torch.cuda.graph_pool_handle()
        torch.cuda.reset_peak_memory_stats()
        for batch, padded in _graph_shapes(self.batch_sizes, self.step, self.max_padded):
            self._capture(batch, padded)
        if self.model.encoder.pos_enc.pe.data_ptr() != self._pe_ptr:
            raise RuntimeError("encoder positional table was reallocated during capture")
        torch.cuda.synchronize()
        self.capture_seconds = time.monotonic() - started_at
        LOGGER.info(
            "Parakeet encoder graphs captured: "
            f"graphs={len(self.graphs)}, batch_sizes={self.batch_sizes}, "
            f"step_s={self.step / AUDIO_SAMPLE_RATE_HZ:g}, "
            f"max_padded_s={self.max_padded / AUDIO_SAMPLE_RATE_HZ:g}, "
            f"pos_enc_rows={encoder.pos_enc.pe.size(1)}, "
            f"pos_enc_dtype={encoder.pos_enc.pe.dtype}, "
            f"output_dtype={self._out.dtype}, "
            f"peak_allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.2f}, "
            f"pos_table_s={self.pos_table_seconds:.2f}, "
            f"capture_s={self.capture_seconds:.1f}"
        )

    def _views(self, batch: int, padded: int):
        return self._signal[: batch * padded].view(batch, padded), self._lengths[:batch]

    def _capture(self, batch: int, padded: int):
        import torch

        signal, lengths = self._views(batch, padded)
        # The serving precision context: a no-op under PRECISION_MODE=full or float32,
        # the only modes allowed with graphs, so nothing is captured under autocast.
        with torch.inference_mode(), _autocast():
            signal.zero_()
            lengths.fill_(padded)
            # Eager dry run: output shape, and kernel/workspace warmup for capture.
            encoded, _ = self.model.forward(input_signal=signal, input_signal_length=lengths)
            shape, dtype = tuple(encoded.shape), encoded.dtype
            del encoded
            if self._out is None:
                # Sized by the first (largest) shape, outside the graph pool.
                self._out = torch.empty(math.prod(shape), dtype=dtype, device="cuda")
                self._out_len = torch.empty(self.batch_sizes[-1], dtype=torch.int64, device="cuda")
            if dtype != self._out.dtype or math.prod(shape) > self._out.numel():
                raise RuntimeError(
                    f"encoder output {shape} {dtype} does not fit the shared output buffer"
                )
            out = self._out[: math.prod(shape)].view(shape)
            out_len = self._out_len[:batch]
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                encoded, encoded_len = self.model.forward(
                    input_signal=signal, input_signal_length=lengths
                )
                out.copy_(encoded)
                out_len.copy_(encoded_len)
                del encoded, encoded_len
        self.graphs[(batch, padded)] = (graph, out, out_len)

    def replay_key(self, waveforms):
        """The captured shape this batch would replay, or None if it runs eagerly."""
        if self.disabled_reason is not None:
            return None
        key = self.key(len(waveforms), max(w.shape[0] for w in waveforms))
        if key is None or key not in self.graphs:
            return None
        if self.model.encoder.pos_enc.pe.data_ptr() != self._pe_ptr:
            self.disable(
                "encoder positional table was reallocated after capture (an input "
                "longer than ENCODER_POS_TABLE_SECONDS ran eagerly)"
            )
            return None
        return key

    def run(self, waveforms, staging=None):
        """Return (encoded, encoded_len) for the batch, or None to run it eagerly.

        With a PinnedStaging the input is copied from pinned memory without blocking
        the host; without one, from pageable memory as before.
        """
        key = self.replay_key(waveforms)
        if key is None:
            return None
        import torch

        batch, padded = key
        graph, out, out_len = self.graphs[key]
        signal, lengths = self._views(batch, padded)
        if staging is None:
            host_signal, host_lengths = _stack_waveforms(waveforms, padded, rows=batch)
            signal.copy_(torch.from_numpy(host_signal))
            lengths.copy_(torch.from_numpy(host_lengths))
        else:
            slot, host_signal, host_lengths = staging.stage(waveforms, batch, padded)
            signal.copy_(host_signal, non_blocking=True)
            lengths.copy_(host_lengths, non_blocking=True)
            staging.copied(slot)
        graph.replay()
        self.replays += 1
        count = len(waveforms)
        return out[:count].clone(), out_len[:count].clone()

    def disable(self, reason: str):
        if self.disabled_reason is None:
            self.disabled_reason = reason
            LOGGER.error(f"Parakeet encoder graphs disabled, running eager: {reason}")


class PinnedStaging:
    """Double-buffered pinned host input for PIPELINE_OVERLAP encoder replays.

    A copy from pageable memory holds the batch worker until it lands; from pinned
    memory it is asynchronous, so the worker can go back to forming the next batch.
    A slot is rewritten only after the copy that read it has finished (its event), so
    one batch can be staged while the previous batch's copy is still in flight.
    """

    def __init__(self, rows: int, samples: int, slots: int = 2):
        import torch

        self._signals = [
            torch.empty(rows * samples, dtype=torch.float32, pin_memory=True)
            for _ in range(slots)
        ]
        self._lengths = [
            torch.empty(rows, dtype=torch.int64, pin_memory=True) for _ in range(slots)
        ]
        self._copied = [None] * slots
        self._next = 0

    def stage(self, waveforms, rows: int, length: int):
        """Fill the next slot; returns (slot, (rows, length) signal, (rows,) lengths)."""
        slot = self._next
        self._next = (slot + 1) % len(self._signals)
        if self._copied[slot] is not None:
            self._copied[slot].synchronize()
        if rows * length > self._signals[slot].numel() or rows > self._lengths[slot].numel():
            raise ValueError(f"batch {rows} x {length} does not fit the pinned staging buffer")
        signal = self._signals[slot][: rows * length].view(rows, length)
        lengths = self._lengths[slot][:rows]
        _fill_waveforms(signal.numpy(), lengths.numpy(), waveforms)
        return slot, signal, lengths

    def copied(self, slot: int):
        """Mark the slot's device copies as enqueued on the current stream."""
        import torch

        event = torch.cuda.Event()
        event.record()
        self._copied[slot] = event


class OverlapPipeline:
    """The decode stage of PIPELINE_OVERLAP: one thread, one decode at a time.

    submit() hands over one encoded batch and blocks while another is already waiting
    (queue depth 1), so the encoder runs at most one batch ahead of the decode in
    flight. Every batch submit() accepts gets exactly one on_done(result) call, with
    decode_fn's outputs or the exception that batch hit; a batch submit() refuses
    raises there and is never called back. drain() waits until every accepted batch
    has been delivered, then runs sync_fn (a device synchronize), so serial GPU work
    after it sees an idle GPU. exclusive() wraps such serial work.
    """

    _STOP = object()
    _POLL_SECONDS = 0.05

    def __init__(self, decode_fn, sync_fn, timeout_seconds: float):
        self._decode_fn = decode_fn
        self._sync_fn = sync_fn
        self._timeout_seconds = timeout_seconds
        self._queue = queue.Queue(maxsize=1)
        self._state = threading.Condition()
        self._pending = 0
        self._closed = False
        self._fatal = None
        self.delivered = 0
        self.drains = 0
        self._thread = threading.Thread(
            target=self._decode_loop, name="parakeet-decode", daemon=True
        )
        self._thread.start()

    def _refuse_if_unusable(self):
        if self._closed:
            raise RuntimeError("Parakeet decode pipeline is shut down")
        if self._fatal is not None:
            raise RuntimeError("Parakeet decode pipeline failed") from self._fatal

    def submit(self, handoff, on_done):
        with self._state:
            self._refuse_if_unusable()
            self._pending += 1
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            try:
                self._queue.put((handoff, on_done), timeout=self._POLL_SECONDS)
                return
            except queue.Full:
                pass
            with self._state:
                expired = time.monotonic() > deadline
                if self._closed or self._fatal is not None or expired:
                    self._pending -= 1
                    self._state.notify_all()
                    self._refuse_if_unusable()
                    raise TimeoutError(
                        "Parakeet decode stage did not take the next batch within "
                        f"{self._timeout_seconds:g} seconds"
                    )

    def _decode_loop(self):
        while True:
            item = self._queue.get()
            if item is self._STOP:
                return
            self._deliver(*item)

    def _deliver(self, handoff, on_done, result=None):
        try:
            if result is None:
                try:
                    if self._fatal is not None:
                        raise RuntimeError("Parakeet decode pipeline failed") from self._fatal
                    result = self._decode_fn(handoff)
                except BaseException as exc:
                    result = exc
            on_done(result)
        except BaseException as exc:
            # on_done is the batch's own delivery; if it fails, stop accepting work.
            LOGGER.exception("Parakeet decode pipeline delivery failed")
            with self._state:
                self._fatal = exc
        finally:
            with self._state:
                self._pending -= 1
                self.delivered += 1
                self._state.notify_all()

    def drain(self):
        deadline = time.monotonic() + self._timeout_seconds
        with self._state:
            while self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Parakeet decode pipeline still has {self._pending} batches "
                        f"after {self._timeout_seconds:g} seconds"
                    )
                self._state.wait(remaining)
            self.drains += 1
        self._sync_fn()

    @contextlib.contextmanager
    def exclusive(self):
        """Serial GPU work on the default stream: drain before it, and sync after it.

        PyTorch creates side streams non-blocking with respect to the default stream,
        so without the second sync the encode and decode streams could run ahead of
        what the serial work wrote, such as a rebuilt decoder's state tensors or the
        encoder graphs' static input.
        """
        self.drain()
        try:
            yield
        finally:
            self._sync_fn()

    def close(self):
        """Decode what was already handed over, stop the thread, fail anything left."""
        with self._state:
            if self._closed:
                return
            self._closed = True
        started_at = time.monotonic()
        LOGGER.info("Parakeet decode pipeline shutdown started")
        deadline = started_at + self._timeout_seconds
        while self._thread.is_alive() and time.monotonic() < deadline:
            try:
                self._queue.put(self._STOP, timeout=self._POLL_SECONDS)
                break
            except queue.Full:
                pass
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        stopped = not self._thread.is_alive()
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not self._STOP:
                self._deliver(*item, result=RuntimeError("Parakeet decode pipeline is shut down"))
        if not stopped:
            LOGGER.error("Parakeet decode thread did not stop; a decode may still be running")
        LOGGER.info(
            "Parakeet decode pipeline shutdown finished: "
            f"stopped={stopped}, delivered={self.delivered}, "
            f"elapsed_s={time.monotonic() - started_at:.3f}"
        )


def _to_jsonable(obj):
    """Recursively convert numpy scalars/arrays in NeMo timestamp dicts."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _restore_eval_after_transcribe(model):
    """Put the model back in inference state after NeMo's transcribe().

    NeMo's ASR transcribe() (2.7.3 and 3.0.0) ends with encoder/decoder/joint
    .unfreeze(partial=True), and unfreeze() calls module.train(), so those modules
    stay in training mode, dropout on, for every later pass. transcribe() also
    restores the checkpoint's featurizer dither and pad_to. The direct path depends on
    eval mode and on the dither=0 / pad_to=0 that load() set, so restore all three.
    Returns the modules transcribe() had left in training mode.
    """
    left_training = [
        name
        for name in ("encoder", "decoder", "joint")
        if getattr(getattr(model, name, None), "training", False)
    ]
    model.eval()
    if DIRECT_FORWARD:
        featurizer = getattr(getattr(model, "preprocessor", None), "featurizer", None)
        if featurizer is not None:
            if hasattr(featurizer, "dither"):
                featurizer.dither = 0.0
            if hasattr(featurizer, "pad_to"):
                featurizer.pad_to = 0
    if left_training and not _TRANSCRIBE_MODE_LOGGED:
        _TRANSCRIBE_MODE_LOGGED.append(True)
        LOGGER.info(
            f"NeMo transcribe() left these modules in training mode: {left_training}; "
            "restored eval mode"
        )
    return left_training


class Model:
    def __init__(self, **kwargs):
        self._model = None
        self._jobs = queue.Queue()
        self._deferred_jobs = collections.deque()
        # predict() calls between entry and enqueue (BATCH_WINDOW_EARLY_EXIT).
        self._admitting = 0
        self._admitting_lock = threading.Lock()
        self._http = None
        self._graphs = None
        # Load-time cfg.decoding and the decoding object the direct path serves with
        # (see _ensure_serving_decoding); None until load() snapshots them.
        self._decoding_cfg = None
        self._serving_decoding = None
        self._parity = {"batches": 0, "clips": 0, "identical": 0, "word_edits": 0, "words": 0}
        # PIPELINE_OVERLAP state, set by _start_pipeline() at the end of load().
        self._pipeline = None
        self._encode_stream = self._decode_stream = None
        self._staging = None
        self._timing_origin = None

    def load(self):
        import httpx
        import nemo
        import nemo.collections.asr as nemo_asr
        import torch

        load_started_at = time.monotonic()
        LOGGER.info(f"Restoring NeMo checkpoint: {NEMO_CHECKPOINT}")
        self._model = nemo_asr.models.ASRModel.restore_from(
            NEMO_CHECKPOINT, map_location=torch.device("cuda")
        )
        self._model.eval()
        if PRECISION_MODE == "full":
            self._cast_full_precision(self._model)
        if not ENCODER_CUDNN:
            # Set once, before any GPU pass: all GPU work runs on one thread
            # (load(), then the batch worker), and warmup captures the decoder's
            # CUDA graph under the same setting that serves.
            torch.backends.cudnn.enabled = False
        LOGGER.info(
            f"NeMo checkpoint restore succeeded: checkpoint={NEMO_CHECKPOINT}, "
            f"nemo_version={getattr(nemo, '__version__', 'unknown')}, "
            f"torch_version={torch.__version__}, torch_cuda={torch.version.cuda}, "
            f"cudnn_version={torch.backends.cudnn.version()}, "
            f"model_class={type(self._model).__name__}, "
            f"encoder_class={type(getattr(self._model, 'encoder', None)).__name__}, "
            f"decoder_class={type(getattr(self._model, 'decoder', None)).__name__}"
        )
        if DIRECT_FORWARD:
            # Match TranscriptionMixin._transcribe_on_begin() once at startup.
            # The direct path intentionally bypasses that per-call setup.
            preprocessor = getattr(self._model, "preprocessor", None)
            featurizer = getattr(preprocessor, "featurizer", None)
            if featurizer is not None:
                if hasattr(featurizer, "dither"):
                    featurizer.dither = 0.0
                if hasattr(featurizer, "pad_to"):
                    featurizer.pad_to = 0
        self._http = httpx.Client(timeout=60, follow_redirects=False)

        LOGGER.info(
            f"Parakeet runtime config: direct_forward={DIRECT_FORWARD}, "
            f"max_batch_size={MAX_BATCH_SIZE}, "
            f"batch_window_ms={BATCH_WINDOW_SECONDS * 1000:g}, "
            f"batch_buckets_seconds={BATCH_BUCKET_SECONDS or 'disabled'}, "
            f"batch_length_bucketing={BATCH_LENGTH_BUCKETING}, "
            f"batch_bucket_lookahead={BATCH_BUCKET_LOOKAHEAD}, "
            f"batch_min_padding_efficiency={BATCH_MIN_PADDING_EFFICIENCY:g}, "
            f"batch_window_early_exit={BATCH_WINDOW_EARLY_EXIT}, "
            f"precision_mode={PRECISION_MODE}, inference_dtype={INFERENCE_DTYPE}, "
            f"encoder_cudnn={ENCODER_CUDNN} "
            f"(torch.backends.cudnn.enabled={torch.backends.cudnn.enabled}), "
            f"startup_transcript_check={STARTUP_TRANSCRIPT_CHECK}, "
            f"timing_log_batches={TIMING_LOG_BATCHES}, "
            f"encoder_cuda_graphs={ENCODER_CUDA_GRAPHS}, "
            f"encoder_graph_batch_sizes={ENCODER_GRAPH_BATCH_SIZES}, "
            f"encoder_graph_max_seconds={ENCODER_GRAPH_MAX_SECONDS:g}, "
            f"encoder_graph_step_seconds={ENCODER_GRAPH_STEP_SECONDS:g}, "
            f"encoder_pos_table_seconds={ENCODER_POS_TABLE_SECONDS:g}, "
            f"graph_parity_batches={GRAPH_PARITY_BATCHES}, "
            f"pipeline_overlap={PIPELINE_OVERLAP}"
        )
        self._log_decoder_graph_mode("load", enforce=False)
        if ENCODER_CUDA_GRAPHS:
            # After the PRECISION_MODE cast, the cuDNN setting and the featurizer's
            # dither/pad_to, so graphs record exactly the serving computation, and
            # before warmup captures the TDT decoder's full graph: on RTX PRO 6000 a
            # CUDA graph captured after NeMo's full-graph decoder has made that
            # decoder graph fault with "illegal memory access" (stt/orukeet/batch).
            # Any capture failure raises and fails the replica.
            self._capture_encoder_graphs()
        # NeMo's transcribe(timestamps=True), run by the startup mode check below and
        # by every timestamps request, flips cfg.decoding.compute_timestamps and
        # rebuilds the decoding object, dropping the decoder CUDA graph warmup
        # captured. Snapshot the decoding setup so _restore_decoding() can rebuild
        # it and warm it again.
        self._decoding_cfg = copy.deepcopy(self._model.cfg.decoding)
        self._serving_decoding = self._model.decoding
        self._warmup()
        self._log_decoder_graph_mode("warmup", enforce=REQUIRE_FULL_CUDA_GRAPH)
        if STARTUP_MODE_CHECK:
            self._check_transcribe_mode(self._model)
            self._log_decoder_graph_mode("after_transcribe", enforce=False)
        if self._model.decoding is not self._serving_decoding:
            self._restore_decoding("rewarm_after_transcribe")
        check_waveform = None
        if STARTUP_TRANSCRIPT_CHECK or PIPELINE_OVERLAP:
            check_waveform = self._decode_waveform(
                self._decode_audio_input({"audio_url": STARTUP_CHECK_AUDIO_URL})
            )
        if STARTUP_TRANSCRIPT_CHECK:
            self._check_startup_transcript(check_waveform)
        if PIPELINE_OVERLAP:
            self._start_pipeline()
        try:
            if PIPELINE_OVERLAP:
                self._check_pipeline_parity(check_waveform)
            self._assert_eval_mode("startup")
        except BaseException:
            self.shutdown()
            raise
        if FREEZE_GC_AFTER_WARMUP:
            gc.collect()
            gc.freeze()
            LOGGER.info("Python GC startup state frozen")

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        graphs = self._graphs
        LOGGER.info(
            "Parakeet startup complete: "
            f"total_s={time.monotonic() - load_started_at:.1f}, "
            f"pipeline_overlap={PIPELINE_OVERLAP}, "
            f"encoder_graphs={len(graphs.graphs) if graphs else 0}, "
            f"encoder_graph_capture_s={graphs.capture_seconds if graphs else 0:.1f}, "
            f"positional_table_s={graphs.pos_table_seconds if graphs else 0:.2f}, "
            f"torch_reserved_gib={torch.cuda.memory_reserved() / 2**30:.2f}, "
            f"gpu_used_gib={(total_bytes - free_bytes) / 2**30:.2f} of {total_bytes / 2**30:.2f}"
        )
        _TIMING_STATE["serving"] = True
        threading.Thread(target=self._batch_worker, daemon=True).start()

    def _capture_encoder_graphs(self):
        sizes = _graph_batch_sizes(ENCODER_GRAPH_BATCH_SIZES, MAX_BATCH_SIZE)
        if MAX_BATCH_SIZE > sizes[-1]:
            LOGGER.warning(
                f"MAX_BATCH_SIZE={MAX_BATCH_SIZE} exceeds the largest encoder graph "
                f"({sizes[-1]}); larger batches run eagerly"
            )
        self._graphs = EncoderGraphs(
            self._model,
            sizes,
            round(ENCODER_GRAPH_STEP_SECONDS * AUDIO_SAMPLE_RATE_HZ),
            round(ENCODER_GRAPH_MAX_SECONDS * AUDIO_SAMPLE_RATE_HZ),
        )
        self._graphs.capture_all(round(ENCODER_POS_TABLE_SECONDS * AUDIO_SAMPLE_RATE_HZ))

    @staticmethod
    def _cast_full_precision(model):
        """Cast encoder, decoder and joint to INFERENCE_DTYPE once (PRECISION_MODE=full).

        The preprocessor has no parameters; its float32 filterbank and window stay
        float32 and it still computes from the float32 waveform. Only its output
        dtype changes: NeMo's AudioPreprocessor casts its output to the dtype of
        `dtype_sentinel_tensor`, which is what model.to(dtype) would change too.
        Any failure here fails load(); there is no fallback to autocast.
        """
        import torch

        if INFERENCE_DTYPE == "float32":
            LOGGER.info("Parakeet full precision: float32, nothing to cast")
            return
        dtype = getattr(torch, INFERENCE_DTYPE)
        preprocessor = model.preprocessor
        sentinel = getattr(preprocessor, "dtype_sentinel_tensor", None)
        if sentinel is None:
            raise RuntimeError(
                "PRECISION_MODE=full: preprocessor has no dtype_sentinel_tensor; "
                "cannot set its output dtype"
            )
        for module_name in ("encoder", "decoder", "joint"):
            module = getattr(model, module_name, None)
            if module is None:
                raise RuntimeError(f"PRECISION_MODE=full: model has no {module_name}")
            module.to(dtype)
        preprocessor.dtype_sentinel_tensor = torch.tensor(
            (), dtype=dtype, device=sentinel.device
        )
        by_dtype = collections.Counter()
        for module_name in ("preprocessor", "encoder", "decoder", "joint"):
            module = getattr(model, module_name)
            for tensor in list(module.parameters()) + list(module.buffers()):
                if tensor.is_floating_point():
                    dtype_name = str(tensor.dtype).replace("torch.", "")
                    by_dtype[f"{module_name}:{dtype_name}"] += tensor.numel()
        stray = [
            key
            for key in by_dtype
            if not key.startswith("preprocessor:") and not key.endswith(INFERENCE_DTYPE)
        ]
        if stray:
            raise RuntimeError(
                f"PRECISION_MODE=full: tensors left outside {INFERENCE_DTYPE}: {stray}"
            )
        LOGGER.info(
            f"Parakeet full precision ready: dtype={INFERENCE_DTYPE}, autocast=off, "
            "preprocessor computes float32 and outputs "
            f"{preprocessor.dtype_sentinel_tensor.dtype}, "
            "float_elements_by_module_dtype="
            + ", ".join(f"{key}={value}" for key, value in sorted(by_dtype.items()))
        )

    def _assert_eval_mode(self, phase: str):
        training = _modules_in_training_mode(self._model)
        if training:
            raise RuntimeError(
                f"Parakeet {phase} check: {len(training)} modules in training mode, "
                f"e.g. {training[:5]}"
            )
        LOGGER.info(f"Parakeet {phase} check passed: no module in training mode")

    def _restore_decoding(self, phase: str, *, decoder_only: bool = False):
        """Rebuild the load-time decoding setup after transcribe() replaced it.

        The rebuilt object decodes with the load-time config (compute_timestamps as
        in the checkpoint), and warmup runs again so its decoder CUDA graph is
        captured, and REQUIRE_FULL_CUDA_GRAPH checked, before it serves. Rebuilt
        rather than putting the old object back: on T4 / NeMo 2.7.3 the old
        object's CUDA-graph TDT decoder hit an illegal memory access on the next
        batch (#873 CI, deployment qe9jg55).
        """
        started_at = time.monotonic()
        LOGGER.info(f"Parakeet decoding restore started: phase={phase}")
        self._model.change_decoding_strategy(
            copy.deepcopy(self._decoding_cfg), verbose=False
        )
        # Set before warmup, whose direct-path passes must not restore again.
        self._serving_decoding = self._model.decoding
        try:
            self._warmup(decoder_only=decoder_only)
            self._log_decoder_graph_mode(phase, enforce=REQUIRE_FULL_CUDA_GRAPH)
        except Exception:
            # Unverified: the next direct-path batch rebuilds and checks again.
            self._serving_decoding = None
            raise
        LOGGER.info(
            f"Parakeet decoding restore finished: phase={phase}, compute_timestamps="
            f"{self._model.cfg.decoding.get('compute_timestamps')}, "
            f"elapsed_s={time.monotonic() - started_at:.3f}"
        )

    def _ensure_serving_decoding(self):
        """Restore the warmed load-time decoding before a direct-path batch.

        A timestamps batch (NeMo transcribe(timestamps=True)) leaves a fresh
        compute_timestamps=True decoding object on the model. Restoring here, on the
        batch worker thread that runs all GPU work, rather than after every
        timestamps batch, means a run of timestamps batches rebuilds nothing, and
        no direct-path batch can run on a half-restored model.
        """
        if (
            self._decoding_cfg is not None
            and self._model.decoding is not self._serving_decoding
        ):
            self._restore_decoding("rewarm_after_timestamps", decoder_only=True)

    def predict(self, request: dict) -> dict:
        # Wrap predict so tracebacks reach the deployment logs.
        try:
            return self._predict_impl(request)
        except Exception:
            LOGGER.exception("Parakeet prediction failed")
            raise

    def _predict_impl(self, request: dict) -> dict:
        request_started_at = time.monotonic()
        preprocessing_started_at = request_started_at
        with self._admitting_lock:
            self._admitting += 1
        admitting = True
        try:
            try:
                audio_bytes = self._decode_audio_input(request)
                with_timestamps = bool(request.get("timestamps", False))
                waveform = self._decode_waveform(audio_bytes)
                duration_seconds = waveform.shape[0] / AUDIO_SAMPLE_RATE_HZ
                job = {
                    "waveform": waveform,
                    "duration_seconds": duration_seconds,
                    "bucket": self._duration_bucket(duration_seconds),
                    "timestamps": with_timestamps,
                    "done": threading.Event(),
                }
            finally:
                PARAKEET_METRICS.observe_latency(
                    "preprocessing", time.monotonic() - preprocessing_started_at
                )

            job["enqueued_at"] = time.monotonic()
            PARAKEET_METRICS.queue_depth.inc()
            self._jobs.put(job)
            # After put(): see _nothing_incoming for why the order matters.
            self._leave_admission()
            admitting = False
            if not job["done"].wait(timeout=PREDICT_TIMEOUT_SECONDS):
                raise TimeoutError(
                    "Parakeet batch worker did not finish within "
                    f"{PREDICT_TIMEOUT_SECONDS:g} seconds"
                )

            postprocessing_started_at = job["inference_finished_at"]
            try:
                if "error" in job:
                    raise job["error"]

                hypothesis = job["output"]
                result = {"transcript": hypothesis.text, "text": hypothesis.text}
                if with_timestamps:
                    stamps = hypothesis.timestamp or {}
                    result["timestamps"] = _to_jsonable(
                        {
                            "word": stamps.get("word", []),
                            "segment": stamps.get("segment", []),
                        }
                    )
                return result
            finally:
                # Includes the worker-to-handler handoff as well as response shaping.
                PARAKEET_METRICS.observe_latency(
                    "postprocessing", time.monotonic() - postprocessing_started_at
                )
        finally:
            if admitting:
                self._leave_admission()
            PARAKEET_METRICS.observe_latency(
                "total", time.monotonic() - request_started_at
            )

    def _leave_admission(self):
        with self._admitting_lock:
            if self._admitting < 1:
                raise RuntimeError("Parakeet admission count went negative")
            self._admitting -= 1

    def _nothing_incoming(self) -> bool:
        """BATCH_WINDOW_EARLY_EXIT: True when no clip already in the pod can join
        the batch being formed. Reads the admission count before the queue: a
        predict() call enqueues before it leaves admission, so a zero count read
        first means every admitted clip is already visible in the queue."""
        if not BATCH_WINDOW_EARLY_EXIT:
            return False
        with self._admitting_lock:
            admitting = self._admitting
        return admitting == 0 and self._jobs.empty()

    def _batch_worker(self):
        """Run all GPU work on one thread and opportunistically microbatch."""
        while True:
            first = self._next_job()
            if BATCH_LENGTH_BUCKETING:
                jobs = self._length_bucketed_batch(first)
            else:
                jobs = self._arrival_order_batch(first)
            self._run_jobs(jobs)

    def _arrival_order_batch(self, first):
        """Fill a batch in arrival order with clips from the first clip's bucket."""
        jobs = [first]
        deadline = time.monotonic() + BATCH_WINDOW_SECONDS

        # First scan jobs deferred by earlier, incompatible batches. This
        # keeps the queue work-conserving without mixing very different
        # audio lengths (which would pad every item to the longest clip).
        for _ in range(len(self._deferred_jobs)):
            if len(jobs) >= MAX_BATCH_SIZE:
                break
            candidate = self._deferred_jobs.popleft()
            if self._jobs_compatible(first, candidate):
                jobs.append(self._select_job(candidate))
            else:
                self._deferred_jobs.append(candidate)

        while len(jobs) < MAX_BATCH_SIZE:
            timeout = deadline - time.monotonic()
            if timeout <= 0 or self._nothing_incoming():
                break
            try:
                candidate = self._jobs.get(timeout=timeout)
            except queue.Empty:
                break
            if self._jobs_compatible(first, candidate):
                jobs.append(self._select_job(candidate))
            else:
                self._deferred_jobs.append(candidate)
        return jobs

    def _length_bucketed_batch(self, first):
        """Build a batch from the oldest clip and the pending clips closest in length.

        Runs on the worker thread, which alone touches _deferred_jobs; here that
        deque is the pending pool, in arrival order. Within the batch window it pulls
        queued clips into the pool until the pool could fill a batch or holds
        BATCH_BUCKET_LOOKAHEAD clips, then takes whatever is already queued, up to
        the lookahead, without waiting. Clips not chosen stay pending, in order.
        With BATCH_WINDOW_EARLY_EXIT it also stops waiting once nothing is incoming.
        """
        deadline = time.monotonic() + BATCH_WINDOW_SECONDS
        pending = self._deferred_jobs
        compatible = sum(self._length_compatible(first, job) for job in pending)
        while len(pending) + 1 < BATCH_BUCKET_LOOKAHEAD:
            timeout = deadline - time.monotonic()
            try:
                if (
                    compatible + 1 < MAX_BATCH_SIZE
                    and timeout > 0
                    and not self._nothing_incoming()
                ):
                    candidate = self._jobs.get(timeout=timeout)
                else:
                    candidate = self._jobs.get_nowait()
            except queue.Empty:
                break
            pending.append(candidate)
            compatible += self._length_compatible(first, candidate)

        candidates = [job for job in pending if self._length_compatible(first, job)]
        chosen = [
            candidates[index - 1]
            for index in _closest_length_batch(
                [first["duration_seconds"]]
                + [job["duration_seconds"] for job in candidates],
                self._length_batch_cap,
                BATCH_MIN_PADDING_EFFICIENCY,
            )[1:]
        ]
        chosen_ids = {id(job) for job in chosen}
        self._deferred_jobs = collections.deque(
            job for job in pending if id(job) not in chosen_ids
        )
        return [first] + [self._select_job(job) for job in chosen]

    def _run_jobs(self, jobs):
        """Run one formed batch and wake every request in it, whatever happens.

        With PIPELINE_OVERLAP an eligible batch is encoded here and handed to the
        decode thread, which wakes its requests; any other batch drains the pipeline
        and runs serially on this thread.
        """
        first = jobs[0]
        inference_started_at = time.monotonic()
        for job in jobs:
            PARAKEET_METRICS.observe_latency(
                "batching", inference_started_at - job["selected_at"]
            )
        PARAKEET_METRICS.batch_size.observe(len(jobs))
        waveforms = [job["waveform"] for job in jobs]
        if self._pipeline is not None and self._overlap_eligible(first, waveforms):
            self._submit_overlapped(jobs, waveforms, inference_started_at)
            return
        result = RuntimeError("Parakeet batch was interrupted")
        try:
            # Timestamps transcribe(), decoding rebuilds, eager encoder batches and
            # graph parity reruns run on the default stream, alone.
            serial = (
                self._pipeline.exclusive()
                if self._pipeline is not None
                else contextlib.nullcontext()
            )
            with serial:
                result = self._infer_waveforms(
                    waveforms, with_timestamps=first["timestamps"]
                )
        except Exception as exc:
            result = exc
        finally:
            self._finish_jobs(jobs, inference_started_at, result)

    @staticmethod
    def _finish_jobs(jobs, inference_started_at, result):
        """Give each request its output, or the batch's exception, and wake it."""
        try:
            if isinstance(result, BaseException):
                error = result
            elif not isinstance(result, (list, tuple)) or len(result) != len(jobs):
                count = len(result) if isinstance(result, (list, tuple)) else type(result).__name__
                error = RuntimeError(f"NeMo returned {count} outputs for {len(jobs)} inputs")
            else:
                error = None
                for job, output in zip(jobs, result):
                    job["output"] = output
            if error is not None:
                for job in jobs:
                    job["error"] = error
        finally:
            inference_finished_at = time.monotonic()
            inference_latency = inference_finished_at - inference_started_at
            for job in jobs:
                job["inference_finished_at"] = inference_finished_at
                PARAKEET_METRICS.observe_latency("inference", inference_latency)
                job["done"].set()

    def _graph_parity_due(self) -> bool:
        return _TIMING_STATE["serving"] and self._parity["batches"] < GRAPH_PARITY_BATCHES

    def _overlap_eligible(self, first, waveforms) -> bool:
        """PIPELINE_OVERLAP: True when the batch can be encoded while the previous
        batch decodes: a direct-path batch on the warmed serving decoding whose
        encoder graph is captured, and no parity rerun due."""
        return (
            not first["timestamps"]
            and self._model.decoding is self._serving_decoding
            and not self._graph_parity_due()
            and self._graphs.replay_key(waveforms) is not None
        )

    def _submit_overlapped(self, jobs, waveforms, inference_started_at):
        try:
            handoff = self._overlap_encode(waveforms)
            self._pipeline.submit(
                handoff,
                lambda result: self._finish_jobs(jobs, inference_started_at, result),
            )
        except Exception as exc:
            # Encode failed or submit refused the batch: it will not be called back.
            self._finish_jobs(jobs, inference_started_at, exc)

    def _overlap_encode(self, waveforms):
        """Encode stage, on the batch worker: stage the input in pinned memory and
        replay the encoder graph on the encode stream, then record the event the
        decode stream waits for. Returns the handoff for _overlap_decode."""
        import torch

        timed = _TIMING_STATE["serving"] and _TIMING_STATE["batches"] < TIMING_LOG_BATCHES
        if timed:
            _TIMING_STATE["batches"] += 1
        stream = self._encode_stream
        with torch.inference_mode(), _autocast(), torch.cuda.stream(stream):
            encode_started = None
            if timed:
                encode_started = torch.cuda.Event(enable_timing=True)
                encode_started.record(stream)
            graphed = self._graphs.run(waveforms, staging=self._staging)
            if graphed is None:
                raise RuntimeError("encoder graph for an overlapped batch did not replay")
            encoded, encoded_length = graphed
            ready = torch.cuda.Event(enable_timing=timed)
            ready.record(stream)
        if _TIMING_STATE["serving"]:
            PARAKEET_METRICS.encoder_batches["graph"].inc()
        return {
            "waveforms": waveforms,
            "encoded": encoded,
            "encoded_length": encoded_length,
            "ready": ready,
            "encode_started": encode_started,
            "submitted_at": time.monotonic(),
        }

    def _overlap_decode(self, handoff):
        """Decode stage, on the decode thread: wait for the batch's encoder event on
        the decode stream, then run the same TDT decode as the serial path."""
        import torch

        stream = self._decode_stream
        timed = handoff["encode_started"] is not None
        with torch.inference_mode(), _autocast(), torch.cuda.stream(stream):
            stream.wait_event(handoff["ready"])
            encoded, encoded_length = handoff["encoded"], handoff["encoded_length"]
            # Allocated on the encode stream and released after this decode: without
            # this, the allocator could hand the memory back to the encode stream
            # while this stream's kernels are still reading it.
            encoded.record_stream(stream)
            encoded_length.record_stream(stream)
            handoff_wait_ms = (time.monotonic() - handoff["submitted_at"]) * 1000
            if timed:
                decode_started = torch.cuda.Event(enable_timing=True)
                decode_started.record(stream)
            hypotheses = self._model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,
                encoded_lengths=encoded_length,
                return_hypotheses=True,
            )
            if timed:
                decode_finished = torch.cuda.Event(enable_timing=True)
                decode_finished.record(stream)
                decode_finished.synchronize()
                origin = self._timing_origin
                LOGGER.info(
                    "Parakeet batch timing: "
                    + _batch_timing_fields(handoff["waveforms"], "graph", encoded.shape[-1])
                    + ", "
                    f"encoder_ms={handoff['encode_started'].elapsed_time(handoff['ready']):.1f}, "
                    f"decoder_ms={decode_started.elapsed_time(decode_finished):.1f}, "
                    "pipeline=overlap, "
                    # Stage starts on one GPU timeline (ms since the pipeline started).
                    f"encoder_start_ms={origin.elapsed_time(handoff['encode_started']):.1f}, "
                    f"decoder_start_ms={origin.elapsed_time(decode_started):.1f}, "
                    f"handoff_wait_ms={handoff_wait_ms:.1f}"
                )
        return hypotheses

    def _start_pipeline(self):
        """PIPELINE_OVERLAP: create the encode and decode streams, the pinned input
        staging and the decode thread. Runs after every serial startup pass."""
        import torch

        started_at = time.monotonic()
        LOGGER.info("Parakeet decode pipeline start")
        if self._graphs is None:
            raise RuntimeError("PIPELINE_OVERLAP needs captured encoder graphs")
        torch.cuda.synchronize()
        self._encode_stream = torch.cuda.Stream()
        self._decode_stream = torch.cuda.Stream()
        self._staging = PinnedStaging(self._graphs.batch_sizes[-1], self._graphs.max_padded)
        self._timing_origin = torch.cuda.Event(enable_timing=True)
        self._timing_origin.record()
        self._pipeline = OverlapPipeline(
            self._overlap_decode, torch.cuda.synchronize, PREDICT_TIMEOUT_SECONDS
        )
        LOGGER.info(
            "Parakeet decode pipeline started: "
            f"staging_rows={self._graphs.batch_sizes[-1]}, "
            f"staging_s={self._graphs.max_padded / AUDIO_SAMPLE_RATE_HZ:g}, "
            f"elapsed_s={time.monotonic() - started_at:.3f}"
        )

    def _check_pipeline_parity(self, waveform):
        """PIPELINE_OVERLAP startup check: decode one multi-clip batch serially and
        twice through the pipeline, back to back so the second encode overlaps the
        first decode, and fail startup unless every transcript is identical."""
        started_at = time.monotonic()
        LOGGER.info("Parakeet pipeline parity check start")
        samples = waveform.shape[0]
        batch = [
            waveform,
            waveform[: samples * 2 // 3],
            waveform[samples // 3 :],
            waveform[: samples // 2],
        ]
        if self._graphs.replay_key(batch) is None:
            raise RuntimeError(
                "pipeline parity check: the check batch has no captured encoder graph "
                f"({self._graphs.disabled_reason or 'shape outside the captured range'})"
            )
        with self._pipeline.exclusive():
            serial = [
                _hypothesis_text(h)
                for h in self._infer_waveforms(batch, with_timestamps=False)
            ]
        results = [None, None]

        def deliver(index):
            def on_done(result):
                results[index] = result

            return on_done

        for index in range(len(results)):
            self._pipeline.submit(self._overlap_encode(batch), deliver(index))
        self._pipeline.drain()
        for index, result in enumerate(results):
            if isinstance(result, BaseException):
                raise RuntimeError(
                    f"pipeline parity check: overlapped pass {index} failed"
                ) from result
            overlapped = [_hypothesis_text(h) for h in result]
            if overlapped != serial:
                differing = sum(a != b for a, b in zip(serial, overlapped))
                raise RuntimeError(
                    f"pipeline parity check failed: overlapped pass {index} changed "
                    f"{differing} of {len(serial)} transcripts; serial={serial!r}, "
                    f"overlapped={overlapped!r}"
                )
        LOGGER.info(
            "Parakeet pipeline parity check passed: "
            f"clips={len(batch)}, overlapped_passes={len(results)}, identical=True, "
            f"elapsed_s={time.monotonic() - started_at:.2f}"
        )

    def shutdown(self):
        """Stop the PIPELINE_OVERLAP decode thread. Batches already handed over are
        decoded; later overlapped batches fail instead of running."""
        if self._pipeline is not None:
            self._pipeline.close()

    def _next_job(self):
        if self._deferred_jobs:
            return self._select_job(self._deferred_jobs.popleft())
        return self._select_job(self._jobs.get())

    @staticmethod
    def _select_job(job):
        job["selected_at"] = time.monotonic()
        PARAKEET_METRICS.queue_depth.dec()
        PARAKEET_METRICS.observe_latency(
            "queueing", job["selected_at"] - job["enqueued_at"]
        )
        return job

    @staticmethod
    def _length_batch_cap(_longest_seconds: float) -> int:
        # This preset has no per-length memory cap; every batch may reach
        # MAX_BATCH_SIZE, as in the coarse-bucket batcher.
        return MAX_BATCH_SIZE

    @staticmethod
    def _length_compatible(first, candidate) -> bool:
        # Length bucketing ignores the coarse buckets but keeps timestamp requests
        # (NeMo transcribe()) out of direct-path batches and vice versa.
        return first["timestamps"] is candidate["timestamps"]

    @staticmethod
    def _jobs_compatible(first, candidate) -> bool:
        return (
            first["timestamps"] is candidate["timestamps"]
            and first["bucket"] == candidate["bucket"]
        )

    @staticmethod
    def _duration_bucket(duration_seconds: float):
        for boundary in BATCH_BUCKET_SECONDS:
            if duration_seconds <= boundary:
                return boundary
        if not BATCH_BUCKET_SECONDS:
            return float("inf")

        # Continue geometrically beyond the configured short-audio buckets so
        # an occasional long recording is not paired with every other outlier.
        boundary = BATCH_BUCKET_SECONDS[-1]
        while duration_seconds > boundary:
            boundary *= 2
        return boundary

    def _infer_waveforms(self, waveforms, *, with_timestamps: bool):
        import torch

        direct = DIRECT_FORWARD and not with_timestamps
        if direct:
            self._ensure_serving_decoding()
        with torch.inference_mode(), _autocast():
            if direct:
                return self._infer_direct(waveforms)
            try:
                return self._model.transcribe(
                    waveforms,
                    batch_size=len(waveforms),
                    timestamps=with_timestamps,
                    verbose=False,
                    num_workers=0,
                )
            finally:
                _restore_eval_after_transcribe(self._model)

    def _infer_direct(self, waveforms):
        """Run NeMo's encoder and batched TDT decoder without a temporary DataLoader."""
        import torch

        timing = (
            _TIMING_STATE["serving"] and _TIMING_STATE["batches"] < TIMING_LOG_BATCHES
        )
        if timing:
            torch.cuda.synchronize()
            started_at = time.perf_counter()
        encoded, encoded_length, path = self._encode(waveforms)
        if _TIMING_STATE["serving"]:
            PARAKEET_METRICS.encoder_batches[path].inc()
        if timing:
            torch.cuda.synchronize()
            encoded_at = time.perf_counter()
        hypotheses = self._model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded,
            encoded_lengths=encoded_length,
            return_hypotheses=True,
        )
        if timing:
            torch.cuda.synchronize()
            _TIMING_STATE["batches"] += 1
            LOGGER.info(
                "Parakeet batch timing: "
                + _batch_timing_fields(waveforms, path, encoded.shape[-1])
                + ", "
                f"encoder_ms={(encoded_at - started_at) * 1000:.1f}, "
                f"decoder_ms={(time.perf_counter() - encoded_at) * 1000:.1f}"
            )
        if path == "graph" and self._graph_parity_due():
            self._log_graph_parity(waveforms, hypotheses)
        return hypotheses

    def _encode(self, waveforms):
        """Preprocessor + encoder for one batch: the captured graph when one fits,
        else eager on the batch padded to its longest clip. Returns
        (encoded, encoded_length, path) with path "graph" or "eager"."""
        if self._graphs is not None:
            graphed = self._graphs.run(waveforms)
            if graphed is not None:
                return (*graphed, "graph")
        encoded, encoded_length = self._eager_encode(waveforms)
        return encoded, encoded_length, "eager"

    def _eager_encode(self, waveforms):
        import torch

        signals, lengths = _stack_waveforms(
            waveforms, max(waveform.shape[0] for waveform in waveforms)
        )
        return self._model.forward(
            input_signal=torch.from_numpy(signals).to(device="cuda"),
            input_signal_length=torch.from_numpy(lengths).to(device="cuda"),
        )

    def _log_graph_parity(self, waveforms, graph_hypotheses):
        """GRAPH_PARITY_BATCHES debug aid: rerun the unpadded batch eagerly and log
        how many transcripts differ from the graphed ones. Logs counts, not text."""
        encoded, encoded_length = self._eager_encode(waveforms)
        eager_hypotheses = self._model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded,
            encoded_lengths=encoded_length,
            return_hypotheses=True,
        )
        parity = self._parity
        parity["batches"] += 1
        for graphed, eager in zip(graph_hypotheses, eager_hypotheses):
            graphed_text, eager_text = _hypothesis_text(graphed), _hypothesis_text(eager)
            parity["clips"] += 1
            parity["identical"] += graphed_text == eager_text
            parity["word_edits"] += _word_edits(eager_text, graphed_text)
            parity["words"] += len(eager_text.split())
        LOGGER.info(
            "Parakeet encoder graph parity: "
            f"batch={len(waveforms)}, "
            f"max_audio_s={max(w.shape[0] for w in waveforms) / AUDIO_SAMPLE_RATE_HZ:.2f}, "
            + ", ".join(f"{key}={value}" for key, value in parity.items())
        )

    def _check_transcribe_mode(self, model):
        """Run the timestamps path (NeMo transcribe()) once and fail if it leaves the
        model in training mode. STARTUP_MODE_CHECK=false skips it."""
        import numpy as np

        rng = np.random.default_rng(0)
        waveform = (0.05 * rng.standard_normal(2 * AUDIO_SAMPLE_RATE_HZ)).astype(np.float32)
        self._infer_waveforms([waveform], with_timestamps=True)
        modes = {
            name: bool(getattr(getattr(model, name), "training", False))
            for name in ("encoder", "decoder", "joint")
            if getattr(model, name, None) is not None
        }
        LOGGER.info(f"Parakeet startup mode check after transcribe(timestamps): training={modes}")
        if any(modes.values()):
            raise RuntimeError(f"model left in training mode after transcribe(): {modes}")

    def _check_startup_transcript(self, waveform):
        """Transcribe the check clip in the serving dtype; fail on a broken decode.

        Runs the preprocessor and encoder under the serving precision, requires a
        finite encoder output, then decodes with the same
        decoding.rnnt_decoder_predictions_tensor() the direct path uses (after
        warmup, so through the captured decoder CUDA graph) and fails startup if
        the WER against the known reference is above STARTUP_CHECK_MAX_WER.
        """
        import torch

        with torch.inference_mode(), _autocast():
            # The serving encode path: with ENCODER_CUDA_GRAPHS, a graph replay.
            encoded, encoded_length, encoder_path = self._encode([waveform])
            encoder_finite = bool(torch.isfinite(encoded).all())
            encoder_dtype = encoded.dtype
            encoder_absmax = float(encoded.float().abs().max())
            hypothesis = self._model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded,
                encoded_lengths=encoded_length,
                return_hypotheses=True,
            )[0]
            del encoded, encoded_length
        text = _hypothesis_text(hypothesis)
        wer = _word_error_rate(STARTUP_CHECK_REFERENCE, text)
        LOGGER.info(
            "Parakeet startup transcript: path=rnnt_decoder_predictions_tensor, "
            f"dtype={INFERENCE_DTYPE}, precision_mode={PRECISION_MODE}, "
            f"encoder_cudnn={ENCODER_CUDNN}, encoder_path={encoder_path}, "
            f"encoder_dtype={encoder_dtype}, "
            f"encoder_finite={encoder_finite}, encoder_absmax={encoder_absmax:.1f}, "
            f"wer={wer:.3f}, text={str(text)[:300]!r}"
        )
        if ENCODER_CUDA_GRAPHS and encoder_path != "graph":
            raise RuntimeError(
                "startup transcript check failed: ENCODER_CUDA_GRAPHS is on but the "
                f"check clip ran {encoder_path} "
                f"({self._graphs.disabled_reason if self._graphs else 'no graphs'})"
            )
        if not encoder_finite:
            raise RuntimeError(
                "startup transcript check failed: non-finite encoder output with "
                f"INFERENCE_DTYPE={INFERENCE_DTYPE}, PRECISION_MODE={PRECISION_MODE}"
            )
        if wer > STARTUP_CHECK_MAX_WER:
            raise RuntimeError(
                f"startup transcript check failed: WER {wer:.3f} > "
                f"{STARTUP_CHECK_MAX_WER:g} with INFERENCE_DTYPE={INFERENCE_DTYPE}, "
                f"PRECISION_MODE={PRECISION_MODE}"
            )

    def _warmup(self, *, decoder_only: bool = False):
        """Warm the serving path. decoder_only, after a decoding rebuild, runs only
        the passes that size the rebuilt decoder's CUDA graph (the encoder-graph
        sizing pass and the longest warmup clip): the encoder is already warm."""
        import numpy as np

        if self._graphs is not None:
            # Size NeMo's full-graph TDT decoder once for the largest batch and the
            # longest graphed length, so it does not re-capture under traffic for
            # any graphed batch (it re-captures when a batch or encoder length is
            # larger than any it has seen). As in stt/orukeet/batch-short.
            started_at = time.monotonic()
            batch = min(MAX_BATCH_SIZE, self._graphs.batch_sizes[-1])
            waveform = np.zeros(self._graphs.max_padded, dtype=np.float32)
            self._infer_waveforms([waveform] * batch, with_timestamps=False)
            LOGGER.info(
                f"Parakeet decoder sized for encoder graphs: batch={batch}, "
                f"audio_s={self._graphs.max_padded / AUDIO_SAMPLE_RATE_HZ:g}, "
                f"elapsed_s={time.monotonic() - started_at:.1f}"
            )
        if not WARMUP_AUDIO_SECONDS:
            return

        durations = (
            (max(WARMUP_AUDIO_SECONDS),) if decoder_only else WARMUP_AUDIO_SECONDS
        )
        for duration_seconds in durations:
            waveform = np.zeros(
                round(duration_seconds * AUDIO_SAMPLE_RATE_HZ), dtype=np.float32
            )
            self._infer_waveforms(
                [waveform for _ in range(WARMUP_BATCH_SIZE)],
                with_timestamps=False,
            )
        LOGGER.info(
            f"Parakeet warmup complete: durations={durations}, "
            f"batch_size={WARMUP_BATCH_SIZE}"
        )

    def _log_decoder_graph_mode(self, phase: str, *, enforce: bool):
        decoder = getattr(getattr(self._model, "decoding", None), "decoding", None)
        computer = getattr(decoder, "decoding_computer", None)
        mode = getattr(computer, "cuda_graphs_mode", None)
        mode_value = getattr(mode, "value", mode)
        observed_mode = mode_value or "unavailable"
        required_mode = "full_graph" if enforce else "any"
        if enforce and mode_value != "full_graph":
            LOGGER.error(
                "Parakeet TDT decoder CUDA graph check failed: "
                f"phase={phase}, observed={observed_mode}, required={required_mode}"
            )
            raise RuntimeError(
                "REQUIRE_FULL_CUDA_GRAPH is enabled, but NeMo selected "
                f"decoder CUDA graph mode {mode_value!r}"
            )
        LOGGER.info(
            "Parakeet TDT decoder CUDA graph check passed: "
            f"phase={phase}, observed={observed_mode}, required={required_mode}"
        )

    @staticmethod
    def _decode_waveform(audio_bytes: bytes):
        """Decode common 16 kHz formats in-process; pipe unusual inputs to ffmpeg."""
        import numpy as np
        import soundfile as sf

        try:
            waveform, sample_rate = sf.read(
                io.BytesIO(audio_bytes), dtype="float32", always_2d=True
            )
            waveform = waveform.mean(axis=1)
            if sample_rate == AUDIO_SAMPLE_RATE_HZ:
                if waveform.size == 0:
                    # A zero-frame WAV; same rejection as the ffmpeg path below.
                    raise ValueError("decoded audio has no samples: input has 0 frames")
                return np.ascontiguousarray(waveform, dtype=np.float32)
        except (RuntimeError, sf.LibsndfileError):
            pass

        return Model._decode_with_ffmpeg(audio_bytes)

    @staticmethod
    def _decode_with_ffmpeg(audio_bytes: bytes):
        """Decode or resample formats unsupported by the in-process fast path."""
        import numpy as np

        # Decode from a seekable temp file, not stdin. MP4/M4A files written
        # with the moov atom after the media data (the default for most
        # encoders, including every podcast-* bench clip) cannot be demuxed
        # from a pipe once they outgrow ffmpeg's probe buffer: ffmpeg logs
        # "partial file", exits 0, and emits no samples.
        with tempfile.NamedTemporaryFile(suffix=".audio") as input_file:
            input_file.write(audio_bytes)
            input_file.flush()
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    input_file.name,
                    "-ac",
                    "1",
                    "-ar",
                    str(AUDIO_SAMPLE_RATE_HZ),
                    "-f",
                    "f32le",
                    "pipe:1",
                ],
                capture_output=True,
            )
        stderr = proc.stderr.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise ValueError(
                f"could not decode audio: {stderr[-FFMPEG_ERROR_CONTEXT_CHARS:]}"
            )
        if not proc.stdout:
            # Fail this request alone with the decoder's message instead of
            # sending a zero-length waveform into a shared batch.
            raise ValueError(
                "decoded audio has no samples: "
                f"{stderr[-FFMPEG_ERROR_CONTEXT_CHARS:] or 'ffmpeg produced no output'}"
            )

        # f32le is already the model's desired waveform representation.
        # frombuffer avoids another decoder pass; copy detaches the array from
        # the temporary subprocess output bytes before they leave this scope.
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()

    def _decode_audio_input(self, request: dict) -> bytes:
        if "audio_b64" in request:
            return base64.b64decode(request["audio_b64"], validate=True)
        if "audio_url" in request:
            resp = self._http.get(request["audio_url"])
            resp.raise_for_status()
            return resp.content
        raise ValueError("provide audio_url or audio_b64")
