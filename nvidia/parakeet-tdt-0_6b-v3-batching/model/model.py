"""
Parakeet TDT 0.6B batching endpoint (NeMo PyTorch fp16 on T4).

Architecture:

  HTTP request → predict() → audio decode (CPU) → enqueue(_BatchRequest)
                                                          │
                                                          ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                    _batch_queue (FIFO)                            │
  └──────────────────────────────────────────────────────────────────┘
                       │                          │
            drain + adaptive wait        (blocks for first item)
                       │                          ▼
              ┌────────┴──────────────────────────────────┐
              │  Collector thread                         │
              │   - drain queue + adaptive collect window │
              │   - pad audio (CPU) to max batch length   │
              │   - run mel preprocessor on GPU           │
              │   - put → _ready_queue (maxsize=1)        │
              └────────┬──────────────────────────────────┘
                       │
              ┌────────▼──────────────────────────────────┐
              │  Inference thread                         │
              │   - take preprocessed batch               │
              │   - encoder + TDT decoder (with CUDA      │
              │     graphs for the per-step decoder loop) │
              │   - dispatch text → request Futures       │
              └───────────────────────────────────────────┘

Key design choices for performance on T4:

  1. Auto-detect dtype: fp16 on Turing (T4, sm75), bf16 on Ampere+
     (sm>=80). Hardcoding bf16 on T4 silently falls back to slow paths
     because T4 has no bf16 tensor cores.

  2. Decompose model.transcribe(): bypass NeMo's high-level wrapper
     (which has per-call freeze/unfreeze + mode switching overhead)
     and call preprocessor → encoder → rnnt_decoder_predictions_tensor
     directly.

  3. Pipelined two-thread design: a collector thread prepares the next
     batch (drain + pad + preprocessor) while an inference thread runs
     encoder + decoder on the previous batch. The single-slot
     `_ready_queue` between them keeps exactly one batch pre-staged.

  4. CUDA graphs ON for the TDT decoder: captures the per-token kernel
     sequence and replays without per-step Python overhead. Big win at
     small/medium batch sizes.

  5. Adaptive collect window: the collector drains the queue
     immediately (no fixed sleep), then waits up to BATCH_COLLECT_MS
     for stragglers ONLY if the initial drain produced ≥2 items. At
     low load (single arrivals) requests fire instantly; at high load
     the wait window grows batches above 1.

  6. Input audio is sorted by length within a batch: minor
     padding-efficiency win for variable-length clips.

Environment variables:
  BATCH_INFERENCE   "true" | "false"        (default: true)
  MAX_BATCH_SIZE    int                     (default: 64)
  BATCH_COLLECT_MS  float, ms               (default: 20)
  CUDA_GRAPHS       "true" | "false"        (default: true)
  TORCH_COMPILE     "true" | "false"        (default: false)
  COMPILE_MODE      reduce-overhead|default (default: reduce-overhead)
"""

import base64
import io
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from concurrent.futures import Future

import numpy as np
import requests
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = "/app/model_cache/parakeet-tdt-0.6b-v3"
MODEL_NEMO_FILE = "parakeet-tdt-0.6b-v3.nemo"
SAMPLE_RATE = 16000


# ── Helpers ───────────────────────────────────────────────────────────


def json_serialize_recursive(obj):
    if isinstance(obj, dict):
        return {k: json_serialize_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize_recursive(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(json_serialize_recursive(v) for v in obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif hasattr(obj, "__dict__"):
        return json_serialize_recursive(obj.__dict__)
    else:
        return str(obj)


def download_and_decode_audio(audio_url: str) -> np.ndarray:
    """Fetch audio over HTTP and decode to 16kHz mono float32 numpy array."""
    response = requests.get(audio_url, timeout=60)
    response.raise_for_status()
    audio_bytes = response.content

    try:
        audio_data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        if sr == SAMPLE_RATE:
            return audio_data
    except Exception:
        sr = None

    # Fall back to ffmpeg for codecs soundfile can't handle (mp3, opus, etc.)
    url_clean = audio_url.split("?")[0].lower()
    ext = os.path.splitext(url_clean)[1] or ".audio"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        result = subprocess.run(
            [
                "ffmpeg", "-i", tmp.name, "-ar", str(SAMPLE_RATE), "-ac", "1",
                "-f", "f32le", "-loglevel", "error", "-",
            ],
            capture_output=True, check=True,
        )
        return np.frombuffer(result.stdout, dtype=np.float32)


def decode_base64_audio(audio_b64: str) -> np.ndarray:
    """Decode base64-encoded audio bytes to 16kHz mono float32 numpy array."""
    raw = base64.b64decode(audio_b64)
    audio_data, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    if sr != SAMPLE_RATE:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(raw)
            tmp.flush()
            result = subprocess.run(
                [
                    "ffmpeg", "-i", tmp.name, "-ar", str(SAMPLE_RATE), "-ac", "1",
                    "-f", "f32le", "-loglevel", "error", "-",
                ],
                capture_output=True, check=True,
            )
            return np.frombuffer(result.stdout, dtype=np.float32)
    return audio_data


def _detect_dtype() -> torch.dtype:
    """bfloat16 on Ampere+ (sm>=80); float16 on Turing/T4 (sm75); fp32 on CPU.

    T4 has no bf16 tensor cores — bf16 there silently falls back to
    slow non-tensor-core paths. fp16 is the right choice on T4.
    """
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


# ── Batch request ─────────────────────────────────────────────────────


class _BatchRequest:
    __slots__ = ("audio_array", "future")

    def __init__(self, audio_array: np.ndarray):
        self.audio_array = audio_array
        self.future: Future = Future()


# ── Model loading ─────────────────────────────────────────────────────


def _load_nemo_model(
    model_path: str, dtype: torch.dtype, use_compile: bool,
    compile_mode: str, enable_cuda_graphs: bool = True,
):
    """Load a NeMo .nemo file, configure it for fast inference, and warm up.

    Configuration:
      - CUDA graphs for the TDT decoder (enabled by default; captures the
        per-token decoder kernel sequence so it replays without per-step
        Python overhead).
      - Local windowed self-attention (window=256) instead of full
        attention. Lower memory + faster on long audio without quality
        cost on typical short-utterance traffic.
      - Subsampling conv chunking factor = 1 (process the whole input
        in one chunk; chunking>1 is for OOM mitigation on very long
        audio that we don't need here).
      - Cast weights to the target dtype (fp16 on T4, bf16 on Ampere+).
    """
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.restore_from(restore_path=model_path)

    if not enable_cuda_graphs:
        model.decoding.decoding.decoding_computer.disable_cuda_graphs()
        logger.info("CUDA graphs DISABLED for TDT decoder")
    else:
        logger.info("CUDA graphs ENABLED for TDT decoder")

    model.change_attention_model("rel_pos_local_attn", [256, 256])
    model.change_subsampling_conv_chunking_factor(1)
    model.to(dtype)
    model.eval()

    if use_compile:
        try:
            model.encoder = torch.compile(model.encoder, mode=compile_mode)
        except Exception as exc:
            logger.warning("torch.compile failed, using eager: %s", exc)

    # Warm up so the first real request doesn't pay JIT / kernel-autotune cost.
    dummy = np.random.randn(SAMPLE_RATE * 5).astype(np.float32)
    with torch.inference_mode():
        model.transcribe([dummy], timestamps=False)

    return model


# ── Model class ───────────────────────────────────────────────────────


class Model:
    def __init__(self, lazy_data_resolver, **kwargs) -> None:
        self._lazy_data_resolver = lazy_data_resolver
        self._hf_access_token = kwargs["secrets"].get("hf_access_token")
        self._transcribe_lock = threading.Lock()

        self._batch_enabled = os.getenv("BATCH_INFERENCE", "true").lower() == "true"
        self._max_batch_size = int(os.getenv("MAX_BATCH_SIZE", "64"))
        self._batch_collect_s = float(os.getenv("BATCH_COLLECT_MS", "20")) / 1000.0
        self._use_compile = os.getenv("TORCH_COMPILE", "false").lower() == "true"
        self._compile_mode = os.getenv("COMPILE_MODE", "reduce-overhead")
        self._cuda_graphs = os.getenv("CUDA_GRAPHS", "true").lower() == "true"

        self.model = None
        self._device = None
        # Inbound request queue (one entry per HTTP request).
        self._batch_queue: queue.Queue[_BatchRequest] = queue.Queue()
        # Single-slot handoff between collector and inference threads.
        # maxsize=1 ensures the collector pre-stages exactly one batch
        # ahead — enough to overlap CPU prep with previous-batch GPU
        # work, but no more (extra slots would just inflate latency).
        self._ready_queue: queue.Queue = queue.Queue(maxsize=1)

    # ── Load ──────────────────────────────────────────────────────────

    def load(self):
        self._lazy_data_resolver.block_until_download_complete()
        model_path = os.path.join(MODEL_CACHE_DIR, MODEL_NEMO_FILE)

        dtype = _detect_dtype()
        gpu = (torch.cuda.get_device_name()
               if torch.cuda.is_available() else "cpu")
        logger.info(
            "Config: max_batch=%d  collect_ms=%.0f  dtype=%s  "
            "compile=%s  cuda_graphs=%s  gpu=%s",
            self._max_batch_size, self._batch_collect_s * 1000, dtype,
            self._use_compile, self._cuda_graphs, gpu,
        )

        t0 = time.time()
        self.model = _load_nemo_model(
            model_path, dtype, self._use_compile, self._compile_mode,
            enable_cuda_graphs=self._cuda_graphs,
        )
        self._device = next(self.model.parameters()).device
        logger.info("Model loaded + warmed up in %.1fs", time.time() - t0)

        if self._batch_enabled:
            threading.Thread(
                target=self._batch_collector, daemon=True, name="collector",
            ).start()
            threading.Thread(
                target=self._batch_inference, daemon=True, name="inference",
            ).start()
            logger.info("Pipelined batch worker started (collector + inference)")

        logger.info("Load complete — ready to serve")

    # ── Pipelined batch worker ───────────────────────────────────────

    def _drain_batch(self) -> list[_BatchRequest]:
        """Collect a batch: block for first item, drain queue, adaptive wait.

        Adaptive collect window: only wait for stragglers if the initial
        drain already produced 2+ items (evidence of concurrent traffic).
        When a request arrives alone, fire it immediately to preserve
        tail latency at low load. At medium-to-high load the queue is
        rarely empty after the first drain, so the wait window kicks in
        and grows the batch above 1.
        """
        first = self._batch_queue.get()
        batch = [first]

        while len(batch) < self._max_batch_size:
            try:
                batch.append(self._batch_queue.get_nowait())
            except queue.Empty:
                break

        if (
            len(batch) >= 2
            and len(batch) < self._max_batch_size
            and self._batch_collect_s > 0
        ):
            deadline = time.monotonic() + self._batch_collect_s
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._batch_queue.get(timeout=remaining))
                except queue.Empty:
                    break

        # Sort by audio length so padding within the batch is minimized
        # (the longest item drives the padded length anyway, but keeping
        # similar lengths together improves cache locality slightly).
        batch.sort(key=lambda r: len(r.audio_array))
        return batch

    def _batch_collector(self):
        """Thread 1: collect batches, pad audio, run mel preprocessor.

        Runs concurrently with _batch_inference. While the GPU is busy
        with batch N's encoder + decoder, this thread builds batch N+1:
          - drain inbound queue (with adaptive collect window)
          - pad audio arrays to the batch's max length
          - move to GPU
          - run NeMo's mel preprocessor (on the default CUDA stream)
          - hand off to _ready_queue for the inference thread
        """
        while True:
            batch = []
            try:
                batch = self._drain_batch()
                audios = [r.audio_array for r in batch]

                max_len = max(len(a) for a in audios)
                padded = np.zeros((len(audios), max_len), dtype=np.float32)
                lengths = np.zeros(len(audios), dtype=np.int64)
                for i, a in enumerate(audios):
                    padded[i, :len(a)] = a
                    lengths[i] = len(a)

                input_signal = torch.from_numpy(padded).to(
                    device=self._device, dtype=torch.float32,
                )
                input_lengths = torch.from_numpy(lengths).to(device=self._device)

                with torch.inference_mode():
                    processed, proc_len = self.model.preprocessor(
                        input_signal=input_signal, length=input_lengths,
                    )

                self._ready_queue.put((processed, proc_len, batch))

            except Exception as exc:
                logger.error("Collector error: %s", exc)
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(exc)

    def _batch_inference(self):
        """Thread 2: run encoder + TDT decoder on preprocessed batches.

        Decomposed pipeline: we call NeMo's preprocessor / encoder /
        decoder modules directly instead of going through the high-level
        `model.transcribe()` API, which has per-call setup/teardown
        overhead (freeze/unfreeze, mode switching, hypothesis post-
        processing) that adds up at high request rates.

        The TDT decoder's internal token loop calls .item() per step,
        which implicitly synchronizes with the GPU — so by the time
        rnnt_decoder_predictions_tensor returns, all GPU work for this
        batch is complete and we can safely dispatch text to futures.
        """
        while True:
            batch = []
            try:
                processed, proc_len, batch = self._ready_queue.get()

                with torch.inference_mode():
                    encoded, encoded_len = self.model.encoder(
                        audio_signal=processed, length=proc_len,
                    )
                    dec_result = self.model.decoding.rnnt_decoder_predictions_tensor(
                        encoder_output=encoded,
                        encoded_lengths=encoded_len,
                        return_hypotheses=False,
                    )

                raw_hyps = (
                    dec_result[0] if isinstance(dec_result, tuple)
                    else dec_result
                )
                for req, hyp in zip(batch, raw_hyps):
                    text = hyp.text if hasattr(hyp, "text") else str(hyp)
                    req.future.set_result(text)

            except Exception as exc:
                logger.error("Inference error: %s", exc)
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(exc)

    # ── Predict ───────────────────────────────────────────────────────

    def predict(self, request: dict):
        """HTTP entrypoint. Accepts:
            audio_url:    URL to fetch and decode
            audio_base64: base64-encoded audio bytes (any soundfile/ffmpeg-readable)
            timestamps:   bool — if true, returns word-level timestamps
                          (uses serial path; not batched).
        """
        audio_url = request.get("audio_url")
        audio_b64 = request.get("audio_base64")
        is_timestamps = request.get("timestamps", False)

        try:
            if audio_b64:
                audio_array = decode_base64_audio(audio_b64)
            else:
                audio_array = download_and_decode_audio(audio_url)

            if self._batch_enabled and not is_timestamps:
                return self._predict_batched(audio_array)
            return self._predict_serial(audio_array, is_timestamps)

        except Exception as exc:
            logger.error("Predict error: %s", exc)
            return {"error": str(exc)}

    def _predict_batched(self, audio_array: np.ndarray) -> dict:
        """Batched path (no timestamps): enqueue and wait for batch worker."""
        req = _BatchRequest(audio_array)
        self._batch_queue.put(req)
        text = req.future.result(timeout=120)
        return {"transcript": text}

    def _predict_serial(self, audio_array: np.ndarray, is_timestamps: bool) -> dict:
        """Serial path: used when timestamps are requested (the batched
        decoder doesn't produce per-token timestamps in this code path).
        Single-threaded under self._transcribe_lock so concurrent serial
        requests don't trip on NeMo's shared transcribe() state.
        """
        with self._transcribe_lock:
            with torch.inference_mode():
                results = self.model.transcribe(
                    [audio_array], timestamps=is_timestamps,
                )
        hyp = results[0]
        if isinstance(hyp, (list, tuple)):
            hyp = hyp[0]

        if not is_timestamps:
            text = hyp.text if hasattr(hyp, "text") else str(hyp)
            return {"transcript": text}
        return {
            "transcript": json_serialize_recursive({
                "text": hyp.text,
                "score": hyp.score,
                "timestep": hyp.timestep,
            })
        }
