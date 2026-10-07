"""Custom Python Truss model for oruk/orukeet (STT).

Orukeet is a 25-language finetune of nvidia/parakeet-tdt-0.6b-v3 with half of the
encoder's temporal depthwise filters replaced by fitted, frozen Gabor kernels. The
kernels materialize as ordinary convolution weights in the checkpoint, so this is
the same FastConformer-TDT transducer served through NVIDIA NeMo — no vLLM/SGLang
implementation exists for this architecture, so this follows the registry's
custom `model/model.py` path (b10-bench protocol: `baseten_predict`).

Contract:
- in:  {"audio_url": "https://..."} or {"audio_b64": "..."}
       optional: {"timestamps": true}
- out: {"transcript": "...", "text": "...", ["timestamps": {"word": [...], "segment": [...]}]}

Audio up to FULL_ATTENTION_MAX_SECONDS (24 min) takes the original path: one
full-attention encoder pass per clip on the original model, microbatched by duration
bucket, with the batch size capped so the summed attention memory never exceeds one
maximum-length clip. Longer audio, up to MAX_AUDIO_SECONDS (3 hr), runs one clip at a
time on a second copy of the checkpoint that load() switches to local attention
(`change_attention_model("rel_pos_local_attn", [256, 256])`, the parakeet-tdt-0.6b-v3
model card's long-form recipe). The two models never change mode after load(), so no
request can run under the wrong attention, and the short path is left as it was. With
MAX_AUDIO_SECONDS equal to FULL_ATTENTION_MAX_SECONDS, load() skips the second copy.
"""

import base64
import collections
import contextlib
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
import traceback

from prometheus_client import Gauge, Histogram

LOGGER = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/orukeet")
NEMO_CHECKPOINT = os.environ.get(
    "NEMO_CHECKPOINT", os.path.join(MODEL_DIR, "orukeet-v0.1.0.nemo")
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
REQUIRE_FULL_CUDA_GRAPH = _env_flag("REQUIRE_FULL_CUDA_GRAPH")
WARMUP_AUDIO_SECONDS = _env_float_list("WARMUP_AUDIO_SECONDS")
WARMUP_BATCH_SIZE = int(os.environ.get("WARMUP_BATCH_SIZE", "1"))
FREEZE_GC_AFTER_WARMUP = _env_flag("FREEZE_GC_AFTER_WARMUP")

if WARMUP_BATCH_SIZE < 1:
    raise ValueError("WARMUP_BATCH_SIZE must be at least 1")

# Hard input limits. Requests above them fail alone with a 413 before any GPU work.
# The duration cap is the "up to 3 hrs" local-attention bound from the parakeet-tdt-0.6b-v3
# model card; with ffmpeg's -t it also stops a decoder producing more samples than that.
MAX_AUDIO_SECONDS = float(os.environ.get("MAX_AUDIO_SECONDS", "10800"))
MAX_AUDIO_INPUT_BYTES = int(
    os.environ.get("MAX_AUDIO_INPUT_BYTES", str(1024 * 1024 * 1024))
)

# Clips up to FULL_ATTENTION_MAX_SECONDS run on the original full-attention model
# (the model card's "up to 24 minutes long with full attention"). Longer clips run
# on a second copy of the checkpoint switched to local attention, as the model card
# and NeMo's "Inference on long audio" guide prescribe. Full attention costs
# 8 heads x T^2 x 4 bytes per score tensor (NeMo computes attention in fp32), with T
# = seconds / 0.08 encoder frames, so it is quadratic in length; local attention
# with a fixed window is linear.
FULL_ATTENTION_MAX_SECONDS = float(os.environ.get("FULL_ATTENTION_MAX_SECONDS", "1440"))
LOCAL_ATTENTION_CONTEXT_SIZE = tuple(
    int(item) for item in os.environ.get("LOCAL_ATTENTION_CONTEXT_SIZE", "256,256").split(",")
)
# 1 = NeMo's auto mode: split the conv subsampling input only when it would exceed
# the 2**31-element indexing limit, which multi-hour inputs do.
SUBSAMPLING_CONV_CHUNKING_FACTOR = int(
    os.environ.get("SUBSAMPLING_CONV_CHUNKING_FACTOR", "1")
)
# At load, run one full-attention encoder pass at FULL_ATTENTION_MAX_SECONDS and one
# local-attention transcription at MAX_AUDIO_SECONDS, so a deployment that cannot
# hold its own limits fails at startup instead of on a customer request.
LONG_AUDIO_STARTUP_CHECK = _env_flag("LONG_AUDIO_STARTUP_CHECK", True)
LONG_AUDIO_BUCKET = "local_attention"

if MAX_AUDIO_SECONDS <= 0:
    raise ValueError("MAX_AUDIO_SECONDS must be positive")
if MAX_AUDIO_INPUT_BYTES <= 0:
    raise ValueError("MAX_AUDIO_INPUT_BYTES must be positive")
if not 0 < FULL_ATTENTION_MAX_SECONDS <= MAX_AUDIO_SECONDS:
    raise ValueError("FULL_ATTENTION_MAX_SECONDS must be in (0, MAX_AUDIO_SECONDS]")
if len(LOCAL_ATTENTION_CONTEXT_SIZE) != 2 or min(LOCAL_ATTENTION_CONTEXT_SIZE) <= 0:
    raise ValueError("LOCAL_ATTENTION_CONTEXT_SIZE must be two positive ints, e.g. 256,256")
# With MAX_AUDIO_SECONDS equal to FULL_ATTENTION_MAX_SECONDS every longer clip gets a
# 413 before routing, so no request can reach the local-attention path. load() then
# skips the second checkpoint copy and its ~2.5 GB of GPU memory.
LONG_AUDIO_ENABLED = MAX_AUDIO_SECONDS > FULL_ATTENTION_MAX_SECONDS

# cuDNN is off on both paths by default. On RTX PRO 6000 (sm_120) with torch 2.8 /
# cuDNN 9.10, cuDNN spends 0.4-0.7 s in the encoder the first time it sees each new
# input length, and real audio almost never repeats a length. Measured at startup on
# deployment 3m4pr0o (first pass of an unseen length / repeat, encoder only):
#   short, 10-13 s clips:     cuDNN on 688-696 / 27-29 ms, off 27-28 / 27 ms
#   long, 25 min local attn:  cuDNN on 734-1099 / 349-361 ms, off 340-346 / 338-345 ms
# PyTorch's own kernels have no per-length setup and were as fast or faster when warm.
SHORT_PATH_CUDNN = _env_flag("SHORT_PATH_CUDNN", False)
LONG_PATH_CUDNN = _env_flag("LONG_PATH_CUDNN", False)

# Diagnostics, off by default. STAGE_PROBE times the preprocessor, encoder and
# decoder on unseen lengths at startup, with cuDNN on and off, for both models.
# TIMING_LOG_BATCHES / TIMING_LOG_REQUESTS log encoder/decoder time for the first N
# batches and phase times for the first N requests after load.
STAGE_PROBE = _env_flag("STAGE_PROBE", False)
TIMING_LOG_BATCHES = int(os.environ.get("TIMING_LOG_BATCHES", "0"))
TIMING_LOG_REQUESTS = int(os.environ.get("TIMING_LOG_REQUESTS", "0"))
_TIMING_STATE = {"serving": False, "batches": 0, "requests": 0}

# Substrings of CUDA errors that leave the context unusable for every later call.
STICKY_CUDA_ERROR_MARKERS = (
    "illegal memory access",
    "device-side assert",
    "unspecified launch failure",
    "misaligned address",
    "illegal instruction",
    "CUBLAS_STATUS_EXECUTION_FAILED",
    "uncorrectable ECC error",
)


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

    def observe_latency(self, phase: str, duration_seconds: float):
        self._request_latency_by_phase[phase].observe(duration_seconds)


# Prometheus uses a process-global registry, so construct each metric once.
PARAKEET_METRICS = ParakeetMetrics()


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


class RequestTooLargeError(ValueError):
    """Input above the deployment's size or duration limit."""


class InferenceError(RuntimeError):
    """A request-scoped inference failure that carries no GPU-tensor frames."""


def _client_error(status_code: int, detail: str) -> Exception:
    """Build an error Truss returns with `status_code` instead of a generic 500."""
    try:
        from fastapi import HTTPException
    except ImportError:  # unit tests without the server stack
        return RequestTooLargeError(detail)
    return HTTPException(status_code=status_code, detail=detail)


@contextlib.contextmanager
def _cudnn_enabled(enabled: bool):
    """Set torch.backends.cudnn.enabled for one block. All GPU work runs on one
    thread at a time (load(), then the batch worker), so the global flag is safe."""
    import torch

    previous = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = enabled
    try:
        yield
    finally:
        torch.backends.cudnn.enabled = previous


def _clear_exception_frames(exc: BaseException):
    """Drop the locals of finished frames an exception chain keeps alive.

    A CUDA OOM traceback holds the encoder activations of the failed pass (8.62 GiB
    allocated at the logged 1 hr OOM) and the padded host batch. Stored and re-raised
    to every request, as the previous batch worker did, it pinned that memory until a
    full GC pass: the next requests on that replica saw 14.19 GiB still allocated.
    """
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)
            current.__traceback__ = None
        current = current.__cause__ or current.__context__


class Model:
    def __init__(self, **kwargs):
        self._model = None
        self._long_model = None
        self._jobs = queue.Queue()
        self._deferred_jobs = collections.deque()
        self._http = None
        self._worker = None
        self._unhealthy_reason = None

    def load(self):
        import httpx
        import nemo
        import torch
        from omegaconf import open_dict

        self._model = self._restore_model()
        LOGGER.info(
            f"NeMo checkpoint restore succeeded: checkpoint={NEMO_CHECKPOINT}, "
            f"nemo_version={getattr(nemo, '__version__', 'unknown')}, "
            f"torch_version={torch.__version__}, torch_cuda={torch.version.cuda}, "
            f"cudnn_version={torch.backends.cudnn.version()}, "
            f"gpu={torch.cuda.get_device_name()}, "
            f"compute_capability={torch.cuda.get_device_capability()}, "
            f"gpu_memory_gib={torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}, "
            f"model_class={type(self._model).__name__}, "
            f"encoder_class={type(getattr(self._model, 'encoder', None)).__name__}, "
            f"decoder_class={type(getattr(self._model, 'decoder', None)).__name__}"
        )

        if LONG_AUDIO_ENABLED:
            # A second, independent copy of the checkpoint serves audio above the
            # full-attention limit. It is switched to local attention once, here, and
            # never switched back: change_attention_model() replaces every encoder
            # layer's attention module in place, so toggling one shared model per
            # request would race the short-clip path and could run a request under the
            # wrong attention. Two copies cost ~2.5 GB of GPU memory instead.
            self._long_model = self._restore_model()
            self._long_model.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=list(LOCAL_ATTENTION_CONTEXT_SIZE),
            )
            self._long_model.change_subsampling_conv_chunking_factor(
                SUBSAMPLING_CONV_CHUNKING_FACTOR
            )
            # The long-form model's TDT decoder runs without CUDA graphs. With two
            # models each capturing a full_graph decoder, the first real request on
            # RTX PRO 6000 / torch 2.8 (deployment qzr785k) hit "illegal memory access"
            # replaying the short model's graph after the long model had captured its
            # own; stt/nemotron-3-diarized-transcription documents the same fault on this
            # GPU when a second graph is captured after NeMo's full_graph decoder. The
            # setting lives in cfg.decoding, so the rebuild transcribe(timestamps=...)
            # does keeps it. Decoder cost is minor next to a multi-hour encoder pass.
            with open_dict(self._long_model.cfg.decoding):
                self._long_model.cfg.decoding.greedy.use_cuda_graph_decoder = False
            self._long_model.change_decoding_strategy(
                self._long_model.cfg.decoding, verbose=False
            )
            LOGGER.info(
                "Parakeet long-form model ready: "
                f"self_attention_model={self._long_model.encoder.self_attention_model}, "
                f"att_context_size={self._long_model.encoder.att_context_size}, "
                f"subsampling_conv_chunking_factor={SUBSAMPLING_CONV_CHUNKING_FACTOR}; "
                f"short-clip model keeps self_attention_model={self._model.encoder.self_attention_model}"
            )
        else:
            LOGGER.info(
                "Parakeet long-form model disabled: "
                f"max_audio_seconds={MAX_AUDIO_SECONDS:g} equals "
                f"full_attention_max_seconds={FULL_ATTENTION_MAX_SECONDS:g}, so every "
                "accepted clip runs on the full-attention model and the second "
                "checkpoint copy is not loaded"
            )
        self._http = httpx.Client(timeout=60, follow_redirects=False)

        LOGGER.info(
            f"Parakeet runtime config: direct_forward={DIRECT_FORWARD}, "
            f"max_batch_size={MAX_BATCH_SIZE}, "
            f"batch_window_ms={BATCH_WINDOW_SECONDS * 1000:g}, "
            f"batch_buckets_seconds={BATCH_BUCKET_SECONDS or 'disabled'}, "
            f"full_attention_max_seconds={FULL_ATTENTION_MAX_SECONDS:g}, "
            f"max_audio_seconds={MAX_AUDIO_SECONDS:g}, "
            f"long_audio_enabled={LONG_AUDIO_ENABLED}, "
            f"max_audio_input_bytes={MAX_AUDIO_INPUT_BYTES}, "
            f"short_path_cudnn={SHORT_PATH_CUDNN}, long_path_cudnn={LONG_PATH_CUDNN}"
        )
        self._log_decoder_graph_mode("load", enforce=False)
        # Capacity checks must run before the short-clip warmup captures the
        # short model's decoder graph: running them after it (deployments qzr785k
        # and 3m4pr9k) made the next use of that graph fault with "illegal memory
        # access" on RTX PRO 6000.
        if LONG_AUDIO_STARTUP_CHECK:
            self._check_long_audio_capacity()
        if self._long_model is not None:
            if STAGE_PROBE:
                # Before the short warmup, for the same load-order reason as above.
                self._probe_stages(
                    self._long_model,
                    "long",
                    ((True, (1500.37, 1512.91)), (False, (1506.53, 1519.29))),
                )
            self._log_decoder_graph_mode(
                "long_form", enforce=False, model=self._long_model
            )
        self._warmup()
        self._log_decoder_graph_mode("warmup", enforce=REQUIRE_FULL_CUDA_GRAPH)
        if STAGE_PROBE:
            self._probe_stages(
                self._model,
                "short",
                ((True, (10.01, 11.37, 12.73)), (False, (10.53, 11.91, 13.29))),
            )
            self._compare_cudnn_transcripts()
        if FREEZE_GC_AFTER_WARMUP:
            gc.collect()
            gc.freeze()
            LOGGER.info("Python GC startup state frozen")

        _TIMING_STATE["serving"] = True
        self._worker = threading.Thread(target=self._batch_worker, daemon=True)
        self._worker.start()

    @staticmethod
    def _restore_model():
        import nemo.collections.asr as nemo_asr
        import torch

        LOGGER.info(f"Restoring NeMo checkpoint: {NEMO_CHECKPOINT}")
        model = nemo_asr.models.ASRModel.restore_from(
            NEMO_CHECKPOINT, map_location=torch.device("cuda")
        )
        model.eval()
        if DIRECT_FORWARD:
            # Match TranscriptionMixin._transcribe_on_begin() once at startup.
            # The direct path intentionally bypasses that per-call setup.
            preprocessor = getattr(model, "preprocessor", None)
            featurizer = getattr(preprocessor, "featurizer", None)
            if featurizer is not None:
                if hasattr(featurizer, "dither"):
                    featurizer.dither = 0.0
                if hasattr(featurizer, "pad_to"):
                    featurizer.pad_to = 0
        return model

    def _check_long_audio_capacity(self):
        """Prove at startup that each loaded attention mode fits its configured limit.

        The full-attention check always runs; the local-attention check runs only when
        the long-form model is loaded (LONG_AUDIO_ENABLED). The full-attention check
        runs the encoder only: the TDT decoder's CUDA-graph
        state is sized by the largest input it has seen, and the short-clip warmup
        shapes must stay the ones it serves. The local-attention check runs a full
        transcription on the long-form model, whose decoder is its own and runs
        without CUDA graphs.
        """
        import numpy as np
        import torch

        checks = [("full_attention", self._model, FULL_ATTENTION_MAX_SECONDS, False)]
        if self._long_model is not None:
            checks.append(
                ("local_attention", self._long_model, MAX_AUDIO_SECONDS, True)
            )
        total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        for mode, model, seconds, decode in checks:
            waveform = np.zeros(round(seconds * AUDIO_SAMPLE_RATE_HZ), dtype=np.float32)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started_at = time.monotonic()
            try:
                if decode:
                    self._infer_waveforms(model, [waveform], with_timestamps=False)
                else:
                    with (
                        torch.inference_mode(),
                        torch.autocast(device_type="cuda", dtype=torch.float16),
                        _cudnn_enabled(self._cudnn_for(model)),
                    ):
                        signal = torch.from_numpy(waveform).to(device="cuda")[None]
                        length = torch.tensor([waveform.shape[0]], device="cuda")
                        encoded, _ = model.forward(
                            input_signal=signal, input_signal_length=length
                        )
                        del signal, length, encoded
                torch.cuda.synchronize()
            except Exception:
                LOGGER.error(
                    "Parakeet long-audio capacity check failed: "
                    f"mode={mode}, audio_seconds={seconds:g}, gpu_memory_gib={total_gib:.2f}. "
                    "Lower FULL_ATTENTION_MAX_SECONDS / MAX_AUDIO_SECONDS or use a larger GPU."
                )
                raise
            finally:
                del waveform
            LOGGER.info(
                "Parakeet long-audio capacity check passed: "
                f"mode={mode}, audio_seconds={seconds:g}, "
                f"peak_allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.2f}, "
                f"peak_reserved_gib={torch.cuda.max_memory_reserved() / 2**30:.2f}, "
                f"gpu_memory_gib={total_gib:.2f}, "
                f"elapsed_s={time.monotonic() - started_at:.2f}"
            )
            torch.cuda.empty_cache()

    def is_healthy(self) -> bool:
        """Truss readiness/liveness hook, polled every 10 s after load() returns.

        It reports unhealthy only for faults a restart fixes: a CUDA context that
        can no longer run kernels, or a dead GPU worker thread. Request-level
        failures (bad audio, a single OOM) never flip it.
        """
        if self._unhealthy_reason is not None:
            LOGGER.error(f"Parakeet health check failed: {self._unhealthy_reason}")
            return False
        if self._worker is not None and not self._worker.is_alive():
            LOGGER.error("Parakeet health check failed: GPU batch worker exited")
            return False
        return True

    def predict(self, request: dict) -> dict:
        # Wrap predict so tracebacks reach the deployment logs.
        try:
            return self._predict_impl(request)
        except Exception as exc:
            if getattr(exc, "status_code", 500) < 500:
                LOGGER.warning(
                    f"Parakeet request rejected: {getattr(exc, 'detail', exc)}"
                )
            else:
                LOGGER.exception("Parakeet prediction failed")
            raise

    def _predict_impl(self, request: dict) -> dict:
        request_started_at = time.monotonic()
        preprocessing_started_at = request_started_at
        try:
            if self._unhealthy_reason is not None:
                raise _client_error(
                    503, f"replica is restarting: {self._unhealthy_reason}"
                )
            try:
                audio_bytes = self._decode_audio_input(request)
                with_timestamps = bool(request.get("timestamps", False))
                waveform = self._decode_waveform(audio_bytes)
                del audio_bytes
                duration_seconds = waveform.shape[0] / AUDIO_SAMPLE_RATE_HZ
                if duration_seconds > MAX_AUDIO_SECONDS:
                    raise _client_error(
                        413,
                        f"audio is {duration_seconds:.0f} s or longer; this deployment "
                        f"accepts up to {MAX_AUDIO_SECONDS:g} s (MAX_AUDIO_SECONDS)",
                    )
                job = {
                    "waveform": waveform,
                    "bucket": self._duration_bucket(duration_seconds),
                    "timestamps": with_timestamps,
                    "done": threading.Event(),
                }
                # The job holds the only reference now; the worker drops it as soon
                # as the batch finishes.
                del waveform
            finally:
                PARAKEET_METRICS.observe_latency(
                    "preprocessing", time.monotonic() - preprocessing_started_at
                )

            job["enqueued_at"] = time.monotonic()
            PARAKEET_METRICS.queue_depth.inc()
            self._jobs.put(job)
            if not job["done"].wait(timeout=PREDICT_TIMEOUT_SECONDS):
                job["cancelled"] = True
                raise TimeoutError(
                    "Parakeet batch worker did not finish within "
                    f"{PREDICT_TIMEOUT_SECONDS:g} seconds"
                )

            postprocessing_started_at = job["inference_finished_at"]
            try:
                if "error" in job:
                    # Pop instead of binding a local, so the raised error is not
                    # reachable from this frame through its own traceback.
                    raise job.pop("error")

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
            finished_at = time.monotonic()
            PARAKEET_METRICS.observe_latency("total", finished_at - request_started_at)
            if _TIMING_STATE["requests"] < TIMING_LOG_REQUESTS:
                _TIMING_STATE["requests"] += 1
                job_times = locals().get("job") or {}
                LOGGER.info(
                    "Parakeet request timing: "
                    + ", ".join(
                        f"{name}_ms={(end - start) * 1000:.1f}"
                        for name, start, end in (
                            ("preprocess", request_started_at, job_times.get("enqueued_at")),
                            ("queue", job_times.get("enqueued_at"), job_times.get("selected_at")),
                            ("batch_and_infer", job_times.get("selected_at"), job_times.get("inference_finished_at")),
                            ("total", request_started_at, finished_at),
                        )
                        if start is not None and end is not None
                    )
                )

    def _batch_worker(self):
        """Run all GPU work on one thread and opportunistically microbatch."""
        while True:
            first = self._next_job()
            jobs = [first]
            deadline = time.monotonic() + BATCH_WINDOW_SECONDS
            max_batch_size = self._max_batch_size(first["bucket"])

            # First scan jobs deferred by earlier, incompatible batches. This
            # keeps the queue work-conserving without mixing very different
            # audio lengths (which would pad every item to the longest clip).
            for _ in range(len(self._deferred_jobs)):
                if len(jobs) >= max_batch_size:
                    break
                candidate = self._deferred_jobs.popleft()
                if self._jobs_compatible(first, candidate):
                    jobs.append(self._select_job(candidate))
                else:
                    self._deferred_jobs.append(candidate)

            while len(jobs) < max_batch_size:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    candidate = self._jobs.get(timeout=timeout)
                except queue.Empty:
                    break
                if self._jobs_compatible(first, candidate):
                    jobs.append(self._select_job(candidate))
                else:
                    self._deferred_jobs.append(candidate)

            inference_started_at = time.monotonic()
            for job in jobs:
                PARAKEET_METRICS.observe_latency(
                    "batching", inference_started_at - job["selected_at"]
                )
            live_jobs = [job for job in jobs if not job.get("cancelled")]
            if live_jobs:
                PARAKEET_METRICS.batch_size.observe(len(live_jobs))
            try:
                if live_jobs:
                    self._run_batch(live_jobs, isolate_failures=True)
            except Exception as exc:
                # Last resort: keep the worker thread alive and answer every job.
                LOGGER.exception("Parakeet batch worker error")
                message = f"{type(exc).__name__}: {exc}"[:FFMPEG_ERROR_CONTEXT_CHARS]
                _clear_exception_frames(exc)
                for job in live_jobs:
                    if "output" not in job:
                        job.setdefault("error", InferenceError(message))
            finally:
                inference_finished_at = time.monotonic()
                inference_latency = inference_finished_at - inference_started_at
                for job in jobs:
                    job["inference_finished_at"] = inference_finished_at
                    # Release the audio now; the request thread only needs output.
                    job["waveform"] = None
                    PARAKEET_METRICS.observe_latency("inference", inference_latency)
                    job["done"].set()

    def _run_batch(self, jobs, *, isolate_failures: bool):
        """Run one batch and give every job either an output or its own error.

        If a batch fails and the CUDA context is still usable, each job is retried
        alone, so a single bad or oversized input fails only its own request.
        """
        if self._unhealthy_reason is not None:
            for job in jobs:
                job["error"] = InferenceError(
                    f"replica is restarting: {self._unhealthy_reason}"
                )
            return
        model = self._model
        if jobs[0]["bucket"] == LONG_AUDIO_BUCKET:
            model = self._long_model
            if model is None:
                # _duration_bucket() refuses this route first; never run on None.
                raise RuntimeError("long-form batch routed with no long-form model")
        try:
            outputs = self._infer_waveforms(
                model,
                [job["waveform"] for job in jobs],
                with_timestamps=jobs[0]["timestamps"],
            )
            if len(outputs) != len(jobs):
                raise RuntimeError(
                    f"NeMo returned {len(outputs)} outputs for {len(jobs)} inputs"
                )
            for job, output in zip(jobs, outputs):
                job["output"] = output
            return
        except Exception as exc:
            message = self._handle_inference_failure(exc, len(jobs))

        if isolate_failures and len(jobs) > 1 and self._unhealthy_reason is None:
            LOGGER.warning(
                f"Retrying {len(jobs)} requests from a failed batch one at a time"
            )
            for job in jobs:
                self._run_batch([job], isolate_failures=False)
            return
        for job in jobs:
            job["error"] = InferenceError(message)

    def _handle_inference_failure(self, exc: Exception, batch_size: int) -> str:
        """Log a failure once, free what it pinned, and check the CUDA context."""
        message = f"{type(exc).__name__}: {exc}"[:FFMPEG_ERROR_CONTEXT_CHARS]
        LOGGER.error(
            f"Parakeet inference failed for a batch of {batch_size}",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        _clear_exception_frames(exc)
        is_oom = "out of memory" in str(exc).lower()
        sticky = any(marker in str(exc) for marker in STICKY_CUDA_ERROR_MARKERS)
        if sticky or not self._cuda_context_usable(release_cache=is_oom):
            self._mark_unhealthy(message)
        return message

    @staticmethod
    def _cuda_context_usable(*, release_cache: bool) -> bool:
        import torch

        try:
            if release_cache:
                torch.cuda.empty_cache()
            torch.cuda.synchronize()
            probe = torch.ones(1, device="cuda")
            return float((probe + 1).item()) == 2.0
        except Exception:
            LOGGER.exception("CUDA context probe failed")
            return False

    def _mark_unhealthy(self, reason: str):
        if self._unhealthy_reason is None:
            self._unhealthy_reason = f"CUDA context unusable after: {reason}"
            LOGGER.error(
                "Parakeet replica marked unhealthy; is_healthy() now returns False "
                f"so the platform restarts it. Cause: {reason}"
            )

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
    def _jobs_compatible(first, candidate) -> bool:
        return (
            first["timestamps"] is candidate["timestamps"]
            and first["bucket"] == candidate["bucket"]
        )

    @staticmethod
    def _duration_bucket(duration_seconds: float):
        if duration_seconds > FULL_ATTENTION_MAX_SECONDS:
            if not LONG_AUDIO_ENABLED:
                # predict()'s 413 fires first when the two limits are equal.
                raise RuntimeError(
                    f"{duration_seconds:.0f} s clip routed to the long-form model, "
                    "which is not loaded (MAX_AUDIO_SECONDS <= FULL_ATTENTION_MAX_SECONDS)"
                )
            # Served by the local-attention model, one clip per batch.
            return LONG_AUDIO_BUCKET
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

    @staticmethod
    def _max_batch_size(bucket) -> int:
        """Cap a batch so its attention memory stays within one maximum clip's.

        Full-attention memory per clip grows with the square of its padded length,
        so B clips of bucket length L cost about as much as one clip of
        L * sqrt(B). Allowing B <= (FULL_ATTENTION_MAX_SECONDS / L)^2 keeps every
        batch inside the single FULL_ATTENTION_MAX_SECONDS pass that load()'s
        capacity check proves fits. With the default 24 min limit, every bucket up
        to 256 s keeps MAX_BATCH_SIZE unchanged; long-form clips always run alone.
        """
        if bucket == LONG_AUDIO_BUCKET:
            return 1
        by_memory = math.floor((FULL_ATTENTION_MAX_SECONDS / bucket) ** 2)
        return max(1, min(MAX_BATCH_SIZE, by_memory))

    def _cudnn_for(self, model) -> bool:
        if self._long_model is not None and model is self._long_model:
            return LONG_PATH_CUDNN
        return SHORT_PATH_CUDNN

    def _infer_waveforms(self, model, waveforms, *, with_timestamps: bool):
        import torch

        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16),
            _cudnn_enabled(self._cudnn_for(model)),
        ):
            if DIRECT_FORWARD and not with_timestamps:
                return self._infer_direct(model, waveforms)
            return model.transcribe(
                waveforms,
                batch_size=len(waveforms),
                timestamps=with_timestamps,
                verbose=False,
                num_workers=0,
            )

    @staticmethod
    def _infer_direct(model, waveforms):
        """Run NeMo's encoder and batched TDT decoder without a temporary DataLoader."""
        import numpy as np
        import torch

        lengths = np.asarray(
            [waveform.shape[0] for waveform in waveforms], dtype=np.int64
        )
        max_length = int(lengths.max())
        signals = np.zeros((len(waveforms), max_length), dtype=np.float32)
        for index, waveform in enumerate(waveforms):
            signals[index, : waveform.shape[0]] = waveform

        started_at = time.perf_counter()
        input_signal = torch.from_numpy(signals).to(device="cuda")
        input_signal_length = torch.from_numpy(lengths).to(device="cuda")
        encoded, encoded_length = model.forward(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
        )
        timing = _TIMING_STATE["serving"] and _TIMING_STATE["batches"] < TIMING_LOG_BATCHES
        if timing:
            torch.cuda.synchronize()
            encoded_at = time.perf_counter()
        hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded,
            encoded_lengths=encoded_length,
            return_hypotheses=True,
        )
        if timing:
            _TIMING_STATE["batches"] += 1
            LOGGER.info(
                "Parakeet batch timing: "
                f"batch={len(waveforms)}, max_audio_s={max_length / AUDIO_SAMPLE_RATE_HZ:.2f}, "
                f"encoder_ms={(encoded_at - started_at) * 1000:.1f}, "
                f"decoder_ms={(time.perf_counter() - encoded_at) * 1000:.1f}"
            )
        return hypotheses

    @staticmethod
    def _probe_stages(model, name, runs):
        """Diagnostic (STAGE_PROBE): time preprocessor, encoder and decoder.

        Each length is new to the process, so its first pass shows any per-length
        setup cost (cuFFT plans, cuDNN plan building) and its second pass the steady
        cost. `runs` is ((cudnn_enabled, lengths_seconds), ...).
        """
        import numpy as np
        import torch

        rng = np.random.default_rng(0)

        def stages(seconds):
            waveform = (
                0.05 * rng.standard_normal(round(seconds * AUDIO_SAMPLE_RATE_HZ))
            ).astype(np.float32)
            out = []
            for _ in range(2):
                with (
                    torch.inference_mode(),
                    torch.autocast(device_type="cuda", dtype=torch.float16),
                ):
                    signal = torch.from_numpy(waveform).to(device="cuda")[None]
                    length = torch.tensor([waveform.shape[0]], device="cuda")
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    feats, feat_len = model.preprocessor(input_signal=signal, length=length)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    encoded, encoded_len = model.encoder(audio_signal=feats, length=feat_len)
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    model.decoding.rnnt_decoder_predictions_tensor(
                        encoder_output=encoded,
                        encoded_lengths=encoded_len,
                        return_hypotheses=True,
                    )
                    torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    del signal, length, feats, feat_len, encoded, encoded_len
                out.append(
                    f"pre={1000 * (t1 - t0):.1f}/enc={1000 * (t2 - t1):.1f}/"
                    f"dec={1000 * (t3 - t2):.1f}"
                )
            return out

        for cudnn_enabled, lengths in runs:
            label = "cudnn_on" if cudnn_enabled else "cudnn_off"
            with _cudnn_enabled(cudnn_enabled):
                for seconds in lengths:
                    first, second = stages(seconds)
                    LOGGER.info(
                        f"Parakeet stage probe: model={name}, {label}, audio_s={seconds}, "
                        f"first_ms[{first}], repeat_ms[{second}]"
                    )
        torch.cuda.empty_cache()

    def _compare_cudnn_transcripts(self):
        """Diagnostic (STAGE_PROBE): transcribe real speech with cuDNN on and off.

        Uses public clips cut into 2-16 s pieces, so every piece fits the short
        model's warmed decoder graph (batch <= 2, <= 30 s) and nothing is recaptured.
        """
        import difflib

        import numpy as np
        import torch

        urls = (
            "https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav",
            "https://test-audios-public.s3.us-west-2.amazonaws.com/10-sec-01-podcast.m4a",
            "https://test-audios-public.s3.us-west-2.amazonaws.com/5-min-01-podcast.m4a",
        )
        rng = np.random.default_rng(1)
        clips = []
        for url in urls:
            try:
                waveform = self._decode_waveform(self._decode_audio_input({"audio_url": url}))
            except Exception as exc:
                LOGGER.warning(f"Parakeet cuDNN transcript check: skipped {url}: {exc}")
                continue
            start = 0
            while start < waveform.shape[0]:
                size = round(rng.uniform(2.0, 16.0) * AUDIO_SAMPLE_RATE_HZ)
                piece = waveform[start : start + size]
                if piece.shape[0] >= AUDIO_SAMPLE_RATE_HZ:
                    clips.append(np.ascontiguousarray(piece))
                start += size

        def transcribe(clip, enabled):
            with (
                torch.inference_mode(),
                torch.autocast(device_type="cuda", dtype=torch.float16),
                _cudnn_enabled(enabled),
            ):
                return self._infer_direct(self._model, [clip])[0].text

        def transcribe_batch(batch, enabled):
            with (
                torch.inference_mode(),
                torch.autocast(device_type="cuda", dtype=torch.float16),
                _cudnn_enabled(enabled),
            ):
                return [h.text for h in self._infer_direct(self._model, batch)]

        identical = edits = words = 0
        single_on = []
        for index, clip in enumerate(clips):
            text_on = transcribe(clip, True)
            text_off = transcribe(clip, False)
            single_on.append(text_on)
            on_words, off_words = text_on.split(), text_off.split()
            words += len(on_words)
            if text_on == text_off:
                identical += 1
                continue
            matcher = difflib.SequenceMatcher(a=on_words, b=off_words, autojunk=False)
            edits += sum(
                max(i2 - i1, j2 - j1)
                for tag, i1, i2, j1, j2 in matcher.get_opcodes()
                if tag != "equal"
            )
            LOGGER.info(
                f"Parakeet cuDNN transcript diff: clip={index}, "
                f"audio_s={clip.shape[0] / AUDIO_SAMPLE_RATE_HZ:.2f}, "
                f"on={text_on[:300]!r}, off={text_off[:300]!r}"
            )
        LOGGER.info(
            f"Parakeet cuDNN transcript check: clips={len(clips)}, identical={identical}, "
            f"word_edits={edits}, words_cudnn_on={words}"
        )

        # Serving batches clips of one duration bucket, padded to the longest. Check
        # that batching (padding) and cuDNN leave transcripts unchanged there too.
        batch_size = 4
        same_on = same_off = on_vs_off = 0
        for start in range(0, len(clips), batch_size):
            batch = clips[start : start + batch_size]
            batched_on = transcribe_batch(batch, True)
            batched_off = transcribe_batch(batch, False)
            for offset, (text_on, text_off) in enumerate(zip(batched_on, batched_off)):
                single = single_on[start + offset]
                same_on += text_on == single
                same_off += text_off == single
                on_vs_off += text_on == text_off
                if text_on != single or text_off != single:
                    LOGGER.info(
                        f"Parakeet batched transcript diff: clip={start + offset}, "
                        f"batch_max_s={max(c.shape[0] for c in batch) / AUDIO_SAMPLE_RATE_HZ:.2f}, "
                        f"single={single[:300]!r}, batched_on={text_on[:300]!r}, "
                        f"batched_off={text_off[:300]!r}"
                    )
        LOGGER.info(
            f"Parakeet batched transcript check: clips={len(clips)}, batch_size={batch_size}, "
            f"batched_cudnn_on_equals_single={same_on}, "
            f"batched_cudnn_off_equals_single={same_off}, "
            f"batched_on_equals_batched_off={on_vs_off}"
        )

    def _warmup(self):
        if not WARMUP_AUDIO_SECONDS:
            return

        import numpy as np

        for duration_seconds in WARMUP_AUDIO_SECONDS:
            waveform = np.zeros(
                round(duration_seconds * AUDIO_SAMPLE_RATE_HZ), dtype=np.float32
            )
            self._infer_waveforms(
                self._model,
                [waveform for _ in range(WARMUP_BATCH_SIZE)],
                with_timestamps=False,
            )
        LOGGER.info(
            f"Parakeet warmup complete: durations={WARMUP_AUDIO_SECONDS}, "
            f"batch_size={WARMUP_BATCH_SIZE}"
        )

    def _log_decoder_graph_mode(self, phase: str, *, enforce: bool, model=None):
        model = self._model if model is None else model
        decoder = getattr(getattr(model, "decoding", None), "decoding", None)
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
            with sf.SoundFile(io.BytesIO(audio_bytes)) as sound_file:
                # Check the header before allocating: a WAV/FLAC over the cap is
                # rejected without decoding it.
                if sound_file.frames > MAX_AUDIO_SECONDS * sound_file.samplerate:
                    raise _client_error(
                        413,
                        f"audio is longer than this deployment's {MAX_AUDIO_SECONDS:g} s "
                        "limit (MAX_AUDIO_SECONDS)",
                    )
                sample_rate = sound_file.samplerate
                waveform = sound_file.read(dtype="float32", always_2d=True)
            waveform = waveform.mean(axis=1)
            if sample_rate == AUDIO_SAMPLE_RATE_HZ:
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
        #
        # Samples go to a temp file too, not a pipe: capture_output would hold
        # the chunked read, the joined bytes and a numpy copy at once (~3x the
        # waveform, ~0.7 GB per hour of audio). np.fromfile reads it once. -t
        # stops decoding one second past the duration cap, so an arbitrarily long
        # input can never produce more than MAX_AUDIO_SECONDS + 1 s of samples.
        with tempfile.TemporaryDirectory(prefix="parakeet-decode-") as work_dir:
            input_path = os.path.join(work_dir, "input.audio")
            output_path = os.path.join(work_dir, "output.f32le")
            with open(input_path, "wb") as input_file:
                input_file.write(audio_bytes)
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-i",
                    input_path,
                    "-t",
                    f"{MAX_AUDIO_SECONDS + 1:g}",
                    "-ac",
                    "1",
                    "-ar",
                    str(AUDIO_SAMPLE_RATE_HZ),
                    "-f",
                    "f32le",
                    "-y",
                    output_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            stderr = proc.stderr.decode("utf-8", errors="replace")
            if proc.returncode != 0:
                raise ValueError(
                    f"could not decode audio: {stderr[-FFMPEG_ERROR_CONTEXT_CHARS:]}"
                )
            waveform = (
                np.fromfile(output_path, dtype=np.float32)
                if os.path.exists(output_path)
                else np.zeros(0, dtype=np.float32)
            )
        if waveform.shape[0] == 0:
            # Fail this request alone with the decoder's message instead of
            # sending a zero-length waveform into a shared batch.
            raise ValueError(
                "decoded audio has no samples: "
                f"{stderr[-FFMPEG_ERROR_CONTEXT_CHARS:] or 'ffmpeg produced no output'}"
            )
        return waveform

    def _decode_audio_input(self, request: dict):
        too_large = (
            f"audio input is larger than this deployment's {MAX_AUDIO_INPUT_BYTES} "
            "byte limit (MAX_AUDIO_INPUT_BYTES)"
        )
        if "audio_b64" in request:
            encoded = request["audio_b64"]
            # Refuse on the encoded length first, before allocating the decoded copy.
            if len(encoded) > (MAX_AUDIO_INPUT_BYTES + 2) // 3 * 4:
                raise _client_error(413, too_large)
            decoded = base64.b64decode(encoded, validate=True)
            if len(decoded) > MAX_AUDIO_INPUT_BYTES:
                raise _client_error(413, too_large)
            return decoded
        if "audio_url" in request:
            # Stream with a byte cap instead of resp.content, so an oversized
            # file is refused before it is buffered in full.
            with self._http.stream("GET", request["audio_url"]) as resp:
                resp.raise_for_status()
                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > MAX_AUDIO_INPUT_BYTES:
                    raise _client_error(413, too_large)
                body = bytearray()
                for piece in resp.iter_bytes():
                    body.extend(piece)
                    if len(body) > MAX_AUDIO_INPUT_BYTES:
                        raise _client_error(413, too_large)
            return body
        raise ValueError("provide audio_url or audio_b64")
