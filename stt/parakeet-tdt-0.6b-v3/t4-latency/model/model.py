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
import gc
import io
import os
import queue
import subprocess
import threading
import time

from prometheus_client import Gauge, Histogram

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/parakeet-tdt-0.6b-v3")
NEMO_CHECKPOINT = os.path.join(MODEL_DIR, "parakeet-tdt-0.6b-v3.nemo")
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


class Model:
    def __init__(self, **kwargs):
        self._model = None
        self._jobs = queue.Queue()
        self._deferred_jobs = collections.deque()
        self._http = None

    def load(self):
        import httpx
        import nemo.collections.asr as nemo_asr
        import torch

        self._model = nemo_asr.models.ASRModel.restore_from(
            NEMO_CHECKPOINT, map_location=torch.device("cuda")
        )
        self._model.eval()
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

        print(
            "Parakeet runtime config: "
            f"direct_forward={DIRECT_FORWARD}, max_batch_size={MAX_BATCH_SIZE}, "
            f"batch_window_ms={BATCH_WINDOW_SECONDS * 1000:g}, "
            f"batch_buckets_seconds={BATCH_BUCKET_SECONDS or 'disabled'}"
        )
        self._log_decoder_graph_mode("load", enforce=False)
        self._warmup()
        self._log_decoder_graph_mode("warmup", enforce=REQUIRE_FULL_CUDA_GRAPH)
        if FREEZE_GC_AFTER_WARMUP:
            gc.collect()
            gc.freeze()
            print("Python GC startup state frozen")

        threading.Thread(target=self._batch_worker, daemon=True).start()

    def predict(self, request: dict) -> dict:
        # Wrap predict so tracebacks reach the deployment logs.
        try:
            return self._predict_impl(request)
        except Exception:
            import traceback

            print("PREDICT FAILED:\n" + traceback.format_exc())
            raise

    def _predict_impl(self, request: dict) -> dict:
        request_started_at = time.monotonic()
        preprocessing_started_at = request_started_at
        try:
            try:
                audio_bytes = self._decode_audio_input(request)
                with_timestamps = bool(request.get("timestamps", False))
                waveform = self._decode_waveform(audio_bytes)
                job = {
                    "waveform": waveform,
                    "bucket": self._duration_bucket(
                        waveform.shape[0] / AUDIO_SAMPLE_RATE_HZ
                    ),
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
            PARAKEET_METRICS.observe_latency(
                "total", time.monotonic() - request_started_at
            )

    def _batch_worker(self):
        """Run all GPU work on one thread and opportunistically microbatch."""
        while True:
            first = self._next_job()
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
            PARAKEET_METRICS.batch_size.observe(len(jobs))
            try:
                outputs = self._infer_waveforms(
                    [job["waveform"] for job in jobs],
                    with_timestamps=first["timestamps"],
                )
                if len(outputs) != len(jobs):
                    raise RuntimeError(
                        f"NeMo returned {len(outputs)} outputs for {len(jobs)} inputs"
                    )
                for job, output in zip(jobs, outputs):
                    job["output"] = output
            except Exception as exc:
                for job in jobs:
                    job["error"] = exc
            finally:
                inference_finished_at = time.monotonic()
                inference_latency = inference_finished_at - inference_started_at
                for job in jobs:
                    job["inference_finished_at"] = inference_finished_at
                    PARAKEET_METRICS.observe_latency("inference", inference_latency)
                    job["done"].set()

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

        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16),
        ):
            if DIRECT_FORWARD and not with_timestamps:
                return self._infer_direct(waveforms)
            return self._model.transcribe(
                waveforms,
                batch_size=len(waveforms),
                timestamps=with_timestamps,
                verbose=False,
                num_workers=0,
            )

    def _infer_direct(self, waveforms):
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

        input_signal = torch.from_numpy(signals).to(device="cuda")
        input_signal_length = torch.from_numpy(lengths).to(device="cuda")
        encoded, encoded_length = self._model.forward(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
        )
        return self._model.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded,
            encoded_lengths=encoded_length,
            return_hypotheses=True,
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
                [waveform for _ in range(WARMUP_BATCH_SIZE)],
                with_timestamps=False,
            )
        print(
            "Parakeet warmup complete: "
            f"durations={WARMUP_AUDIO_SECONDS}, batch_size={WARMUP_BATCH_SIZE}"
        )

    def _log_decoder_graph_mode(self, phase: str, *, enforce: bool):
        decoder = getattr(getattr(self._model, "decoding", None), "decoding", None)
        computer = getattr(decoder, "decoding_computer", None)
        mode = getattr(computer, "cuda_graphs_mode", None)
        mode_value = getattr(mode, "value", mode)
        print(
            f"Parakeet TDT decoder CUDA graph mode ({phase}): {mode_value or 'unavailable'}"
        )
        if enforce and mode_value != "full_graph":
            raise RuntimeError(
                "REQUIRE_FULL_CUDA_GRAPH is enabled, but NeMo selected "
                f"decoder CUDA graph mode {mode_value!r}"
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
                return np.ascontiguousarray(waveform, dtype=np.float32)
        except (RuntimeError, sf.LibsndfileError):
            pass

        return Model._decode_with_ffmpeg(audio_bytes)

    @staticmethod
    def _decode_with_ffmpeg(audio_bytes: bytes):
        """Decode or resample formats unsupported by the in-process fast path."""
        import numpy as np

        proc = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                "pipe:0",
                "-ac",
                "1",
                "-ar",
                str(AUDIO_SAMPLE_RATE_HZ),
                "-f",
                "f32le",
                "pipe:1",
            ],
            input=audio_bytes,
            capture_output=True,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            raise ValueError(
                f"could not decode audio: {stderr[-FFMPEG_ERROR_CONTEXT_CHARS:]}"
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
