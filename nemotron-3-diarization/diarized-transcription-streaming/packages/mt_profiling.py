"""Profiling hooks for the multitalker streaming truss (MT_PROFILE=1 only; scratchpad copy).

Wraps the NeMo calls one streaming step is made of with synchronize-bounded wall timers and
records them into a thread-local per-step dict; every PROFILE_EVERY steps one step runs under
torch.profiler (CPU+CUDA, shapes) and its tables/counters are attached to the session so the
WebSocket handler can ship them to the client. A sampler thread polls nvidia-smi once a second.
"""

import collections
import logging
import os
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("MT_PROFILE", "0") == "1"
PROFILE_EVERY = int(os.environ.get("MT_PROFILE_EVERY", "50"))
LOG_EVERY = max(1, int(os.environ.get("MT_PROFILE_LOG_EVERY", "1")))   # MTPROF log line every n steps
GPU_LOG_EVERY = max(0, int(os.environ.get("MT_GPU_LOG_EVERY", "10")))  # MTGPU log line every n samples (0=off)
_tls = threading.local()


def current():
    return getattr(_tls, "rec", None)


def set_current(rec):
    _tls.rec = rec


def timed(name, fn, torch):
    """Wrap fn so its wall time (GPU-synchronised when rec['_sync']) accumulates in rec[name]."""
    def w(*a, **k):
        rec = current()
        if rec is None or rec.get("_profiling"):
            return fn(*a, **k)
        if rec.get("_sync"):
            torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            if rec.get("_sync"):
                torch.cuda.synchronize()
            rec[name] = rec.get(name, 0.0) + (time.perf_counter() - t)
            rec["_n_" + name] = rec.get("_n_" + name, 0) + 1
    return w


def wrap_active_speakers(fn, torch):
    """get_active_speakers_info wrapper that also records how many ASR instances run this step."""
    inner = timed("gather", fn, torch)

    def w(active_speakers, chunk_audio, chunk_lengths):
        rec = current()
        if rec is not None:
            rec["k_active"] = sum(len(s) for s in active_speakers)
        return inner(active_speakers, chunk_audio, chunk_lengths)
    return w


def wrap_shared(asr, diar, torch):
    diar.forward_streaming_step = timed("diar", diar.forward_streaming_step, torch)
    asr.encoder.cache_aware_stream_step = timed("enc", asr.encoder.cache_aware_stream_step, torch)
    asr.decoding.rnnt_decoder_predictions_tensor = timed(
        "dec", asr.decoding.rnnt_decoder_predictions_tensor, torch)
    asr.set_speaker_targets = timed("spk_tgt", asr.set_speaker_targets, torch)


def wrap_session(streamer, torch):
    im = streamer.instance_manager
    im.get_active_speakers_info = wrap_active_speakers(im.get_active_speakers_info, torch)
    im.update_seglsts = timed("seglst", im.update_seglsts, torch)
    im.update_asr_state = timed("upd_state", im.update_asr_state, torch)
    streamer.forward_pre_encoded = timed("pre_enc", streamer.forward_pre_encoded, torch)
    streamer._find_active_speakers = timed("gate", streamer._find_active_speakers, torch)
    streamer._prepare_diar_chunk = timed("diar_prep", streamer._prepare_diar_chunk, torch)


def run_profiled(fn, torch):
    """Run fn under torch.profiler and return (tables, counters)."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        fn()
        torch.cuda.synchronize()
    wall_us = (time.perf_counter() - t0) * 1e6
    ka = prof.key_averages()
    evs = prof.events()
    kernels = [e for e in evs if e.device_type == DeviceType.CUDA]
    rt = collections.Counter(e.name for e in evs if e.device_type == DeviceType.CPU
                             and (e.name.startswith("cuda") or e.name.startswith("Memcpy")
                                  or e.name.startswith("Memset")))
    dev_attr = "self_device_time_total" if hasattr(ka[0], "self_device_time_total") else "self_cuda_time_total"
    gpu_busy_us = sum(getattr(e, dev_attr) for e in ka)
    n_aten = sum(e.count for e in ka if e.key.startswith("aten::"))
    counters = {
        "wall_us": round(wall_us),
        "self_cpu_total_us": round(ka.self_cpu_time_total),
        "python_only_us": round(wall_us - ka.self_cpu_time_total),
        "gpu_busy_us": round(gpu_busy_us),
        "n_kernel_events": len(kernels),
        "n_kernel_names": len({e.name for e in kernels}),
        "n_aten_calls": n_aten,
        "runtime_calls": {k: v for k, v in rt.most_common(30)},
        "launches": rt.get("cudaLaunchKernel", 0) + rt.get("cudaGraphLaunch", 0),
        "graph_launches": rt.get("cudaGraphLaunch", 0),
        "syncs": sum(v for k, v in rt.items() if "Synchronize" in k),
        "memcpy": sum(v for k, v in rt.items() if "Memcpy" in k or "memcpy" in k.lower()),
        "torch_threads": torch.get_num_threads(),
    }
    tables = {
        "cuda": ka.table(sort_by="cuda_time_total", row_limit=40),
        "cpu": ka.table(sort_by="cpu_time_total", row_limit=40),
        "self_cpu": ka.table(sort_by="self_cpu_time_total", row_limit=25),
    }
    return tables, counters


class GpuSampler(threading.Thread):
    """Polls nvidia-smi every second; keeps (t, util%, mem MiB) samples in a ring buffer."""

    def __init__(self, torch):
        super().__init__(daemon=True, name="gpu-sampler")
        self.samples = collections.deque(maxlen=7200)
        self._torch = torch

    def run(self):
        try:
            p = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                  "--format=csv,noheader,nounits", "-l", "1"],
                                 stdout=subprocess.PIPE, text=True)
            pending = []
            for line in p.stdout:
                try:
                    u, m = [float(x) for x in line.strip().split(",")]
                except ValueError:
                    continue
                s = (round(time.time(), 2), u, m)
                self.samples.append(s)
                pending.append(s)
                if GPU_LOG_EVERY and len(pending) >= GPU_LOG_EVERY:
                    # GPU utilisation timeline in the deployment log (util %, memory MiB per second).
                    logger.info("MTGPU t0=%.0f util=%s mem=%s", pending[0][0], [int(x[1]) for x in pending],
                                [int(x[2]) for x in pending])
                    pending = []
        except Exception as e:  # noqa: BLE001 — fall back to torch's NVML wrapper
            logger.warning("nvidia-smi sampler failed (%s); using torch.cuda.utilization", e)
            while True:
                try:
                    self.samples.append((round(time.time(), 2), float(self._torch.cuda.utilization()),
                                         self._torch.cuda.memory_reserved() / 2**20))
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(1.0)

    def since(self, t):
        return [s for s in self.samples if s[0] > t]
