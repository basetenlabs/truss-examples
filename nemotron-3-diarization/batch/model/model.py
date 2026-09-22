"""Batch truss for nvidia/Nemotron-3-Diarization-preview (NeMo streaming Sortformer).

TRACKING ONLY — evaluation-license, early-access model. Loads the gated .nemo checkpoint,
runs whole-file diarization, returns speaker turns ({start, end, speaker}) + raw segments.
Up to 8 speakers, arrival-ordered labels.

Request-level latency: the offered algorithmic-latency profiles are pre-loaded as separate
model instances (the streaming knobs live on shared module state, so a dedicated instance
per profile lets a request pick its latency without mutating another in-flight request).
`diarization_input.latency` selects one.

Performance: the default engine (`NEMO_DIAR_ENGINE=graphs`, model/graph_runner.py) runs
NVIDIA's whole-file streaming loop with every step — encoder, head and the speaker-cache/FIFO
update — replayed from a CUDA graph captured per state shape, so the launch-bound streaming
profiles run at the GPU's pace and, in fp32, produce NeMo's eager output bit for bit (no
torch.compile drift). `NEMO_DIAR_ENGINE=nemo` keeps the reference-script path (bf16 autocast,
torch.compile on the encoder, growing state) for A/B. The server adds what the script cannot:
concurrent HTTP requests for the same profile coalesce into ONE batched call (window re-armed by
each arrival, cap NEMO_DIAR_MB_CAP, longest files first), audio is decoded to pinned PCM in the
request thread and handed to the GPU without temp files or DataLoader workers, post-processing
runs off the GPU thread, and every response carries its server-side timing profile.
"""

import base64
import concurrent.futures
import contextlib
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # graph_runner.py lives next to this file
os.environ.setdefault("TQDM_DISABLE", "1")  # NeMo's post-processing wraps every file in a tqdm bar

logger = logging.getLogger(__name__)

# (spkcache_len, fifo_len, chunk_len, chunk_right_context, spkcache_update_period), 80 ms frames.
# Input-buffer latency = (chunk_len + chunk_right_context) * 80 ms.
PROFILES = {
    "offline": (264, 40, 340, 40, 300),   # 30.4 s — best DER
    "low": (264, 264, 9, 4, 222),         # 1.04 s
    "verylow": (264, 264, 6, 2, 222),     # 0.64 s
    "ultralow": (264, 264, 3, 1, 222),    # 0.32 s
}
# Extra profiles for experiments: NEMO_DIAR_CUSTOM_PROFILES="name:spkcache,fifo,chunk,rc,update;name2:..."
for _spec in os.environ.get("NEMO_DIAR_CUSTOM_PROFILES", "").split(";"):
    if ":" in _spec:
        _n, _v = _spec.split(":", 1)
        PROFILES[_n.strip()] = tuple(int(x) for x in _v.split(","))
SAMPLE_RATE = 16000
MB_WINDOW_S = float(os.environ.get("NEMO_DIAR_MB_WINDOW_MS", "100")) / 1000
MB_MAX_WAIT_S = float(os.environ.get("NEMO_DIAR_MB_MAX_WAIT_MS", "1500")) / 1000
MB_FILL_S = float(os.environ.get("NEMO_DIAR_MB_FILL_MS", "400")) / 1000
MB_CAP = int(os.environ.get("NEMO_DIAR_MB_CAP", "32"))
# No size-1 graph: dynamo specialises batch 1 into its own compiled encoder and that graph carries a
# one-sided missed-speech bias (+0.73 pt at bs=1 in bf16 AND TF32; 0 when the same file runs as
# identical rows in a larger graph). bf16 also keeps a confusion residual at 2 rows, so a single
# request runs as 4 rows (same latency in the launch-bound regime). Per-profile via _<PROFILE>.
GRAPH_SIZES = tuple(int(x) for x in os.environ.get("NEMO_DIAR_GRAPH_SIZES", "4,8,12,16,20,24,28,32").split(","))


def _graph_sizes(profile):
    return tuple(int(x) for x in _env("NEMO_DIAR_GRAPH_SIZES", ",".join(map(str, GRAPH_SIZES)), profile).split(","))
WARM_S = float(os.environ.get("NEMO_DIAR_WARM_S", "100"))
DTYPES = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}


def _env(name: str, default: str, profile: str | None = None) -> str:
    """Env knob; NAME_<PROFILE> overrides NAME for one profile (e.g. NEMO_DIAR_DTYPE_ULTRALOW=fp32)."""
    if profile is not None and f"{name}_{profile.upper()}" in os.environ:
        return os.environ[f"{name}_{profile.upper()}"]
    return os.environ.get(name, default)


def _flag(name: str, default: str, profile: str | None = None) -> bool:
    return _env(name, default, profile) == "1"


def _nemo_build_id() -> str:
    """Resolved NeMo overlay commit (pip records it for VCS installs), for the startup log."""
    try:
        import importlib.metadata as md
        import json as _json
        info = _json.loads(md.distribution("nemo_toolkit").read_text("direct_url.json") or "{}")
        return f"{md.version('nemo_toolkit')}@{info.get('vcs_info', {}).get('commit_id', '?')[:12]}"
    except Exception:  # noqa: BLE001
        return "unknown"


def _bad_request(detail: str):
    """Client errors must surface as HTTP 400, not the generic 500 a bare exception gives."""
    from fastapi import HTTPException
    return HTTPException(status_code=400, detail=detail)


def _compiled_graphs() -> int:
    """torch.compile graph count; a rise after warmup means a recompile in the request path."""
    try:
        import torch
        return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
    except Exception:  # noqa: BLE001
        return -1


_GRAPHS_AFTER_WARMUP = None


def _time_method(obj, name: str, sink: dict, key: str):
    """Wrap a bound method so its wall time accumulates in sink[key] (one caller thread)."""
    fn = getattr(obj, name)

    def timed(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            sink[key] = sink.get(key, 0.0) + time.perf_counter() - t0

    setattr(obj, name, timed)


def _postprocess(model, preds, pp_params):
    """NeMo's `_diarize_output_processing` for one file: frame probabilities -> 'start end speaker' lines."""
    from nemo.collections.asr.parts.utils.speaker_utils import generate_diarization_output_lines
    from nemo.collections.asr.parts.utils.vad_utils import predlist_to_timestamps
    ts = predlist_to_timestamps(batch_preds_list=[preds.unsqueeze(0)], audio_rttm_map_dict={"x": {"offset": 0.0}},
                                cfg_vad_params=pp_params, unit_10ms_frame_count=model.output_subsampling_factor,
                                bypass_postprocessing=False)[0]
    return generate_diarization_output_lines(speaker_timestamps=ts, model_spk_num=len(ts))


class _Batcher:
    """Per-profile collector: coalesce queued requests, run one batched inference.

    Requests enqueue as soon as their bytes arrive and decode in parallel while the window
    runs, so a burst of uploads spread over a few hundred ms still lands in one batch. The
    window re-arms on every arrival and closes after MB_WINDOW_S of silence (MB_FILL_S while the
    count is below a captured batch size), MB_MAX_WAIT_S total, or MB_CAP items. A collector
    thread forms batches and waits for their decodes; the GPU thread runs them back to back, so
    the next batch's decode and window overlap the current forward. With the graph engine
    post-processing is one batched GPU binarization (`GraphRunner.segments`).
    """

    def __init__(self, model, name, *, runner=None, autocast_dtype=None):
        self.model = model
        self.name = name
        self.runner = runner
        self.autocast_dtype = autocast_dtype
        self.q = queue.Queue()
        self.ready = queue.Queue(maxsize=1)
        self.busy = threading.Event()
        self.timers = {}
        self.pp_params = None
        self.t_prev_end = None
        # post-processing runs here so the GPU thread moves on to the next batch's forward
        self.post = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix=f"post-{name}")
        if runner is None:
            for meth, key in (("_diarize_input_processing", "dl"), ("_diarize_forward", "fwd"),
                              ("_diarize_output_processing", "post")):
                _time_method(model, meth, self.timers, key)
        else:
            from nemo.collections.asr.parts.utils.vad_utils import load_postprocessing_from_yaml
            self.pp_params = load_postprocessing_from_yaml(None)
        threading.Thread(target=self._run, name=f"mb-{name}", daemon=True).start()

    def submit(self, decode_fut):
        """decode_fut resolves to 16 kHz float32 PCM (or raises a client error)."""
        fut = concurrent.futures.Future()
        self.q.put((decode_fut, fut, time.perf_counter()))
        return fut

    def _window(self, n):
        """Quiet time that closes the window: MB_WINDOW_S at a captured batch size, MB_FILL_S below one
        (a batch of 5 runs at the size-8 graph and costs the same, so a straggler is worth waiting for)."""
        if self.runner is None or n in self.runner.sizes or n >= MB_CAP:
            return MB_WINDOW_S
        return MB_FILL_S

    def _collect(self, seed):
        """Admit arrivals into one batch. Closes when the window has been quiet AND the GPU is idle AND
        every admitted decode is done, or at MB_MAX_WAIT_S / MB_CAP. While a forward is in flight the
        window never closes: arrivals keep coalescing into the batch that will run next (closing early
        would hand the GPU a 1-row batch that costs a full bs=1 step). Items still decoding when the
        batch closes are carried over as the seed of the next one."""
        batch = list(seed)
        if not batch:
            batch.append(self.q.get())
        t_first = time.perf_counter()
        deadline = t_first + self._window(len(batch))
        was_busy = self.busy.is_set()
        while len(batch) < MB_CAP:
            now = time.perf_counter()
            if now >= t_first + MB_MAX_WAIT_S:
                break
            decoding = [i for i, (dec, _, _) in enumerate(batch) if not dec.done()]
            if was_busy and not self.busy.is_set():
                # the forward just finished: give stragglers one fill window before handing over a
                # batch that would run padded to the next captured size (holding longer to fill the
                # cap was measured a no-op: per-row cost is flat from 16 rows up)
                was_busy = False
                deadline = max(deadline, now + self._window(len(batch)))
            if now >= deadline and not self.busy.is_set():
                if not decoding or now >= t_first + MB_MAX_WAIT_S / 2:
                    break
                if len(decoding) < len(batch):
                    carry = [batch[i] for i in decoding]
                    return [b for i, b in enumerate(batch) if i not in set(decoding)], carry
            try:
                batch.append(self.q.get(timeout=0.02))
            except queue.Empty:
                continue
            deadline = max(deadline, time.perf_counter() + self._window(len(batch)))
        return batch, []

    def infer(self, audios):
        """One batched call on the GPU thread: -> (list of per-file segment lists or futures, timing)."""
        import torch
        self.timers.clear()
        t0 = time.perf_counter()
        if self.runner is not None:
            out, out_lens, t = self.runner.run(audios)
            # post-processing off this thread: one future per file, resolved when the batch's segments are done
            segs = [concurrent.futures.Future() for _ in audios]
            batch_fut = self.post.submit(self.runner.segments, out, out_lens, self.pp_params)

            def _split(f, segs=segs):
                try:
                    for fut, seg in zip(segs, f.result()):
                        fut.set_result(seg)
                except Exception as e:  # noqa: BLE001
                    for fut in segs:
                        fut.set_exception(e)
            batch_fut.add_done_callback(_split)
            info = {"fwd_ms": round(t["fwd"] * 1000), "h2d_ms": round(t["h2d"] * 1000),
                    "mel_ms": round(t["mel"] * 1000), "steps_ms": round(t["steps"] * 1000), "graph_bs": t["B"],
                    "nonfinite_preds": t["nonfinite_preds"],
                    "nonfinite_state": t["nonfinite_state"], "pred_max": t["pred_max"], "pred_min": t["pred_min"],
                    "core_dtype": self.runner.stats()["dtype"]}
            if t["nonfinite_preds"] or t["nonfinite_state"]:
                logger.warning("[%s] non-finite values: preds %d state %d (batch of %d)", self.name,
                               t["nonfinite_preds"], t["nonfinite_state"], len(audios))
        else:
            ctx = (torch.autocast(device_type="cuda", dtype=self.autocast_dtype) if self.autocast_dtype is not None
                   else contextlib.nullcontext())
            np_audios = [a.numpy() if isinstance(a, torch.Tensor) else a for a in audios]
            with ctx:
                segs = self.model.diarize(audio=np_audios, sample_rate=SAMPLE_RATE, batch_size=len(audios),
                                          num_workers=0, verbose=False)
            info = {"dl_ms": round(self.timers.get("dl", 0) * 1000), "fwd_ms": round(self.timers.get("fwd", 0) * 1000),
                    "post_ms": round(self.timers.get("post", 0) * 1000)}
        if len(segs) != len(audios):
            raise RuntimeError(f"inference returned {len(segs)} results for {len(audios)} inputs")
        info["batch_ms"] = round((time.perf_counter() - t0) * 1000)
        return segs, info

    def _collector(self):
        """Form batches and wait for their decodes off the GPU thread, so the next batch's decode and
        window overlap the current forward; `ready` holds at most one batch, so under load the queue
        keeps filling while the GPU is busy and the following batch is bigger."""
        carry = []
        while True:
            batch, carry = self._collect(carry)
            ready = []
            for dec, fut, t_in in batch:  # a request whose audio failed to decode fails alone
                try:
                    pcm, dec_s = dec.result()
                    ready.append((pcm, fut, t_in, dec_s))
                except Exception as e:  # noqa: BLE001
                    fut.set_exception(e)
            if ready:
                self.ready.put(ready)

    def _run(self):
        global _GRAPHS_AFTER_WARMUP
        threading.Thread(target=self._collector, name=f"collect-{self.name}", daemon=True).start()
        while True:
            ready = self.ready.get()
            self.busy.set()
            ready.sort(key=lambda r: -len(r[0]))       # longest first: sync state pads every row to the longest
            audios = [r[0] for r in ready]
            audio_s = sum(len(a) for a in audios) / SAMPLE_RATE
            t0 = time.perf_counter()
            gap_ms = round((t0 - self.t_prev_end) * 1000) if self.t_prev_end else None   # GPU idle between batches
            try:
                segs, info = self.infer(audios)
                self.t_prev_end = time.perf_counter()
                info.update(batch_n=len(ready), batch_audio_s=round(audio_s, 1), gap_ms=gap_ms)
                for (pcm, fut, t_in, dec_s), s in zip(ready, segs):
                    fut.set_result((s, dict(info, queue_ms=round((t0 - t_in) * 1000), decode_ms=round(dec_s * 1000),
                                            audio_s=round(len(pcm) / SAMPLE_RATE, 2))))
            except Exception as e:  # noqa: BLE001 — fail the whole batch, callers see the error
                for _, fut, _, _ in ready:
                    fut.set_exception(e)
                logger.exception("[%s] batch of %d failed", self.name, len(ready))
                self.busy.clear()
                continue
            self.busy.clear()
            if self.runner is None:
                graphs = _compiled_graphs()
                if _GRAPHS_AFTER_WARMUP is not None and graphs > _GRAPHS_AFTER_WARMUP:
                    logger.warning("[%s] recompile under traffic: graphs %d -> %d (batch=%d)",
                                   self.name, _GRAPHS_AFTER_WARMUP, graphs, len(ready))
                    _GRAPHS_AFTER_WARMUP = graphs
            logger.info("[%s] batched %d req, %.0fs audio: fwd %.2fs (RTFx %.0f) batch %.2fs %s", self.name,
                        len(ready), audio_s, info["fwd_ms"] / 1000, audio_s / max(1e-3, info["fwd_ms"] / 1000),
                        info["batch_ms"] / 1000, {k: v for k, v in info.items() if k.endswith("_ms")})


class Model:
    def __init__(self, **kwargs):
        self._secrets = kwargs.get("secrets", {})
        self._models = {}
        self._batchers = {}
        self._runners = {}
        self._shared_enc = None
        # ffmpeg decodes run here, overlapping the batching window (a subprocess each, no GIL).
        self._decoder = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("NEMO_DIAR_DECODE_WORKERS", "8")), thread_name_prefix="decode")

    @staticmethod
    def _knobs(p):
        engine = _env("NEMO_DIAR_ENGINE", "graphs", p)
        dtype = _env("NEMO_DIAR_DTYPE", "bf16" if _flag("NEMO_DIAR_BF16", "1", p) else "fp32", p)
        if dtype not in DTYPES:
            raise ValueError(f"NEMO_DIAR_DTYPE must be one of {list(DTYPES)}, got {dtype!r}")
        return {"engine": engine, "dtype": dtype, "compile_on": _flag("NEMO_DIAR_COMPILE", "1", p),
                "attn_mode": _env("NEMO_DIAR_ATTN_MODE", "", p) or None, "post_fp32": _flag("NEMO_DIAR_POST_FP32", "0", p),
                "pre_fp32": _flag("NEMO_DIAR_FP32_PRE", "0", p), "head_fp32": _flag("NEMO_DIAR_FP32_HEAD", "0", p),
                "fp32_layers": [int(x) for x in _env("NEMO_DIAR_FP32_LAYERS", "", p).split(",") if x.strip()],
                "rope_fp32": _flag("NEMO_DIAR_FP32_ROPE", "0", p), "gelu_fp32": _flag("NEMO_DIAR_FP32_GELU", "0", p),
                "async_state": engine == "nemo" and _flag("NEMO_DIAR_ASYNC_STATE", "0", p)}

    def load(self):
        global _GRAPHS_AFTER_WARMUP
        import torch
        mm = torch.backends.cuda.matmul
        self._matmul_flags_before = {"tf32": mm.allow_tf32, "fp16_reduced": mm.allow_fp16_reduced_precision_reduction,
                                     "bf16_reduced": mm.allow_bf16_reduced_precision_reduction,
                                     "precision": torch.get_float32_matmul_precision()}
        if _flag("NEMO_DIAR_NO_REDUCED_REDUCTION", "1"):
            # split-K GEMMs must reduce partial sums in fp32 (small-batch bf16/fp16 kernels otherwise
            # carry a one-sided rounding bias into the speech probabilities; see BENCHMARK.md)
            mm.allow_fp16_reduced_precision_reduction = False
            mm.allow_bf16_reduced_precision_reduction = False
        logger.info("matmul flags before/after: %s -> reduced fp16=%s bf16=%s", self._matmul_flags_before,
                    mm.allow_fp16_reduced_precision_reduction, mm.allow_bf16_reduced_precision_reduction)
        profiles = [p.strip() for p in os.environ.get(
            "NEMO_DIAR_PROFILES", "offline,low,ultralow").split(",") if p.strip()]
        knobs = {p: self._knobs(p) for p in profiles}
        compile_on = any(k["compile_on"] for k in knobs.values())
        # b10cache: reuse torch.compile artifacts written by an earlier replica of this
        # deployment (b10-transfer keys on /cache/model), so only the first replica compiles.
        cache_status = self._load_compile_cache() if compile_on else None
        t_all = time.perf_counter()
        for p in profiles:
            t0 = time.perf_counter()
            self.add_instance(p, p, **knobs[p])
            logger.info("profile %s %s ready in %.1fs", p, knobs[p], time.perf_counter() - t0)
        self._default_profile = profiles[0]
        if cache_status is not None and not str(cache_status).endswith("SUCCESS"):
            self._save_compile_cache()
        _GRAPHS_AFTER_WARMUP = _compiled_graphs()
        logger.info("loaded profiles %s in %.1fs (micro-batch window=%.0f/%.0fms max_wait=%.0fms cap=%d, graph sizes "
                    "%s, cache=%s, compiled graphs=%d, nemo=%s)", knobs, time.perf_counter() - t_all, MB_WINDOW_S * 1000,
                    MB_FILL_S * 1000, MB_MAX_WAIT_S * 1000, MB_CAP, GRAPH_SIZES, cache_status, _GRAPHS_AFTER_WARMUP,
                    _nemo_build_id())

    def add_instance(self, name, profile, *, engine, dtype, compile_on, async_state, attn_mode=None, post_fp32=False,
                     pre_fp32=False, head_fp32=False, fp32_layers=(), rope_fp32=False, gelu_fp32=False,
                     fp8=False, autotune=False):
        """One model instance + batcher under `name` (== profile in the shipped truss; the lab loads variants)."""
        import torch
        m = self._build_instance(profile, dtype=dtype, engine=engine, compile_on=compile_on and engine == "nemo",
                                 async_state=async_state, post_fp32=post_fp32 and engine == "graphs",
                                 pre_fp32=pre_fp32 and engine == "graphs", head_fp32=head_fp32 and engine == "graphs",
                                 fp32_layers=fp32_layers if engine == "graphs" else (),
                                 rope_fp32=rope_fp32 and engine == "graphs", gelu_fp32=gelu_fp32 and engine == "graphs",
                                 fp8=fp8 and engine == "graphs")
        self._models[name] = m
        if engine == "graphs":
            from graph_runner import GraphRunner
            runner = GraphRunner(torch, m, dtype=getattr(torch, DTYPES[dtype]), graphs=True, sizes=_graph_sizes(profile),
                                 compile_encoder=compile_on, attn_mode=attn_mode, post_fp32=post_fp32,
                                 pre_fp32=pre_fp32, head_fp32=head_fp32, name=name)
            cfg = torch._inductor.config
            prev = (cfg.max_autotune_gemm, cfg.max_autotune_gemm_backends, cfg.force_disable_caches)
            if autotune:
                # GEMM autotune for this instance's compile (Inductor config is read at compile time; the
                # FX-graph cache does not key on it, so caches are off for this compile)
                cfg.max_autotune_gemm, cfg.max_autotune_gemm_backends = True, "TRITON,CUBLAS"
                cfg.force_disable_caches = True
                torch._dynamo.reset()                 # dynamo's per-code cache would serve another instance's graph
            try:
                runner.warm(seconds=WARM_S)
            finally:
                cfg.max_autotune_gemm, cfg.max_autotune_gemm_backends, cfg.force_disable_caches = prev
            self._runners[name] = runner
            self._batchers[name] = _Batcher(m, name, runner=runner)
        else:
            ac = None if dtype == "fp32" else getattr(torch, DTYPES[dtype])
            self._batchers[name] = _Batcher(m, name, autocast_dtype=ac)
        return m

    def _build_instance(self, profile: str, *, dtype: str, engine: str, async_state: bool, compile_on: bool,
                        post_fp32: bool = False, pre_fp32: bool = False, head_fp32: bool = False, fp32_layers=(),
                        rope_fp32: bool = False, gelu_fp32: bool = False, fp8: bool = False):
        """One model instance: restore, streaming params, weight cast; for the `nemo` engine also the
        reference script's optional fixed-shape state and torch.compile(encoder) with batch-shape warmup."""
        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel

        if compile_on:
            # Headroom over the default 8: hitting the limit silently drops the encoder
            # to eager for the life of the process.
            torch._dynamo.config.cache_size_limit = 64
        m = SortformerEncLabelModel.restore_from(
            restore_path=os.environ["NEMO_DIAR_MODEL_PATH"], map_location="cuda", strict=False)
        self._apply_profile(m, *PROFILES[profile])
        if async_state:
            # Fixed-shape streaming state inside diarize(): one compiled graph per batch-size
            # class, at the cost of running every step at full context capacity.
            m.async_streaming = True
            m.async_pad_to_max = True
        if dtype != "fp32":
            low = getattr(torch, DTYPES[dtype])
            if post_fp32 or pre_fp32 or head_fp32:
                # fp32 islands: only the encoder in the low dtype, plus pre-encode / head back to fp32
                m.encoder.to(dtype=low)
                if pre_fp32 or post_fp32:
                    m.encoder.pre_encode.float()
                if not (head_fp32 or post_fp32):
                    m.sortformer_modules.to(dtype=low)
            else:
                m = m.to(dtype=low)
            if fp32_layers:
                from graph_runner import fp32_layers as _fp32_layers
                _fp32_layers(torch, m.encoder, fp32_layers, low)
            if rope_fp32:
                from graph_runner import patch_rope_fp32
                patch_rope_fp32(torch, m.encoder)
            if gelu_fp32:
                from graph_runner import patch_gelu_fp32
                patch_gelu_fp32(torch, m.encoder)
            if fp8:
                from graph_runner import patch_linears_fp8
                patch_linears_fp8(torch, m.encoder)
        m.eval()
        if compile_on:
            self._compile_or_fallback(m, profile, None if dtype == "fp32" else getattr(torch, DTYPES[dtype]))
        return m

    @staticmethod
    def _load_compile_cache():
        try:
            from b10_transfer import load_compile_cache
            status = load_compile_cache()
            logger.info("torch.compile cache: %s", status)
            return status
        except Exception as e:  # noqa: BLE001 — caching is an optimisation only
            logger.warning("compile cache unavailable: %s", type(e).__name__)
            return None

    @staticmethod
    def _save_compile_cache():
        try:
            from b10_transfer import save_compile_cache
            logger.info("torch.compile cache saved: %s", save_compile_cache())
        except Exception as e:  # noqa: BLE001
            logger.warning("compile cache save failed: %s", type(e).__name__)

    def _apply_profile(self, m, spkcache, fifo, chunk, rc, update):
        sm = m.sortformer_modules
        sm.spkcache_len = spkcache
        sm.fifo_len = fifo
        sm.chunk_len = chunk
        sm.chunk_right_context = rc
        sm.spkcache_update_period = update
        try:
            m._check_streaming_parameters()
        except Exception as e:  # noqa: BLE001 — surface but don't fail on a preview API
            logger.warning("streaming param check (%s): %s", chunk, e)

    def _compile_or_fallback(self, m, profile, autocast_dtype):
        """torch.compile the encoder (~4x RTFx on streaming profiles).

        Warmup is the gate; on failure the instance reverts to eager. Profiles in
        NEMO_DIAR_COMPILE_SKIP (verylow trips an Inductor fusion bug and would burn ~30 s
        failing) stay eager. Encoder weights are identical across profiles, so with
        NEMO_DIAR_COMPILE_SHARE one compiled encoder serves them all.
        """
        import numpy as np
        import torch

        skip = {x.strip() for x in os.environ.get("NEMO_DIAR_COMPILE_SKIP", "").split(",")}
        if profile in skip:
            logger.info("profile %s: compile skipped (known Inductor failure); eager", profile)
            return
        share = _flag("NEMO_DIAR_COMPILE_SHARE", "0")
        orig = m.encoder
        try:
            if share and self._shared_enc is not None:
                m.encoder = self._shared_enc
            else:
                m.encoder = torch.compile(m.encoder, dynamic=True)
            # Micro-batching presents new batch shapes; the first call of each shape pays
            # ~30-40 s of compile. Pre-pay for the common batch sizes here.
            rng = np.random.default_rng(0)
            wavs = [(rng.standard_normal(SAMPLE_RATE * 30) * 0.01).astype(np.float32) for _ in range(MB_CAP)]
            ctx = (torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype is not None
                   else contextlib.nullcontext())
            sizes = sorted({n for n in (1, 4, 8, 16, MB_CAP) if n <= MB_CAP})
            with ctx:
                for n in sizes:
                    m.diarize(audio=wavs[:n], sample_rate=SAMPLE_RATE, batch_size=n, num_workers=0,
                              verbose=False)
            if share and self._shared_enc is None:
                self._shared_enc = m.encoder
            logger.info("profile %s: encoder compiled + warmed (bs %s)", profile, "/".join(map(str, sizes)))
        except Exception as e:  # noqa: BLE001 — Inductor bug on some shapes; eager is correct
            m.encoder = orig
            logger.warning("profile %s: compile failed (%s); running eager", profile, type(e).__name__)

    def _fetch_audio(self, audio: dict) -> bytes:
        if audio.get("audio_b64"):
            try:
                return base64.b64decode(audio["audio_b64"], validate=True)
            except (ValueError, TypeError) as e:
                raise _bad_request(f"audio_b64 is not valid base64: {e}") from e
        if audio.get("url"):
            try:
                with urllib.request.urlopen(audio["url"], timeout=120) as r:
                    return r.read()
            except (urllib.error.URLError, ValueError) as e:
                raise _bad_request(f"could not fetch audio.url: {e}") from e
        raise _bad_request("audio requires 'url' or 'audio_b64'")

    @staticmethod
    def _decode(raw: bytes):
        """Any container/codec -> (16 kHz mono float32 PCM in pinned host memory, seconds spent) via
        ffmpeg on pipes. Pinned so the batch's H2D copy is asynchronous and off the GPU thread's clock."""
        import torch
        t0 = time.perf_counter()
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1",
                            "-ar", str(SAMPLE_RATE), "pipe:1"], input=raw, capture_output=True)
        if p.returncode != 0 or len(p.stdout) < 2:
            tail = p.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["ffmpeg failed"]
            raise _bad_request(f"audio could not be decoded: {tail[0]}")
        i16 = torch.frombuffer(bytearray(p.stdout), dtype=torch.int16)
        try:
            pcm = torch.empty(i16.shape, dtype=torch.float32, pin_memory=True)
        except RuntimeError:  # no CUDA context in this thread yet, or pinned pool exhausted
            pcm = torch.empty(i16.shape, dtype=torch.float32)
        torch.div(i16, 32768.0, out=pcm)
        return pcm, time.perf_counter() - t0

    def predict(self, request: dict) -> dict:
        t_start = time.perf_counter()
        di = request.get("diarization_input")
        if not isinstance(di, dict) or not isinstance(di.get("audio"), dict):
            raise _bad_request("request must be {'diarization_input': {'audio': {...}}}")
        profile = di.get("latency") or self._default_profile
        if profile not in self._models:
            raise _bad_request(f"latency must be one of {list(self._models)}")
        raw = self._fetch_audio(di["audio"])
        t_fetched = time.perf_counter()
        # Enqueue first, decode while the batching window runs.
        decode_fut = self._decoder.submit(self._decode, raw)
        segments, info = self._batchers[profile].submit(decode_fut).result(timeout=1800)
        if isinstance(segments, concurrent.futures.Future):
            t_p = time.perf_counter()
            segments = segments.result(timeout=600)
            info["post_ms"] = round((time.perf_counter() - t_p) * 1000)

        turns = [self._to_turn(s) for s in segments]
        turns = [t for t in turns if t and t["end"] > t["start"]]
        timing = dict(info, fetch_ms=round((t_fetched - t_start) * 1000),
                      total_ms=round((time.perf_counter() - t_start) * 1000))
        return {"turns": turns, "segments": [str(s) for s in segments],
                "num_speakers": len({t["speaker"] for t in turns}),
                "latency": profile, "batch_n": info["batch_n"], "timing": timing}

    @staticmethod
    def _to_turn(seg):
        """Parse a NeMo diarize segment into {start, end, speaker}.

        Handles 'start end speaker' strings and (start, end, spk) / (spk, start,
        end) tuples/lists.
        """
        if isinstance(seg, str):
            p = seg.split()
            if len(p) < 3:
                return None
            return {"start": float(p[0]), "end": float(p[1]), "speaker": str(p[2])}
        if isinstance(seg, (list, tuple)) and len(seg) >= 3:
            a, b, c = seg[0], seg[1], seg[2]
            if isinstance(a, str):  # [spk, start, end]
                return {"start": float(b), "end": float(c), "speaker": str(a)}
            return {"start": float(a), "end": float(b), "speaker": str(c)}
        return None
