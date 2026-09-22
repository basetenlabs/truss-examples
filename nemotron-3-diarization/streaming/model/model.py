"""Streaming WebSocket truss for nvidia/Nemotron-3-Diarization-preview.

Live speaker diarization over a WebSocket: the client streams PCM16 mono 16 kHz audio
chunk-by-chunk; the server drives NeMo's streaming Sortformer loop with per-connection state and
emits speaker turns incrementally (replace-style partials + a final).

Request-level latency: the streaming knobs (chunk/right-context/fifo/spkcache) live on
shared module state, so one model instance cannot safely serve mixed profiles at once.
We therefore pre-load ONE instance per profile at load() and route each connection to
the instance matching its requested latency.

Throughput (``packages/xsession.py``): by default every profile instance is driven by ONE stepper
thread that batches the ready chunks of all its connections into one diarizer forward per tick
(NeMo's async fixed-shape state gives every row the same shape whatever its position; the core is
replayed from a CUDA graph per batch size, fp32, bit-identical to eager). Connections never touch
the GPU: the handler slices the raw window a chunk needs and awaits the tick. Outbound frames go
through a per-connection one-slot mailbox so a slow reader of the (growing, replace-style) partials
never stalls that connection's inbound audio. Turns are tracked incrementally on the CPU.

The mel preprocessor uses ``normalize: NA`` (no per-utterance normalization), so mel
frames are frame-local: we compute them on a trailing raw window with a few frames of
left context and get values identical to the whole-file path.

Wire protocol (client → server):
  {"latency": "low"}                                             # optional handshake first frame
  {"type": "input_audio_buffer.append", "audio": "<b64 pcm16>"}  # repeat
  {"type": "input_audio_buffer.commit"}                          # finalize + flush
server → client:
  {"type": "diarization", "is_final": false, "processed_s": t, "num_speakers": n,
   "turns": [{"start": s, "end": e, "speaker": "speaker_0"}, ...]}   # replace-style
  {"type": "diarization", "is_final": true, ...}                     # at commit
"""

import base64
import json
import logging
import os
import threading
import time

import numpy as np

from xsession import (
    FRAME_S,
    HOP,
    LPAD,
    SUB,
    WIN,
    Chunk,
    GpuSampler,
    Stepper,
    SyncGraphs,
    TurnTracker,
    env_bool,
    patch_attention_sdpa,
)

logger = logging.getLogger(__name__)

# (spkcache_len, fifo_len, chunk_len, chunk_right_context, spkcache_update_period), 80 ms frames.
PROFILES = {
    "offline": (264, 40, 340, 40, 300),   # 30.4 s buffer — best DER
    "low": (264, 264, 9, 4, 222),         # 1.04 s
    "verylow": (264, 264, 6, 2, 222),     # 0.64 s
    "ultralow": (264, 264, 3, 1, 222),    # 0.32 s
}
SR = 16000
DEFAULT_THRESHOLD = 0.5


def _nemo_build_id() -> str:
    """Resolved NeMo overlay commit (pip records it for VCS installs), for the startup log."""
    try:
        import importlib.metadata as md
        import json as _json
        info = _json.loads(md.distribution("nemo_toolkit").read_text("direct_url.json") or "{}")
        return f"{md.version('nemo_toolkit')}@{info.get('vcs_info', {}).get('commit_id', '?')[:12]}"
    except Exception:  # noqa: BLE001
        return "unknown"


class Model:
    def __init__(self, **kwargs):
        self._secrets = kwargs.get("secrets", {})
        self._models = {}
        self._steppers = {}
        self._sync = {}
        self._live = 0
        self._gpu = None
        self.prof = False
        # Event-loop (handler) accounting: frames handled, wall spent in the handler's synchronous
        # Python per frame / per chunk result, partial bytes posted; process CPU for GIL accounting.
        self._hstat = {"frames": 0, "frame_s": 0.0, "chunks": 0, "chunk_s": 0.0, "partials": 0, "bytes": 0}
        # Per-profile admission: live sessions are weighted in `low`-row-equivalents against one GPU
        # budget (predict_concurrency is a global WebSocket cap and cannot see the profile mix; an
        # all-`ultralow` replica saturates the GPU at ~250 while 560 `low` streams hold).
        self._budget = float(os.environ.get("NEMO_DIAR_MAX_LIVE_LOW", "560"))
        max_ul = float(os.environ.get("NEMO_DIAR_MAX_LIVE_ULTRALOW", "200"))
        self._weight = {"low": 1.0, "verylow": 1.5, "ultralow": self._budget / max(1.0, max_ul), "offline": 0.2}
        self._live_units = 0.0
        self._live_by_profile = {}
        self._rejected = 0
        self._t0 = time.time()
        self._cpu0 = time.process_time()

    def load(self):
        import concurrent.futures

        import torch
        from nemo.collections.asr.models import SortformerEncLabelModel

        self._torch = torch
        # The only CPU tensor work is the per-tick pinned-buffer copy; with the default thread count
        # PyTorch's OpenMP workers spin between ticks (3+ cores at 25 ticks/s) and starve the event loop.
        torch.set_num_threads(1)
        # The pool only builds messages (and runs the per-connection A/B paths); the batched stepper
        # path never parks a thread per connection.
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("NEMO_DIAR_WORKERS", "32")))
        path = os.environ["NEMO_DIAR_MODEL_PATH"]
        profiles = [p.strip() for p in os.environ.get(
            "NEMO_DIAR_PROFILES", "offline,low,ultralow").split(",") if p.strip()]
        self.xsession = env_bool("NEMO_DIAR_XSESSION", "1")
        self.sync_graphs = not self.xsession and env_bool("NEMO_DIAR_SYNC_GRAPHS", "0")
        compile_on = not self.xsession and not self.sync_graphs and env_bool("NEMO_DIAR_COMPILE", "1")
        # Fused stepper + Inductor: the encoder is torch.compile'd (dynamic batch) and the compiled kernels
        # are what the fused CUDA graph captures -- NeMo's reference gains ~1.7x at bs=32 from fusion.
        # Numerics: compiled kernels round differently from eager (A/B; DER-gated).
        fused_compile = self.xsession and env_bool("NEMO_DIAR_FUSED_COMPILE", "0")
        # Core dtype (fp32 | bf16): bf16 casts the encoder + Sortformer head weights once (NVIDIA's
        # reference path runs bf16); mel, the streaming state and NeMo's update stay fp32.
        core_dtype_name = os.environ.get("NEMO_DIAR_CORE_DTYPE", "fp32").strip().lower()
        core_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[core_dtype_name]
        # Fixed-shape streaming state (NeMo async mode, padded to capacity): the encoder sees one
        # sequence length for the whole session, which is what lets rows at different positions share
        # one batched call (and, in the per-connection compile path, keeps torch.compile at one graph).
        async_state = self.xsession or (not self.sync_graphs and env_bool("NEMO_DIAR_ASYNC_STATE", "1"))
        self.prof = env_bool("NEMO_DIAR_PROF", "0")
        core = os.environ.get("NEMO_DIAR_CORE", "graphs").strip().lower()
        b_max = int(os.environ.get("NEMO_DIAR_XSESSION_B", "32"))
        b_max_offline = int(os.environ.get("NEMO_DIAR_XSESSION_B_OFFLINE", "8"))
        wait_ms = float(os.environ.get("NEMO_DIAR_XSESSION_WAIT_MS", "0"))
        b_target = int(os.environ.get("NEMO_DIAR_XSESSION_B_TARGET", "16"))
        b_min = int(os.environ.get("NEMO_DIAR_XSESSION_B_MIN", "1"))
        tick_mode = os.environ.get("NEMO_DIAR_XSESSION_TICK", "fused").strip().lower()
        slots = int(os.environ.get("NEMO_DIAR_XSESSION_SLOTS", "640"))
        prof_sync = env_bool("NEMO_DIAR_PROF_SYNC", "0")
        if env_bool("NEMO_DIAR_TF32", "0"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        # cuBLAS may accumulate split-K partial sums in bf16/fp16 (both flags default True in this container);
        # split-K selection changes with the row count, i.e. a shape-dependent rounding bias. Off by default.
        rpr = env_bool("NEMO_DIAR_REDUCED_PRECISION_REDUCTION", "0")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = rpr
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = rpr
        self._shared_enc = None
        torch._dynamo.config.cache_size_limit = 64
        cache_status = self._load_compile_cache() if (compile_on or fused_compile) else None
        t_all = time.perf_counter()
        for p in profiles:
            m = SortformerEncLabelModel.restore_from(
                restore_path=path, map_location="cuda", strict=False)
            self._apply_profile(m, *PROFILES[p])
            m.async_streaming = async_state
            m.async_pad_to_max = async_state
            # The async FIFO update can randomise the first pop (meant to stagger offline batches);
            # a deterministic schedule identical to sync mode is what we want.
            m.sortformer_modules.async_desync_updates = False
            m.eval()
            try:
                m.preprocessor.featurizer.dither = 0.0  # deterministic mel across seams
            except Exception:  # noqa: BLE001
                pass
            self._models[p] = m
            t0 = time.perf_counter()
            if self.xsession:
                if os.environ.get("NEMO_DIAR_ATTN", "flex").strip().lower() == "sdpa":
                    patch_attention_sdpa(torch, m.encoder)
                if core_dtype != torch.float32:
                    for mod in (m.encoder, m.transformer_encoder, m.sortformer_modules):
                        if mod is not None:
                            mod.to(core_dtype)
                if fused_compile:
                    m.encoder = torch.compile(m.encoder, dynamic=True)
                st = Stepper(torch, m, p, core=core, b_max=b_max_offline if p == "offline" else b_max,
                             wait_ms=wait_ms, tick_mode=tick_mode, slots=slots, prof_sync=prof_sync,
                             b_target=b_target, core_dtype=core_dtype, b_min=b_min)
                st.start()
                self._steppers[p] = st
                self._warm(p)                  # drives the stepper through every chunk geometry
                n = st.warm_graphs()
                state = f"xsession tick={tick_mode} core={core} graphs={n} compiled_encoder={fused_compile} dtype={core_dtype_name}"
            elif self.sync_graphs:
                self._sync[p] = SyncGraphs(torch, m)
                self._warm(p)
                state = f"sync-graphs graphs={len(self._sync[p].graphs.graphs)}"
            else:
                state = self._compile_or_fallback(m, p, compile_on)
            logger.info("profile %s: %s in %.1fs", p, state, time.perf_counter() - t0)
        self._default_profile = profiles[0]
        if cache_status is not None and not str(cache_status).endswith("SUCCESS"):
            self._save_compile_cache()
        self._graphs_after_warmup = self._compiled_graphs()
        self._gpu = None
        if self.prof:
            self._gpu = GpuSampler(torch)
            self._gpu.start()
        if env_bool("NEMO_DIAR_BENCH", "0"):
            self._bench()
            for p in list(self._steppers):
                self._selfcheck(p)
        logger.info("loaded streaming profiles %s in %.1fs (xsession=%s, sync_graphs=%s, compile=%s, async_state=%s, "
                    "core_dtype=%s, reduced_precision_reduction=%s, cache=%s, gpu %.1f GB reserved, nemo=%s, steppers=%s)",
                    list(self._models), time.perf_counter() - t_all, self.xsession, self.sync_graphs, compile_on,
                    async_state, core_dtype_name, rpr, cache_status, torch.cuda.memory_reserved() / 2**30, _nemo_build_id(),
                    json.dumps({p: s.summary() for p, s in self._steppers.items()}))

    def _compiled_graphs(self) -> int:
        try:
            return int(self._torch._dynamo.utils.counters["stats"]["unique_graphs"])
        except Exception:  # noqa: BLE001
            return -1

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
        except Exception as e:  # noqa: BLE001
            logger.warning("streaming param check (%s): %s", chunk, e)

    def _warm_audio_frames(self, profile):
        """Mel frames of a synthetic session that walks every shape the live path can produce: both
        caches ramp from empty to capacity plus one cache update, then a flush tail with partial right
        context (and the padded first chunk at the start)."""
        sm = self._models[profile].sortformer_modules
        clc = getattr(sm, "chunk_left_context", 1)
        chunk, rc = sm.chunk_len, sm.chunk_right_context
        ramp = sm.spkcache_len + sm.fifo_len + sm.spkcache_update_period + clc + chunk + rc
        n_chunks = -(-ramp // chunk) + 2
        steady = (n_chunks * chunk + rc) * SUB
        tail = chunk * SUB + (rc * SUB) // 2
        return steady, tail

    def _warm(self, profile, audio=None):
        steady, tail = self._warm_audio_frames(profile)
        sess = _Session(self, profile)
        if audio is None:
            audio = np.zeros((steady + tail) * HOP + WIN, dtype=np.float32)
        sess.add_audio(audio)
        t0 = time.perf_counter()
        n = 0
        for flush in (False, True):
            for ch in sess.next_chunks(flush):
                sess.step_blocking(ch)
                n += 1
        sess.message_text(True, True)
        sess.close()
        return n, time.perf_counter() - t0

    def _compile_or_fallback(self, m, profile, compile_on):
        """Per-connection path (NEMO_DIAR_XSESSION=0): torch.compile the encoder, warmup-gated with an
        eager fallback; profiles in NEMO_DIAR_COMPILE_SKIP go straight to eager."""
        orig = m.encoder
        skip = {x.strip() for x in os.environ.get("NEMO_DIAR_COMPILE_SKIP", "").split(",")}
        share = os.environ.get("NEMO_DIAR_COMPILE_SHARE", "0") == "1"
        if compile_on and profile not in skip:
            try:
                if share and self._shared_enc is not None:
                    m.encoder = self._shared_enc
                else:
                    m.encoder = self._torch.compile(m.encoder, dynamic=True)
                self._warm(profile)
                if share and self._shared_enc is None:
                    self._shared_enc = m.encoder
                return "compiled"
            except Exception as e:  # noqa: BLE001 — eager is correct, just slower
                m.encoder = orig
                logger.warning("profile %s: compile failed (%s); running eager", profile, type(e).__name__)
        try:
            self._warm(profile)
            return "eager"
        except Exception as e:  # noqa: BLE001
            logger.warning("warmup %s failed: %s", profile, e)
            return "eager-unwarmed"

    def _bench(self):
        """NEMO_DIAR_BENCH=1: per-chunk step cost of the loaded path on the first profile, after the
        cache ramp (steady state), from 60 s of noise; plus the batched core alone at B=1..b_max."""
        torch = self._torch
        p = "low" if "low" in self._models else self._default_profile
        rng = np.random.default_rng(0)
        steady, tail = self._warm_audio_frames(p)
        audio = (rng.standard_normal((steady + tail) * HOP + WIN) * 0.02).astype(np.float32)
        sess = _Session(self, p)
        sess.add_audio(audio)
        chunks = sess.next_chunks(False)
        times = []
        for ch in chunks:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            sess.step_blocking(ch)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        sess.close()
        tail_n = max(1, len(times) // 3)
        logger.info("BENCH %s per-chunk step ms (B=1, whole path incl. mel): all p50 %.2f, steady p50 %.2f p95 %.2f (n=%d)",
                    p, float(np.median(times)), float(np.median(times[-tail_n:])),
                    float(np.percentile(times[-tail_n:], 95)), len(times))
        st = self._steppers.get(p)
        if st is not None and st.tick_mode == "fused":
            rows = st.bench_fused()
            logger.info("BENCH %s fused tick (mel+core+update+writeback, graph replay): %s", p,
                        "; ".join(f"B={b}: {ms} ms/tick, {per} ms/row" for b, ms, per in rows))
        elif st is not None:
            sm = st.sm
            rows = []
            with torch.inference_mode():
                for b in sorted({1, 2, 4, 8, 16, st.b_max}):
                    if b > st.b_max:
                        continue
                    s = sm.init_streaming_state(batch_size=b, async_streaming=True, device=st.device)
                    feats = torch.randn((b, st.f_steady, 128), device=st.device)
                    flen = torch.full((b,), st.f_steady, dtype=torch.int64, device=st.device)
                    for _ in range(3):
                        st.run_core(feats, flen, s.spkcache, s.spkcache_lengths, s.fifo, s.fifo_lengths)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(20):
                        st.run_core(feats, flen, s.spkcache, s.spkcache_lengths, s.fifo, s.fifo_lengths)
                    torch.cuda.synchronize()
                    ms = (time.perf_counter() - t0) / 20 * 1000
                    rows.append(f"B={b}: {ms:.2f} ms/tick, {ms / b:.2f} ms/row")
            logger.info("BENCH %s batched core (%s): %s", p, st.core_mode, "; ".join(rows))

    @staticmethod
    def _synthetic_speech(seconds, rng):
        """Two alternating harmonic 'talkers' (f0 120 / 220 Hz, 4 Hz syllabic AM, 3 s turns) plus noise:
        enough voiced structure for the model to fire on, so the self-check exercises the speaker
        cache and its compression, not only silence."""
        t = np.arange(int(seconds * SR)) / SR
        sig = np.zeros_like(t)
        for f0, phase in ((120.0, 0), (220.0, 1)):
            gate = ((t // 3.0).astype(int) % 2 == phase).astype(np.float32)
            am = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
            voiced = sum(np.sin(2 * np.pi * f0 * k * t + rng.random()) / k for k in range(1, 12))
            sig += gate * am * voiced
        sig = 0.15 * sig / np.abs(sig).max() + 0.003 * rng.standard_normal(len(t))
        return sig.astype(np.float32)

    def _selfcheck(self, p):
        """NEMO_DIAR_BENCH=1: the fused graph tick vs the legacy eager tick (NeMo's own update) on one
        synthetic 100 s session, chunk by chunk: max |Δ| of the chunk predictions and turn equality."""
        st = self._steppers.get(p)
        if st is None or st.tick_mode != "fused":
            return
        ref = Stepper(self._torch, self._models[p], p, core="eager", b_max=1, tick_mode="legacy", core_dtype=st.core_dtype)
        ref.start()
        audio = self._synthetic_speech(100.0, np.random.default_rng(1))
        sa, sb = _Session(self, p), _Session(self, p)
        sb.stepper = ref
        sa.add_audio(audio)
        sb.add_audio(audio)
        diffs, active = [], []
        for flush in (False, True):
            for x, y in zip(sa.next_chunks(flush), sb.next_chunks(flush)):
                for s_, c_ in ((st, x), (ref, y)):
                    c_.done = threading.Event()
                    s_.submit(c_)
                    c_.done.wait()
                    if c_.error is not None:
                        raise c_.error
                diffs.append(float(np.abs(x.result - y.result).max()))
                active.append(float((x.result > 0.5).mean()))
                sa._absorb(x, x.result)
                sb._absorb(y, y.result)
        same = sa.tracker.turns() == sb.tracker.turns()
        logger.info("SELFCHECK %s fused-graph vs legacy-eager: %d chunks, max|d| %.2e, mean|d| %.2e, active frac %.3f, "
                    "turns identical=%s (%d turns)", p, len(diffs), max(diffs), float(np.mean(diffs)),
                    float(np.mean(active)), same, len(sa.tracker.turns()))
        sa.close()
        sb.close()

    # ------------------------------------------------------------------ WebSocket handler
    async def websocket(self, ws):
        import asyncio

        loop = asyncio.get_event_loop()
        sess = None
        # Partials are replace-style, so a partial the socket has not taken yet is superseded by the
        # next one: outbound frames go through a one-slot mailbox drained by a sender task, and
        # receiving never waits on sending. Otherwise a slow reader of the growing partials stalls this
        # connection's next audio frames behind ws.send_text. Finals and errors are never dropped.
        box = {"partial": None, "must": [], "dropped": 0, "closed": False}
        wake = asyncio.Event()

        async def sender():
            while True:
                await wake.wait()
                wake.clear()
                while box["must"] or box["partial"] is not None:
                    if box["must"]:
                        await ws.send_text(box["must"].pop(0))
                    elif box["partial"] is not None:
                        text, box["partial"] = box["partial"], None
                        await ws.send_text(text)
                if box["closed"]:
                    return

        def post(text, must=False):
            if must:
                box["must"].append(text)
            else:
                if box["partial"] is not None:
                    box["dropped"] += 1
                box["partial"] = text
            wake.set()

        sender_task = loop.create_task(sender())

        async def close_with(text):
            box["partial"] = None
            post(text, must=True)
            box["closed"] = True
            wake.set()
            await sender_task

        hs = self._hstat
        try:
            while True:
                frame = await ws.receive_text()
                t_f = time.perf_counter()
                try:
                    data = json.loads(frame)
                    if not isinstance(data, dict):
                        raise ValueError("frame must be a JSON object")
                    mtype = data.get("type")
                    if sess is None:
                        profile = data.get("latency") or self._default_profile
                        if profile not in self._models:
                            raise ValueError(f"latency must be one of {list(self._models)}")
                        threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
                        if not 0.0 < threshold < 1.0:
                            raise ValueError("threshold must be in (0, 1)")
                        cost = self._weight.get(profile, 1.0)
                        if self._live_units + cost > self._budget + 1e-9:
                            self._rejected += 1
                            logger.info("capacity: rejected %s session (%.0f/%.0f low-equivalents live: %s)", profile,
                                        self._live_units, self._budget, self._live_by_profile)
                            await close_with(json.dumps({"type": "error", "error": (
                                f"capacity: this replica is at its GPU budget for {profile} sessions "
                                f"({self._live_units:.0f}/{self._budget:.0f} low-equivalent streams live, "
                                f"{profile} counts {cost:.1f}); retry on another replica")}))
                            return
                        sess = _Session(self, profile, threshold=threshold)
                        self._live += 1
                        self._live_units += cost
                        self._live_by_profile[profile] = self._live_by_profile.get(profile, 0) + 1
                        if mtype not in ("input_audio_buffer.append", "input_audio_buffer.commit"):
                            continue  # bare handshake frame
                    pcm = None
                    if mtype == "input_audio_buffer.append" and data.get("audio"):
                        pcm = np.frombuffer(base64.b64decode(data["audio"], validate=True), dtype=np.int16)
                except (ValueError, TypeError) as e:  # malformed client frame: tell them, then close
                    await close_with(json.dumps({"type": "error", "error": str(e)}))
                    return
                if pcm is not None:
                    sess.add_audio(pcm.astype(np.float32) / 32768.0)
                    chunks = sess.next_chunks(False)
                    hs["frames"] += 1
                    hs["frame_s"] += time.perf_counter() - t_f
                    for ch in chunks:
                        await sess.step_async(ch, loop)
                    if chunks:
                        # Building the partial is O(uncommitted turns) now, so it stays on the loop: a
                        # thread hop per chunk cost more than the message.
                        t_c = time.perf_counter()
                        text = sess.message_text(False)
                        post(text)
                        hs["chunks"] += len(chunks)
                        hs["partials"] += 1
                        hs["bytes"] += len(text)
                        hs["chunk_s"] += time.perf_counter() - t_c
                elif mtype == "input_audio_buffer.commit":
                    for ch in sess.next_chunks(True):
                        await sess.step_async(ch, loop)
                    await close_with(sess.message_text(True, True))
                    return
        except Exception as e:  # noqa: BLE001
            # Client disconnects land here and are routine; anything else is a server fault
            # (e.g. CUDA) and must be visible in logs, with a best-effort error frame.
            if "Disconnect" in type(e).__name__ or "ConnectionClosed" in type(e).__name__:
                logger.info("ws session ended: %s", type(e).__name__)
                return
            logger.exception("ws session failed")
            try:
                await close_with(json.dumps({"type": "error", "error": f"internal error: {type(e).__name__}"}))
            except Exception:  # noqa: BLE001 — socket already gone
                pass
        finally:
            if not sender_task.done():
                sender_task.cancel()
            if sess is not None:
                self._live = max(0, self._live - 1)
                self._live_units = max(0.0, self._live_units - self._weight.get(sess.profile, 1.0))
                self._live_by_profile[sess.profile] = max(0, self._live_by_profile.get(sess.profile, 1) - 1)
                sess.close()
                st = self._steppers.get(sess.profile)
                logger.info("session %s: %d chunks, %.1fs audio, %d live, partials_coalesced=%d, handler=%s, stepper=%s",
                            sess.profile, sess.n_chunks, sess.stt_feat * FRAME_S, self._live, box["dropped"],
                            json.dumps(self.handler_stats()), json.dumps(st.summary()) if st is not None else "off")

    def handler_stats(self):
        """Event-loop cost per frame / per chunk and whole-process CPU (all threads), for GIL accounting."""
        hs = self._hstat
        up = max(1e-9, time.time() - self._t0)
        return {"frames": hs["frames"], "frame_us": round(hs["frame_s"] / max(1, hs["frames"]) * 1e6, 1),
                "chunks": hs["chunks"], "partial_us": round(hs["chunk_s"] / max(1, hs["partials"]) * 1e6, 1),
                "partial_kb": round(hs["bytes"] / max(1, hs["partials"]) / 1024, 1),
                "loop_busy_frac": round((hs["frame_s"] + hs["chunk_s"]) / up, 3),
                "process_cores": round((time.process_time() - self._cpu0) / up, 2), "live": self._live,
                "live_units": round(self._live_units, 1), "budget": self._budget, "by_profile": dict(self._live_by_profile),
                "rejected": self._rejected}


class _Session:
    """Per-connection state: raw audio window, chunk geometry, streaming state, incremental turns."""

    def __init__(self, model: "Model", profile: str, threshold: float = DEFAULT_THRESHOLD):
        self.m = model
        self.torch = model._torch
        self.profile = profile
        self.model = model._models[profile]
        self.stepper = model._steppers.get(profile)
        self.syncg = model._sync.get(profile)
        sm = self.model.sortformer_modules
        self.chunk = sm.chunk_len
        self.rc = sm.chunk_right_context
        self.clc = getattr(sm, "chunk_left_context", 1)
        self.f_steady = (self.clc + self.chunk + self.rc) * SUB
        # Stepper rows get their (async) state lazily in the stepper thread; the per-connection paths
        # build theirs here. total_preds only exists on the committed per-connection path.
        self.state = None
        self.total_preds = None
        if self.stepper is None:
            self.state = sm.init_streaming_state(
                batch_size=1, async_streaming=getattr(self.model, "async_streaming", False), device=self.model.device)
            if self.syncg is None:
                self.total_preds = self.torch.zeros((1, 0, sm.n_spk), device=self.model.device)
        self.slot = None   # fused stepper: row in the profile's state tables (acquired on the first chunk)
        self.closed = False
        self.tracker = TurnTracker(sm.n_spk, threshold)
        self.raw = np.zeros(0, dtype=np.float32)
        self.raw_off = 0   # absolute sample index of raw[0]; consumed audio is dropped
        self.pending = []  # frames received since the last chunk (joined only when a chunk is ready)
        self.n_pending = 0
        self.stt_feat = 0  # next chunk start, in mel frames
        self.n_chunks = 0
        self.recs = []
        self._gpu_t = time.time()

    def add_audio(self, samples: np.ndarray):
        self.pending.append(samples)
        self.n_pending += len(samples)

    def _avail_frames(self):
        return max(0, (self.raw_off + len(self.raw) + self.n_pending - WIN) // HOP + 1)

    def _flush_pending(self):
        if self.pending:
            self.raw = np.concatenate([self.raw] + self.pending)
            self.pending, self.n_pending = [], 0

    def close(self):
        """Release the fused stepper's slot (no-op for the per-connection paths / before the first chunk)."""
        if self.stepper is not None and not self.closed:
            self.closed = True
            self.stepper.release(self)

    def _trim(self):
        """Keep only the trailing window the next chunk's mel needs (bounded buffer)."""
        keep_from = max(0, (self.stt_feat - self.clc * SUB - LPAD) * HOP)
        if keep_from > self.raw_off:
            self.raw = self.raw[keep_from - self.raw_off:]
            self.raw_off = keep_from

    def next_chunks(self, flush: bool):
        """Every fully-available chunk (or the tail on flush) as ``Chunk`` descriptors: raw window +
        geometry, no GPU work. A short chunk (the flush tail; the first chunk when the profile has
        encoder left context) is right-padded to the steady shape by the stepper, the padding counted
        as masked right context and the true length passed, so the core sees one shape per profile."""
        out = []
        avail = self._avail_frames()
        while True:
            if not flush:
                end_feat = self.stt_feat + self.chunk * SUB
                right_off = self.rc * SUB
                if avail < end_feat + right_off:
                    break
            else:
                if self.stt_feat >= avail:
                    break
                end_feat = min(self.stt_feat + self.chunk * SUB, avail)
                right_off = min(self.rc * SUB, avail - end_feat)
            left_off = min(self.clc * SUB, self.stt_feat)
            a, b = self.stt_feat - left_off, end_feat + right_off
            disc = min(a, LPAD)
            start = (a - disc) * HOP
            self._flush_pending()
            # Window end = the bound _avail_frames guarantees; frame b-1's centered STFT reaches only
            # (b-1)*HOP + 256, so the frames are exact and every row of a group has the same length.
            seg = np.ascontiguousarray(self.raw[start - self.raw_off: (b - 1) * HOP + WIN - self.raw_off])
            f_true = b - a
            pad = self.f_steady - f_true
            out.append(Chunk(self, seg, a, b, left_off, right_off + max(0, pad), f_true, disc, flush))
            self.stt_feat = end_feat
            self._trim()
        return out

    # ------------------------------------------------------------------ step dispatch
    async def step_async(self, ch: Chunk, loop):
        if self.stepper is not None:
            ch.loop, ch.fut = loop, loop.create_future()
            self.stepper.submit(ch)
            preds = await ch.fut
        else:
            preds = await loop.run_in_executor(self.m._pool, self._step_local, ch)
        self._absorb(ch, preds)

    def step_blocking(self, ch: Chunk):
        if self.stepper is not None:
            ch.done = threading.Event()
            self.stepper.submit(ch)
            ch.done.wait()
            if ch.error is not None:
                raise ch.error
            preds = ch.result
        else:
            preds = self._step_local(ch)
        self._absorb(ch, preds)

    def _absorb(self, ch: Chunk, preds):
        self.tracker.add(preds)
        self.n_chunks += 1
        if self.m.prof:
            self.recs.append({"step": self.n_chunks, "t": round(time.time(), 3), "b": ch.batch,
                              "wait": round((ch.t_start - ch.t_submit) * 1000, 3) if ch.t_start else 0.0,
                              "step_wall": round((ch.t_end - ch.t_start) * 1000, 3) if ch.t_end else 0.0,
                              "live": self.m._live})

    def _feats(self, ch: Chunk):
        """Mel for one chunk on the caller's thread (per-connection paths only), padded to steady shape."""
        torch = self.torch
        sig = torch.as_tensor(ch.seg, dtype=torch.float32, device="cuda")[None, :]
        ln = torch.tensor([sig.shape[1]], device="cuda")
        with torch.inference_mode():
            feats, _ = self.model.preprocessor(input_signal=sig, length=ln)
        feats_t = feats[:, :, ch.disc: ch.disc + (ch.b - ch.a)].transpose(1, 2).contiguous()   # [1, F, 128]
        if ch.f_true < self.f_steady:
            feats_t = torch.nn.functional.pad(feats_t, (0, 0, 0, self.f_steady - ch.f_true))
        return feats_t, torch.tensor([ch.f_true], device=self.model.device)

    def _step_local(self, ch: Chunk):
        """Per-connection A/B paths: sync-mode per-length graphs, or the committed compile path."""
        t0 = time.perf_counter()
        ch.t_start, ch.batch = t0, 1
        feats_t, flen = self._feats(ch)
        if self.syncg is not None:
            self.state, preds = self.syncg.step(feats_t, flen, self.state, ch.left_off, ch.right_off)
        else:
            n0 = self.total_preds.shape[1]
            with self.torch.inference_mode():
                self.state, self.total_preds = self.model.forward_streaming_step(
                    processed_signal=feats_t, processed_signal_length=flen,
                    streaming_state=self.state, total_preds=self.total_preds,
                    left_offset=ch.left_off, right_offset=ch.right_off)
            preds = self.total_preds[0, n0:].float().cpu().numpy()
            base = getattr(self.m, "_graphs_after_warmup", None)
            graphs = self.m._compiled_graphs() if base is not None else -1
            if base is not None and graphs > base:
                logger.warning("torch.compile recompiled under traffic: unique_graphs %d -> %d (chunk f_true=%d "
                               "left=%d right=%d)", base, graphs, ch.f_true, ch.left_off, ch.right_off)
                self.m._graphs_after_warmup = graphs
        ch.t_end = time.perf_counter()
        return preds

    # ------------------------------------------------------------------ output
    def message_text(self, is_final: bool, end_of_audio: bool = False) -> str:
        """The frame as JSON text; the turns array comes pre-serialised from the tracker (same bytes as
        json.dumps of the whole message would give, without re-encoding thousands of turns per partial)."""
        turns, n_spk = self.tracker.turns_json()
        head = {"type": "diarization", "is_final": is_final,
                "processed_s": round(self.stt_feat * FRAME_S, 2), "num_speakers": n_spk}
        tail = {}
        if end_of_audio:
            tail["is_end_of_audio_flush"] = True
        if self.m.prof:
            tail["prof"], self.recs = self.recs, []
            if self.m._gpu is not None:
                samples = self.m._gpu.since(self._gpu_t)
                if samples:
                    self._gpu_t = samples[-1][0]
                tail["gpu"] = samples
        text = json.dumps(head)[:-1] + ', "turns": ' + turns
        return text + (", " + json.dumps(tail)[1:] if tail else "}")
