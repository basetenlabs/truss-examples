"""Cross-session batching for the multitalker streaming truss (``MT_XSESSION=1``).

The committed build runs one NeMo step per connection under a replica-wide lock: ~12.7 ms of
lock-held time per 1.12 s chunk, so the replica saturates at ~75 real-time streams whatever the
GPU is doing (it is 60-70 % busy at that point, most of it launch latency at batch 1). This module
replaces the lock with ONE stepper thread that takes every session that has a full chunk ready and
runs the whole step for all of them in one pass:

  * one log-mel call per tick (``MT_XMEL``): connections hand the stepper the raw window their next
    chunk needs (numpy only, no GPU work on the connection side); rows of the same geometry share one
    preprocessor call and one per-chunk normalisation (the same tensor ``MelChunker.next_chunk``
    yields row by row; ``MT_XMEL_VERIFY=1`` checks that per row).
  * one diarizer call for B sessions (``MT_XDIAR=1``): every session's Sortformer state lives in a
    preallocated slot of ``[slots, ...]`` tables in NeMo's *async* layout (fixed ``[spkcache 264 |
    fifo 264 | chunk 14]`` with per-row lengths -- the per-row-length form of the sync update). A tick
    is one CUDA-graph replay per batch size covering gather by slot -> pre-encode -> concat+pad ->
    encoder -> head -> NeMo's async cache/FIFO update (speaker-cache compression computed for every
    row and selected with ``where``, so there is no host sync inside) -> speaker gate -> write-back
    by slot. Nothing is allocated per tick, so the footprint is the tables (~1.1 MB per slot).
    ``MT_XDIAR=0`` keeps the committed per-row sync-mode graph path (bit-identical to the lock build).
  * one gate: the last 28 diarizer frames of every row -> one ``> 0.5`` -> one D2H copy.
  * one ASR encoder graph replay + one batched RNNT decode over the SigmaS (session, speaker)
    rows of the tick, padded to the smallest of ``MT_XSESSION_ROW_BUCKETS`` (8/16/32; one encoder
    graph each) so the shapes the GEMMs see are fixed per bucket.
  * per-row cache write-back, per-row seglst update with a per-session chunk offset.

Connections await an asyncio future the stepper resolves from its thread (one loop wake-up per
tick), so a live session parks no pool thread; load-time warm sessions use a blocking future.

What NeMo's lockstep path could not do (NEMO_MULTITALKER_ARCH.md s.4) is done here per row:
the step-0 reset happens at admission, ``_offset_chunk_start_time`` lives on the session, the
sync-mode shape decisions are replaced by the async per-row-length update, ``total_preds`` is
one tensor per session.

Scheduling: the stepper drains whatever is queued when it finishes a tick (plus an optional
``MT_XSESSION_WAIT_MS`` coalescing window). At low load B is 1 and lag equals the step time; as
arrivals outpace the step the queue forms batches by itself, so the per-session-chunk cost falls
with load instead of the queue growing without bound.
"""

import collections
import json
import logging
import math
import os
import queue
import threading
import time
import types
from concurrent.futures import Future

import numpy as np

import mt_profiling as P

logger = logging.getLogger(__name__)

LIVE_SAFE = ("xsession_wait_ms", "xsession_b", "xdiar_rows", "diar_pad_rows")   # re-read every tick; flip under traffic

STATE_FIELDS = ("spkcache", "spkcache_preds", "spkcache_lengths", "spkcache_compressed", "fifo",
                "fifo_lengths", "mean_sil_emb", "n_sil_frames")


def _env_bool(name, default):
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "off", "no")


class XFlags:
    """Cross-session knobs, attached as attributes onto ``mt_fast.Flags`` (so ``as_dict`` shows them)."""

    @staticmethod
    def attach(flags):
        flags.xsession = _env_bool("MT_XSESSION", "1")
        flags.xsession_b = int(os.environ.get("MT_XSESSION_B", "16"))              # sessions per tick, max
        flags.xsession_rows = int(os.environ.get("MT_XSESSION_ROWS", "32"))        # largest ASR row count (64 faults at load)
        flags.xsession_wait_ms = float(os.environ.get("MT_XSESSION_WAIT_MS", "0"))  # coalescing window
        flags.xdiar = _env_bool("MT_XDIAR", "0")                # 1 = slot-table batched async-layout diarizer
        flags.xdiar_rows = int(os.environ.get("MT_XDIAR_ROWS", "0"))   # pin the diarizer batch to a constant B
        flags.xslots = int(os.environ.get("MT_XSLOTS", "320"))          # diarizer state slots (live sessions)
        # MT_XDIAR=0: run each session's sync-mode diarizer call at this many rows (the row duplicated),
        # so the FlexAttention kernel inside the core is the >= 2-row Inductor variant, not the bs=1 one.
        flags.diar_pad_rows = int(os.environ.get("MT_DIAR_PAD_ROWS", "1"))
        flags.xmel = _env_bool("MT_XMEL", "1")                  # mel in the stepper, one call per tick
        flags.xmel_verify = _env_bool("MT_XMEL_VERIFY", "0")    # compare every row with the per-row mel
        flags.xsession_row_buckets = XFlags.parse_buckets(
            os.environ.get("MT_XSESSION_ROW_BUCKETS", "8,16,32"), flags.xsession_rows)
        if flags.xsession:
            # The pad wrapper reads flags.pad_rows at call time; the stepper sets it per tick to the
            # smallest bucket that holds the tick's SigmaS rows (each bucket = one encoder graph).
            flags.pad_rows = flags.xsession_rows
        return flags

    @staticmethod
    def parse_buckets(s, rows):
        b = sorted({int(x) for x in str(s).replace(";", ",").split(",") if x.strip() and int(x) > 0})
        if b and b[-1] < rows:
            b.append(rows)
        return [x for x in b if x <= rows]


def warm_asr_buckets(fast):
    """One encoder graph per ASR row bucket, smallest first (each bucket names itself in the log).

    A 64-row bucket is not warmable on this NeMo build: the first 64-row ASR call (encoder graph
    capture + decoder state at that width) hits ``cudaErrorIllegalAddress`` -- fp32 or bf16, as the
    first bucket or after 8/16/32 (reproduced on the batch preset in fp32), while 8/16/32 never have.
    Hence the 32-row cap; SigmaS above it runs as 32-row slabs.
    """
    flags = fast.flags
    rows = flags.pad_rows
    warmed = []
    for b in sorted(set(flags.xsession_row_buckets) | {rows}):
        flags.pad_rows = b
        logger.info("ASR warm: bucket %d rows", b)
        fast.warm_asr()
        warmed.append(b)
    flags.pad_rows = rows
    return warmed


def apply_xcontrol(flags, warmed_buckets, ctl):
    """Apply the xsession keys of a control handshake; returns the remaining keys for FastPath.

    ``LIVE_SAFE`` knobs are plain attributes the stepper re-reads every tick. ``xdiar`` changes the
    per-session state layout and ``row_buckets`` needs captured graphs, so those go through
    ``FastPath.apply_control``'s no-live-session gate (returned here, applied by the caller).
    """
    for k in LIVE_SAFE:
        if k in ctl:
            v = ctl[k]
            setattr(flags, k, float(v) if k == "xsession_wait_ms" else max(0, int(v)))
    rest = {k: v for k, v in ctl.items() if k not in LIVE_SAFE}
    if "row_buckets" in rest:
        want = XFlags.parse_buckets(rest["row_buckets"], flags.xsession_rows) if str(rest["row_buckets"]) not in ("0", "") else []
        missing = [b for b in want if b not in warmed_buckets]
        if missing:
            raise ValueError(f"row_buckets {missing} were not warmed at load (have {warmed_buckets})")
        rest["_row_buckets"] = want
    return rest


class _Item:
    """One chunk of one session as handed to the stepper: the raw window (``seg``, numpy) and its mel
    geometry, or -- legacy -- ready features (``chunk``)."""

    __slots__ = ("sess", "seg", "geom", "chunk", "last", "fut", "loop", "afut", "t_submit", "active", "err")

    def __init__(self, sess, seg, geom, chunk, last):
        self.sess, self.seg, self.geom, self.chunk, self.last = sess, seg, geom, chunk, last
        self.fut = self.loop = self.afut = None
        self.t_submit = time.perf_counter()
        self.active = []
        self.err = None


def _resolve_many(items):
    for it in items:
        if it.afut.done():
            continue
        if it.err is not None:
            it.afut.set_exception(it.err)
        else:
            it.afut.set_result(True)


def finish_all(items):
    """Resolve a tick's items: one loop wake-up per asyncio loop, one future per blocking waiter."""
    by_loop = {}
    for it in items:
        if it.loop is not None:
            by_loop.setdefault(it.loop, []).append(it)
        elif it.fut is not None:
            if it.err is not None:
                it.fut.set_exception(it.err)
            else:
                it.fut.set_result(True)
    for loop, its in by_loop.items():
        try:
            loop.call_soon_threadsafe(_resolve_many, its)
        except RuntimeError:   # loop closed (shutdown)
            pass


class GraphCache:
    """CUDA-graph replay of a pure function of tensors, keyed by the input shapes.

    Same recipe as ``mt_fast.DiarGraphs``: one eager dry run on a fresh key (kernel selection),
    capture into a private graph with static input buffers, then ``copy_`` + replay. Outputs are
    the graph's static output buffers (valid until the next replay of the same key). The function
    may read and write preallocated module tensors (the slot tables): they sit at fixed addresses,
    so the captured kernels see them on every replay.
    """

    def __init__(self, torch, fn, name, device):
        self.torch, self.fn, self.name, self.device = torch, fn, name, device
        self.graphs = {}
        self.disabled = False
        self.replays = 0
        self.eager_calls = 0

    def run(self, *inputs):
        torch = self.torch
        key = tuple((tuple(t.shape), str(t.dtype)) for t in inputs)
        g = self.graphs.get(key)
        if g is None:
            if self.disabled or torch.cuda.is_current_stream_capturing():
                self.eager_calls += 1
                return self.fn(*[t.to(self.device) for t in inputs])
            try:
                g = self._capture(key, inputs)
            except Exception:  # noqa: BLE001 - eager is correct, just slower
                logger.exception("%s graph capture failed for %s; running eager from now on", self.name, key)
                self.disabled = True
                return self.fn(*[t.to(self.device) for t in inputs])
        for buf, t in zip(g["in"], inputs):
            buf.copy_(t, non_blocking=True)
        g["graph"].replay()
        self.replays += 1
        return g["out"]

    def _capture(self, key, inputs):
        torch = self.torch
        device = self.device
        static = [t.to(device).clone() for t in inputs]
        outs = self.fn(*static)                       # dry run: shapes + kernel warm-up
        out_bufs = [torch.empty_like(o) for o in outs]
        del outs
        torch.cuda.synchronize(device)
        s = torch.cuda.Stream(device)
        s.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s, capture_error_mode="thread_local"):
            outs = self.fn(*static)
            for b, o in zip(out_bufs, outs):
                b.copy_(o)
        torch.cuda.current_stream(device).wait_stream(s)
        g = {"graph": graph, "in": static, "out": tuple(out_bufs)}
        self.graphs[key] = g
        logger.info("%s graph captured for %s (%d graphs)", self.name, key[0][0], len(self.graphs))
        return g


class XStepper(threading.Thread):
    """The batched stepper: owns the queue, the thread, the slot tables and the per-tick gather/scatter."""

    def __init__(self, torch, fast, flags, live_fn, pre=None, norm_type=None):
        super().__init__(daemon=True, name="xstepper")
        from nemo.collections.asr.modules.sortformer_modules import StreamingSortformerState
        from nemo.collections.asr.parts.preprocessing.features import normalize_batch

        self.torch, self.fast, self.flags = torch, fast, flags
        self._State = StreamingSortformerState
        self._normalize_batch = normalize_batch
        self.asr, self.diar = fast.asr, fast.diar
        self.sm = self.diar.sortformer_modules
        # FastPath patches init_streaming_state to force the sync layout; the async layout needs the original.
        self._init_state = fast._orig["init_state"]
        self.device = fast.device
        self.live_fn = live_fn
        self.pre, self.norm_type = pre, norm_type            # shared mel preprocessor (MT_XMEL)
        self.q = queue.SimpleQueue()
        scfg = self.asr.encoder.streaming_cfg
        self.cache_frames = scfg.pre_encode_cache_size[1]      # 9 ASR-only pre-encode cache frames
        self.chunk_frames = scfg.chunk_size[1]                 # 112
        self.drop = scfg.drop_extra_pre_encoded                 # 2
        self.n_spk = self.sm.n_spk
        self.l_max = (self.sm.spkcache_len + self.sm.fifo_len + self.sm.chunk_left_context + self.sm.chunk_len
                      + self.sm.chunk_right_context)
        self.lengths = torch.full((1,), self.chunk_frames + self.cache_frames, dtype=torch.int64, device=self.device)
        self.ticks = 0
        self.rows_served = 0
        self.b_hist = collections.Counter()
        self.k_hist = collections.Counter()
        self.t_busy = 0.0
        self.t_started = time.time()
        self.tick_ms = collections.deque(maxlen=4096)   # recent tick walls (always on; summary p50/p99)
        self.phase_t = collections.Counter()             # wall per phase (perf_counter marks, no sync)
        self.xmel_stats = {"rows": 0, "equal": 0, "groups": 0}
        self._lock = threading.Lock()   # serialises admit()/release() against the tick (shared module reads)
        # --- diarizer slot tables (MT_XDIAR=1): allocated once, so the flag can be flipped by control
        self.n_slots = max(1, flags.xslots)
        self._alloc_tables(self.n_slots + max(flags.xsession_b, 32))   # top rows: warm-up scratch
        self.free = list(range(self.n_slots))
        self.live_slots = 0
        self.slot_fail = 0
        self.fused_diar = GraphCache(torch, self._fused_diar, "xdiar fused", self.device)

    # ------------------------------------------------------------------ slot tables
    def _alloc_tables(self, n):
        torch, sm, dev = self.torch, self.sm, self.device
        d = sm.fc_d_model
        self.tables = {
            "spkcache": torch.zeros((n, sm.spkcache_len, d), device=dev),
            "spkcache_preds": torch.zeros((n, sm.spkcache_len, sm.n_spk), device=dev),
            "spkcache_lengths": torch.zeros((n,), dtype=torch.long, device=dev),
            "spkcache_compressed": torch.zeros((n,), dtype=torch.bool, device=dev),
            "fifo": torch.zeros((n, sm.fifo_len, d), device=dev),
            "fifo_lengths": torch.zeros((n,), dtype=torch.long, device=dev),
            "mean_sil_emb": torch.zeros((n, d), device=dev),
            "n_sil_frames": torch.zeros((n,), dtype=torch.long, device=dev),
        }
        # Speaker gate history: the last two chunks' predictions (28 frames) per slot; the per-slot
        # speaker cap (handshake max_speakers) zeroes the columns above it inside the graph.
        self.gate_hist = torch.zeros((n, 2 * sm.chunk_len, sm.n_spk), device=dev)
        self.spk_cap = torch.full((n,), sm.n_spk, dtype=torch.long, device=dev)

    def _acquire_slot(self, sess):
        if not self.free:
            self.slot_fail += 1
            raise RuntimeError(f"no free diarizer slot ({self.n_slots} live)")
        slot = self.free.pop()
        self.live_slots += 1
        for t in self.tables.values():
            t[slot].zero_()                     # stream-ordered after the slot's last tick
        self.gate_hist[slot].zero_()
        self.spk_cap[slot] = min(sess.max_speakers, self.n_spk)
        sess.slot = slot
        return slot

    def release(self, sess):
        """Session ended: free its slot. Under the tick lock, so no replay is in flight for it."""
        slot = getattr(sess, "slot", None)
        if slot is None:
            return
        with self._lock:
            sess.slot = None
            self.free.append(slot)
            self.live_slots -= 1

    # ------------------------------------------------------------------ session lifecycle
    def admit(self, sess):
        """Per-row step-0 reset: NeMo's ``reset`` for a batch of one; with MT_XDIAR=1 the session's
        diarizer state is a zeroed slot of the tables instead of a per-session tensor set."""
        im = sess.streamer.instance_manager
        sess.slot = None
        with self._lock, self.torch.inference_mode():
            im.reset(batch_size=1)
            im.to(self.device)
            if self.flags.xdiar:
                self._acquire_slot(sess)
        sess.offset = 0.0
        sess.streamer._offset_chunk_start_time = 0.0

    def submit(self, sess, seg, geom, last, chunk=None):
        """Blocking submission (warm-up sessions): returns a concurrent Future."""
        it = _Item(sess, seg, geom, chunk, last)
        it.fut = Future()
        self.q.put(it)
        return it.fut

    def submit_async(self, sess, seg, geom, last, loop):
        """Submission from a connection coroutine: returns an asyncio future resolved by the tick."""
        it = _Item(sess, seg, geom, None, last)
        it.loop, it.afut = loop, loop.create_future()
        self.q.put(it)
        return it.afut

    # ------------------------------------------------------------------ thread loop
    def run(self):
        torch = self.torch
        while True:
            items = [self.q.get()]
            wait = self.flags.xsession_wait_ms / 1000.0
            deadline = time.perf_counter() + wait
            while len(items) < self.flags.xsession_b:
                try:
                    if wait > 0:
                        rem = deadline - time.perf_counter()
                        if rem <= 0:
                            break
                        items.append(self.q.get(timeout=rem))
                    else:
                        items.append(self.q.get_nowait())
                except queue.Empty:
                    break
            # A session waits on its future before it can submit again, so rows are distinct
            # sessions by construction; defer duplicates anyway rather than corrupt a state.
            seen, batch, defer = set(), [], []
            for it in items:
                (defer if id(it.sess) in seen else batch).append(it)
                seen.add(id(it.sess))
            for it in defer:
                self.q.put(it)
            t0 = time.perf_counter()
            try:
                with self._lock, torch.inference_mode():
                    self.tick(batch)
            except Exception as e:  # noqa: BLE001 - every row of the tick fails together
                logger.exception("xsession tick failed (B=%d)", len(batch))
                for it in batch:
                    it.err = e
            dt = time.perf_counter() - t0
            self.t_busy += dt
            self.tick_ms.append(dt * 1000)
            finish_all(batch)

    def summary(self):
        up = max(1e-9, time.time() - self.t_started)
        tm = sorted(self.tick_ms)
        pct = (lambda p: round(tm[min(len(tm) - 1, int(len(tm) * p))], 1)) if tm else (lambda p: None)
        tot = sum(self.phase_t.values()) or 1e-9
        return {"ticks": self.ticks, "rows": self.rows_served, "busy_frac": round(self.t_busy / up, 3),
                "tick_ms_p50": pct(.5), "tick_ms_p99": pct(.99),
                "b_hist": dict(sorted(self.b_hist.items())), "k_hist": dict(sorted(self.k_hist.items())),
                "phases_ms_per_tick": {k: round(v / max(1, self.ticks) * 1000, 3) for k, v in self.phase_t.items()},
                "phases_share": {k: round(v / tot, 3) for k, v in self.phase_t.items()},
                "xdiar": bool(self.flags.xdiar), "live_slots": self.live_slots, "slots": self.n_slots,
                "slot_fail": self.slot_fail, "xdiar_graphs": len(self.fused_diar.graphs),
                "xdiar_eager": self.fused_diar.eager_calls, "xmel": dict(self.xmel_stats)}

    # ------------------------------------------------------------------ the tick
    def tick(self, items):
        torch = self.torch
        B = len(items)
        prof = P.ENABLED
        sync = prof and any(getattr(it.sess, "prof_sync", True) for it in items)
        rec = {"b": B, "_sync": sync, "t": round(time.time(), 3)} if prof else None
        if prof:
            P.set_current(rec)
        t_start = time.perf_counter()
        marks = {"_t": t_start}

        def mark(name):
            if prof and sync:
                torch.cuda.synchronize()
            now = time.perf_counter()
            self.phase_t[name] += now - marks["_t"]
            if prof:
                rec[name] = rec.get(name, 0.0) + now - marks["_t"]
            marks["_t"] = now

        try:
            chunk = self._mel(items)                                             # [B, F, 121]
            mark("mel")
            if self.flags.xdiar and items[0].sess.streamer._diar_uses_feature_stacking:
                chunk_preds, active_mask = self._diar_slots(items, chunk)
            else:
                chunk_preds = self._diar_per_row(items, chunk)
                active_mask = self._gate_per_row(items, chunk_preds)
            mark("diar")
            # Per speaker, frames of inactivity at the END of this chunk (0..14): the turn builder's
            # diarizer end-of-turn signal (MT_TURN_DIAR_EOT_S). One D2H copy for the tick.
            inactive = (~(chunk_preds > 0.5)).flip(1).cumprod(1).sum(1).tolist()       # [B, n_spk]
            for i, it in enumerate(items):
                st = it.sess.streamer
                if st._cache_gating:
                    it.active = [k for k, m in enumerate(active_mask[i]) if m]
                else:
                    it.active = list(range(st.n_active_speakers_per_stream))
                it.sess.last_active = it.active     # read by the handler's frame ("active", MT_SPEECH_EVENTS)
                it.sess.tail_inactive = inactive[i]
            mark("gate")

            self._asr(items, chunk, rec, mark)

            for it in items:
                s = it.sess
                if it.active:
                    s.streamer.instance_manager.batch_asr_states[0].update_sessionwise_seglsts_for_parallel(
                        offset=s.offset)
                s.offset += s.streamer._frame_hop_length * s.streamer._frame_len_sec
                s.streamer._offset_chunk_start_time = s.offset
            mark("seglst")
        finally:
            if prof:
                P.set_current(None)
        self.ticks += 1
        self.rows_served += B
        self.b_hist[B] += 1
        if prof:
            wall = time.perf_counter() - t_start
            live = self.live_fn()
            for it in items:
                r = {k: v for k, v in rec.items() if not k.startswith("_")}
                r.update(sid=it.sess.sid, step=it.sess.step_num, last=it.last, live=live,
                         wait=t_start - it.t_submit, step_wall=wall, k_row=len(it.active),
                         mel=getattr(it.sess, "_mel_s", 0.0),
                         k_known=len(it.sess.streamer.instance_manager.batch_asr_states[0].get_speakers()))
                it.sess._mel_s = 0.0
                out = {k: (round(v * 1000, 3) if isinstance(v, float) and k != "t" else v) for k, v in r.items()}
                it.sess.recs.append(out)
                logger.info("MTPROF %s", json.dumps(out))
        if self.ticks % 1000 == 0:
            logger.info("xsession stats: %s", json.dumps(self.summary()))

    # ------------------------------------------------------------------ mel
    def _mel(self, items):
        """Features for every row: one preprocessor + normalisation call per mel geometry group
        (steady rows all share one), legacy rows pass their ready ``chunk`` through."""
        torch = self.torch
        out = [None] * len(items)
        groups = collections.OrderedDict()
        for i, it in enumerate(items):
            if it.seg is None:
                out[i] = it.chunk
            else:
                groups.setdefault(it.geom, []).append(i)
        for geom, idxs in groups.items():
            first, disc, nf, _n = geom
            sig = torch.from_numpy(np.stack([items[i].seg for i in idxs])).to(self.device)
            chunk = self._mel_rows(sig, first, disc, nf)
            self.xmel_stats["groups"] += 1
            if self.flags.xmel_verify:
                for j, i in enumerate(idxs):
                    ref = self._mel_rows(sig[j:j + 1], first, disc, nf)
                    self.xmel_stats["rows"] += 1
                    self.xmel_stats["equal"] += int(bool(torch.equal(ref, chunk[j:j + 1])))
            for j, i in enumerate(idxs):
                out[i] = chunk[j:j + 1]
        return torch.cat(out) if len(out) > 1 else out[0]

    def _mel_rows(self, sig, first, disc, nf):
        """``MelChunker.next_chunk`` for B windows of one geometry, bit-identical per row.

        The STFT, magnitude and power are per-element or fixed-size ops, so they run batched
        (``linear_spec=True``). The filterbank GEMM is a cuBLAS bmm whose kernel -- hence rounding --
        changes with the batch, and the per-chunk normalisation reduces over frames with a
        batch-dependent kernel config; both run per row at the exact B=1 shapes the per-row mel uses
        (measured: the fully batched call matched the per-row features only on B=1 ticks). Then
        ``disc`` recomputed context frames are dropped, a first chunk gets zero pre-encode cache frames,
        and NeMo's per-chunk normalisation is applied."""
        torch = self.torch
        B = sig.shape[0]
        fbf = self.pre.featurizer
        ln = torch.full((B,), sig.shape[1], dtype=torch.int64, device=self.device)
        with torch.amp.autocast("cuda", enabled=False):
            spec, _ = fbf.forward(sig, ln, linear_spec=True)                    # [B, n_fft/2+1, T]
            fb = fbf.fb.to(spec.dtype)
            one = torch.full((1,), nf + (self.cache_frames if first else 0), dtype=torch.int64, device=self.device)
            rows = []
            for i in range(B):
                x = torch.matmul(fb, spec[i:i + 1])                             # the B=1 GEMM shape
                if fbf.log:
                    if fbf.log_zero_guard_type == "add":
                        x = torch.log(x + fbf.log_zero_guard_value_fn(x))
                    else:
                        x = torch.log(torch.clamp(x, min=fbf.log_zero_guard_value_fn(x)))
                x = x[:, :, disc: disc + nf]
                if first:
                    x = torch.cat([x.new_zeros((1, x.shape[1], self.cache_frames)), x], dim=-1)
                x, _, _ = self._normalize_batch(x=x.contiguous(), seq_len=one, normalize_type=self.norm_type)
                rows.append(x)
        return torch.cat(rows) if B > 1 else rows[0]

    # ------------------------------------------------------------------ diarizer: per-row sync path
    def _diar_per_row(self, items, chunk):
        """Fallback (``MT_XDIAR=0``): the committed build's per-session sync-mode graph path, one
        call per row (bit-identical to NeMo's sync path, ~3.8 ms per row)."""
        torch = self.torch
        out = []
        pad = max(1, int(getattr(self.flags, "diar_pad_rows", 1)))
        fields = ("spkcache", "spkcache_preds", "fifo", "fifo_preds", "mean_sil_emb", "n_sil_frames")
        for i, it in enumerate(items):
            st = it.sess.streamer
            ds = st.instance_manager.diar_states
            dc, dl, ddrop = st._prepare_diar_chunk(chunk[i:i + 1], self.lengths, self.drop)
            state = ds.streaming_state
            if pad > 1:
                # Duplicate the row: identical rows, the core runs at `pad` rows, row 0 is kept.
                dc, dl = dc.expand(pad, -1, -1), dl.expand(pad)
                state = self._State()
                for k in fields + ("spkcache_compressed", "spk_perm"):
                    v = getattr(ds.streaming_state, k, None)
                    setattr(state, k, v.expand(pad, *v.shape[1:]).contiguous() if isinstance(v, torch.Tensor) and v.dim() >= 1 else v)
            new_state, preds = self.diar.forward_streaming_step(
                processed_signal=dc.transpose(1, 2), processed_signal_length=dl, streaming_state=state,
                total_preds=torch.zeros((pad, 0, self.n_spk), device=self.device), drop_extra_pre_encoded=ddrop,
                right_offset=getattr(st, "_diar_right_offset", 0))
            if pad > 1:
                for k in fields + ("spkcache_compressed", "spk_perm"):
                    v = getattr(new_state, k, None)
                    if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == pad:
                        setattr(new_state, k, v[:1])
                preds = preds[:1]
            ds.streaming_state = new_state
            out.append(preds)
        return torch.cat(out)

    def _gate_per_row(self, items, chunk_preds):
        """Per-row diarizer bookkeeping and the speaker gate (one D2H copy for the tick)."""
        torch = self.torch
        maxes = []
        for i, it in enumerate(items):
            s = it.sess
            ds = s.streamer.instance_manager.diar_states
            cp = chunk_preds[i:i + 1]
            if s.max_speakers < cp.shape[2]:
                cp[:, :, s.max_speakers:] = 0.0
            ds.diar_pred_out_stream = torch.cat([ds.diar_pred_out_stream, cp], dim=1)
            ds.previous_chunk_preds = cp
            gate = s.streamer._nframes_per_chunk * s.streamer._cache_gating_buffer_size
            maxes.append(ds.diar_pred_out_stream[0, -gate:].amax(dim=0))
        return (torch.stack(maxes) > 0.5).tolist()

    # ------------------------------------------------------------------ diarizer: slot tables
    def _diar_slots(self, items, chunk):
        """One graph replay for every row of the tick (see ``_fused_diar``); returns the chunk
        predictions ``[B, 14, n_spk]`` and the speaker gate as a Python list."""
        torch = self.torch
        B = len(items)
        Bp = max(B, self.flags.xdiar_rows) if self.flags.xdiar_rows > 0 else B
        idx = [it.sess.slot for it in items] + [self.n_slots + i for i in range(Bp - B)]   # pad rows: scratch
        idx_t = torch.tensor(idx, dtype=torch.long)
        dchunk = chunk[:, :, self.cache_frames:]                                 # [B, F, 112]
        if Bp > B:
            dchunk = torch.cat([dchunk, dchunk.new_zeros((Bp - B, *dchunk.shape[1:]))])
        cp, gmax = self.fused_diar.run(idx_t, dchunk)
        cp = cp[:B].clone()           # static output buffer: detach the rows from the next replay
        active = (gmax[:B] > 0.5).tolist()
        for i, it in enumerate(items):
            it.sess.streamer.instance_manager.diar_states.previous_chunk_preds = cp[i:i + 1]
        return cp, active

    def _fused_diar(self, idx, dchunk):
        """Whole diarizer step for the rows whose state sits at table rows ``idx``: pre-encode -> gather
        -> concat+pad -> encoder -> head -> NeMo's async update -> gate -> write-back. Pure tensor ops,
        no host sync (capturable)."""
        torch, sm, diar = self.torch, self.sm, self.diar
        lens = torch.full((dchunk.shape[0],), dchunk.shape[-1], dtype=torch.int64, device=self.device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            chunk_embs, chunk_lens = diar._call_pre_encode(dchunk.transpose(1, 2), lens)
            st = types.SimpleNamespace(**{k: self.tables[k].index_select(0, idx) for k in STATE_FIELDS})
            preds, enc_lens = self._core_eager(chunk_embs, chunk_lens, st.spkcache, st.spkcache_lengths,
                                               st.fifo, st.fifo_lengths)
            high_res = None
            if diar.high_resolution:
                high_res = preds
                preds = sm.downsample_preds(high_res, diar.upsample_factor)
            n_frames = preds.shape[1]
            mask = torch.arange(n_frames, device=preds.device).view(1, -1, 1) < enc_lens.view(-1, 1, 1)
            preds = preds.masked_fill(~mask, 0.0)                                  # == apply_mask_to_preds
            saved_sc, saved_f = st.spkcache_lengths.clone(), st.fifo_lengths.clone()
            max_chunk_len = chunk_embs.shape[1]
            cl = chunk_lens.clamp(min=0, max=max_chunk_len)
            chunk_preds = self._update_async(st, chunk_embs, cl, preds, max_chunk_len)
            if diar.high_resolution:
                chunk_preds = diar._extract_async_high_resolution_chunk_preds(
                    high_resolution_preds=high_res, spkcache_lengths=saved_sc, fifo_lengths=saved_f,
                    chunk_lengths=cl, max_chunk_len=max_chunk_len, lc_enc=0)
            native = 1 if diar.high_resolution else diar.encoder.subsampling_factor
            ds_f = diar.output_subsampling_factor // native
            if ds_f > 1:
                chunk_preds = sm.downsample_preds(chunk_preds, ds_f)
        for k in STATE_FIELDS:
            self.tables[k].index_copy_(0, idx, getattr(st, k))
        cp = chunk_preds.float()
        cap = self.spk_cap.index_select(0, idx)                                       # per-session max_speakers
        cp = cp * (torch.arange(cp.shape[2], device=cp.device).view(1, 1, -1) < cap.view(-1, 1, 1)).to(cp.dtype)
        hist = self.gate_hist.index_select(0, idx)
        n_new = cp.shape[1]
        new_hist = torch.cat([hist[:, n_new:], cp], dim=1)
        self.gate_hist.index_copy_(0, idx, new_hist)
        return cp, new_hist.amax(dim=1)

    def _update_async(self, st, chunk, cl, preds, max_chunk_len):
        """``SortformerModules.streaming_update_async`` on a namespace of gathered rows (lc = rc = 0),
        with the speaker-cache compression made sync-free (``_update_spkcache_all``)."""
        sm = self.sm
        max_spkcache_len, max_fifo_len = st.spkcache.shape[1], st.fifo.shape[1]
        max_pop_out_len = max(sm.spkcache_update_period, max_fifo_len, max_chunk_len)
        max_pop_out_len = min(max_pop_out_len, max_chunk_len + max_fifo_len)
        cur_sc_preds, cur_fifo_preds, chunk_preds = sm._gather_async_predictions(
            st, preds, st.spkcache_lengths, st.fifo_lengths, cl, max_spkcache_len, max_fifo_len, max_chunk_len, 0)
        pop_len, new_fifo_len = sm._compute_async_fifo_pop_lengths(st.spkcache_lengths, st.fifo_lengths, cl, max_fifo_len)
        pop_embs, pop_preds, valid_pop = sm._update_async_fifo(
            st, chunk, cur_fifo_preds, chunk_preds, st.fifo_lengths, pop_len, new_fifo_len, max_chunk_len,
            max_pop_out_len, 0)
        sm._update_async_silence_profile(st, pop_embs, pop_preds, valid_pop)
        self._update_spkcache_all(st, cur_sc_preds, pop_embs, pop_preds, pop_len)
        return chunk_preds

    def _update_spkcache_all(self, st, cur_sc_preds, pop_embs, pop_preds, pop_len):
        """``_update_async_spkcache`` with compression run for every row and selected by ``where``
        (NeMo's version picks the rows with a host sync; the values are identical)."""
        torch, sm = self.torch, self.sm
        batch_size, max_pop_out_len, emb_dim = pop_embs.shape
        n_spk = pop_preds.shape[2]
        max_spkcache_len = st.spkcache.shape[1]
        sc_len = st.spkcache_lengths
        upd_len = sc_len + pop_len
        need = upd_len > sm.spkcache_len
        first = (~st.spkcache_compressed) & need
        cand_old_preds = torch.where(first.view(-1, 1, 1), cur_sc_preds, st.spkcache_preds)
        cand_embs = torch.cat([st.spkcache, pop_embs.to(st.spkcache.dtype), st.spkcache.new_zeros((batch_size, 1, emb_dim))], dim=1)
        cand_preds = torch.cat([cand_old_preds, pop_preds, st.spkcache_preds.new_zeros((batch_size, 1, n_spk))], dim=1)
        pos = torch.arange(max_spkcache_len + max_pop_out_len, device=pop_preds.device).unsqueeze(0)
        logical = pos.expand(batch_size, -1)
        valid = pos < upd_len.unsqueeze(1)
        phys = torch.where(logical < sc_len.unsqueeze(1), logical, max_spkcache_len + logical - sc_len.unsqueeze(1))
        phys = torch.where(valid, phys, max_spkcache_len + max_pop_out_len)
        upd_sc = torch.gather(cand_embs, 1, phys.unsqueeze(-1).expand(-1, -1, emb_dim))
        upd_preds = torch.gather(cand_preds, 1, phys.unsqueeze(-1).expand(-1, -1, n_spk))
        comp_sc, comp_preds = self._compress_all(upd_sc, upd_preds, st.mean_sil_emb)
        nc = need.view(-1, 1, 1)
        st.spkcache = torch.where(nc, comp_sc, upd_sc[:, : sm.spkcache_len])
        st.spkcache_preds = torch.where(nc, comp_preds, upd_preds[:, : sm.spkcache_len])
        st.spkcache_compressed = st.spkcache_compressed | need
        st.spkcache_lengths = upd_len.clamp(max=sm.spkcache_len)

    def _compress_all(self, emb_seq, preds, mean_sil_emb):
        """``_compress_spkcache(permute_spk=False)`` with device-side indices only (same values)."""
        torch, sm = self.torch, self.sm
        batch_size, n_frames, n_spk = preds.shape
        dev = preds.device
        if sm.use_learnable_sil_emb:
            mean_sil_emb = sm.learnable_sil_emb.to(dtype=emb_seq.dtype, device=dev).unsqueeze(0).expand(batch_size, -1)
        per_spk = sm.spkcache_len // n_spk - sm.spkcache_sil_frames_per_spk
        strong = math.floor(per_spk * sm.strong_boost_rate)
        weak = math.floor(per_spk * sm.weak_boost_rate)
        min_pos = math.floor(per_spk * sm.min_pos_scores_rate)
        scores = sm._get_log_pred_scores(preds)
        is_speech = preds > 0.5
        scores = torch.where(is_speech, scores, float("-inf"))
        is_pos = scores > 0
        replace = (~is_pos) * is_speech * (is_pos.sum(dim=1).unsqueeze(1) >= min_pos)
        scores = torch.where(replace, float("-inf"), scores)
        if sm.scores_boost_latest > 0:
            scores[:, sm.spkcache_len:, :] += sm.scores_boost_latest
        for n_boost, scale in ((strong, 2), (weak, 1)):
            _, top = torch.topk(scores, n_boost, dim=1, largest=True, sorted=False)
            scores = scores.scatter_add(1, top, torch.full_like(top, -scale * math.log(0.5), dtype=scores.dtype))
        if sm.spkcache_sil_frames_per_spk > 0:
            pad = torch.full((batch_size, sm.spkcache_sil_frames_per_spk, n_spk), float("inf"), device=dev)
            scores = torch.cat([scores, pad], dim=1)
        n_tot = scores.shape[1]
        n_no_sil = n_tot - sm.spkcache_sil_frames_per_spk
        flat = scores.permute(0, 2, 1).reshape(batch_size, -1)
        top_v, top_i = torch.topk(flat, sm.spkcache_len, dim=1, sorted=False)
        top_i = torch.where(top_v != float("-inf"), top_i, sm.max_index)
        top_s, _ = torch.sort(top_i, dim=1)
        disabled = top_s == sm.max_index
        top_s = torch.remainder(top_s, n_tot)
        disabled = disabled | (top_s >= n_no_sil)
        top_s = torch.where(disabled, 0, top_s)
        emb_g = torch.gather(emb_seq, 1, top_s.unsqueeze(-1).expand(-1, -1, emb_seq.shape[2]))
        emb_g = torch.where(disabled.unsqueeze(-1), mean_sil_emb.to(emb_g.dtype).unsqueeze(1).expand(-1, sm.spkcache_len, -1), emb_g)
        preds_g = torch.gather(preds, 1, top_s.unsqueeze(-1).expand(-1, -1, n_spk))
        preds_g = torch.where(disabled.unsqueeze(-1), 0.0, preds_g)
        return emb_g, preds_g

    def _core_eager(self, chunk_embs, chunk_lens, spkcache, spkcache_lengths, fifo, fifo_lengths):
        """concat+pad -> encoder -> head at the fixed ``l_max`` width (capture-safe: no host scalars)."""
        embs, lens = self._concat_and_pad([spkcache, fifo, chunk_embs], [spkcache_lengths, fifo_lengths, chunk_lens],
                                          self.l_max)
        emb_seq, emb_len = self.diar.frontend_encoder(processed_signal=embs, processed_signal_length=lens,
                                                      bypass_pre_encode=True)
        preds = self.diar.forward_infer(emb_seq=emb_seq, emb_seq_length=emb_len)
        return preds, emb_len

    def _concat_and_pad(self, embs, lengths, output_length):
        """``SortformerModules.concat_and_pad`` with the host-scalar ``torch.tensor`` replaced by a
        device fill so the function can be captured into a CUDA graph."""
        torch = self.torch
        device, dtype = embs[0].device, embs[0].dtype
        batch_size, emb_dim = embs[0].shape[0], embs[0].shape[2]
        total_lengths = torch.sum(torch.stack(lengths), dim=0)
        flat = torch.zeros(batch_size * output_length + 1, emb_dim, device=device, dtype=dtype)
        start = torch.zeros(batch_size, dtype=torch.int64, device=device)
        offsets = torch.arange(batch_size, dtype=torch.long, device=device).unsqueeze(1) * output_length
        pad_idx = torch.full((), batch_size * output_length, dtype=torch.long, device=device)
        for emb, length in zip(embs, lengths):
            src = torch.arange(emb.shape[1], dtype=torch.long, device=device).unsqueeze(0)
            valid = src < length.unsqueeze(1)
            dst = torch.where(valid, offsets + start.unsqueeze(1) + src, pad_idx).reshape(-1)
            flat.index_copy_(0, dst, emb.reshape(-1, emb_dim).type_as(flat))
            start = start + length
        return flat[: batch_size * output_length].view(batch_size, output_length, emb_dim), total_lengths

    def warm_core(self, b_max):
        """Capture the fused diarizer graph for every batch size the stepper can issue (scratch slots)."""
        torch = self.torch
        if not self.flags.xdiar:
            return
        t0 = time.perf_counter()
        sizes = [self.flags.xdiar_rows] if self.flags.xdiar_rows > 0 else list(range(1, b_max + 1))
        F = self.asr.encoder._feat_in if hasattr(self.asr.encoder, "_feat_in") else 128
        with torch.inference_mode():
            for b in sizes:
                idx = torch.arange(self.n_slots, self.n_slots + b, dtype=torch.long)
                dchunk = torch.zeros((b, F, self.chunk_frames), device=self.device)
                self.fused_diar.run(idx, dchunk)
            for t in self.tables.values():
                t[self.n_slots:].zero_()
            self.gate_hist[self.n_slots:].zero_()
        torch.cuda.synchronize()
        logger.info("xdiar fused graphs captured for B=%s in %.1fs (%d graphs, tables %d slots)", sizes,
                    time.perf_counter() - t0, len(self.fused_diar.graphs), self.n_slots)

    # ------------------------------------------------------------------ ASR
    def _asr(self, items, chunk, rec, mark):
        """Gather the (session, speaker) rows of the tick, one padded encoder+decoder call per
        ``rows`` slab, scatter caches/hypotheses back."""
        torch = self.torch
        fast = self.fast
        bypass = not fast.no_bypass
        feats, flens = chunk, None
        if bypass:   # eager encoder: pre-encode once for the whole tick, then replicate rows
            lens = self.lengths.expand(len(items))
            feats, flens = items[0].sess.streamer.forward_pre_encoded(chunk, lens, self.drop)
        rows, sig, tgt, bg, lc, lt, ln, hyps = [], [], [], [], [], [], [], []
        for i, it in enumerate(items):
            if not it.active:
                continue
            im = it.sess.streamer.instance_manager
            st = im.batch_asr_states[0]
            pcp = im.diar_states.previous_chunk_preds[0]                       # [14, n_spk]
            binp = pcp > 0.5
            for spk in it.active:
                if spk not in st.get_speakers():
                    im.add_speaker(0, spk)
                others = [o for o in it.active if o != spk]
                rows.append((i, spk))
                sig.append(feats[i])
                tgt.append(pcp[:, spk])
                bg.append(binp[:, others].any(dim=-1) if others else torch.zeros_like(binp[:, 0]))
                lc.append(st.cache_last_channel[:, spk])
                lt.append(st.cache_last_time[:, spk])
                ln.append(st.cache_last_channel_len[spk])
                hyps.append(st.previous_hypothesis[spk])
        S = len(rows)
        if rec is not None:
            rec["k_active"] = S
        self.k_hist[S] += 1
        if S == 0:
            mark("gather")
            return
        sig_t = torch.stack(sig)
        len_t = (torch.stack([flens[i] for i, _ in rows]) if bypass
                 else torch.full((S,), chunk.shape[-1], dtype=torch.int64, device=self.device))
        tgt_t = (torch.stack(tgt) > 0.5).float()
        bg_t = torch.stack(bg).float()
        lc_t = torch.stack(lc).transpose(0, 1)
        lt_t = torch.stack(lt).transpose(0, 1)
        ln_t = torch.stack(ln)
        mark("gather")
        max_rows = self.flags.xsession_rows
        cap = max_rows if max_rows > 0 else S
        saved_pad = fast.flags.pad_rows
        try:
            for a in range(0, S, cap):
                b = min(S, a + cap)
                # Smallest warmed bucket that holds this slab: the pad wrapper and the speaker-target
                # buffers read flags.pad_rows at call time.
                fits = [x for x in self.flags.xsession_row_buckets if x >= b - a]
                fast.flags.pad_rows = fits[0] if fits else max_rows
                if rec is not None:
                    rec["rows"] = fast.flags.pad_rows
                self.asr.set_speaker_targets(tgt_t[a:b], bg_t[a:b])
                pred_out, _, n_lc, n_lt, n_ln, n_hyps = self.asr.conformer_stream_step(
                    processed_signal=sig_t[a:b], processed_signal_length=len_t[a:b],
                    cache_last_channel=lc_t[:, a:b], cache_last_time=lt_t[:, a:b], cache_last_channel_len=ln_t[a:b],
                    keep_all_outputs=False, previous_hypotheses=hyps[a:b], previous_pred_out=[None] * (b - a),
                    drop_extra_pre_encoded=self.drop, return_transcription=True, bypass_pre_encode=bypass)
                mark("asr")
                for r in range(a, b):
                    i, spk = rows[r]
                    items[i].sess.streamer.instance_manager.batch_asr_states[0].update_asr_state(
                        spk, n_lc[:, r - a], n_lt[:, r - a], n_ln[r - a], n_hyps[r - a], pred_out[r - a])
                mark("upd_state")
        finally:
            fast.flags.pad_rows = saved_pad
