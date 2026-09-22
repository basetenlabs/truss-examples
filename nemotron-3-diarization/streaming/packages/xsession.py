"""Cross-session batched stepper for the streaming Nemotron diarizer (``NEMO_DIAR_XSESSION=1``).

The committed build ran one ``forward_streaming_step`` per connection from a thread pool: ~60 real-time
``low`` streams per RTX-PRO-6000, the GPU serialising 60 batch-1 forwards. This module replaces that
with ONE stepper thread per latency profile. A connection's handler slices the raw window its next
chunk needs (numpy only, no GPU work on the event loop) and hands it to the stepper; the stepper takes
every queued chunk and runs one pass for all of them.

Two tick implementations share the thread, the queue and the grouping (``NEMO_DIAR_XSESSION_TICK``):

``fused`` (default) -- every session's streaming state lives in one preallocated slot table per
profile (``[slots, ...]`` tensors, a slot per live session). A tick is ONE CUDA graph replay per batch
size covering the whole step: mel of the raw windows -> gather the rows' state by slot index ->
pre-encode -> concat+pad -> FastConformer -> Sortformer head -> NeMo's async cache/FIFO update ->
write the rows' state back by slot index. NeMo's ``_update_async_spkcache`` decides speaker-cache
compression with a host sync (``torch.where(need_compress)`` + ``if len(idx)``), which forced the
stepper thread to wait for the GPU in the middle of every tick; here compression is computed for every
row of the batch and selected with ``torch.where`` (identical values for the rows that need it -- topk,
sort and gather are per row), so the tick has no host sync until its device-to-host copy. Inputs travel
through pinned host buffers (H2D and D2H both asynchronous) and ticks are pipelined one deep: the
stepper packs and issues tick k+1 while the GPU runs tick k, then waits on tick k's event with the GIL
released. Non-steady geometries (a session's first chunk, a flush tail) run the same function eagerly.

``legacy`` -- the previous build's tick: per-session state tensors gathered with ``torch.cat`` into a
``StreamingSortformerState``, the core replayed from a CUDA graph per batch size, NeMo's
``streaming_update_async`` on the batch, state scattered back as row views. Kept for A/B and as the
reference for the per-tick budget.

Both are fp32 with the eager kernels; rows of a group share one mel call (frame-local features,
``normalize: NA``) and one core call in NeMo's *async* streaming layout (fixed ``[spkcache | fifo |
chunk]`` buffers with per-row valid lengths), so rows at any position in their calls share the call.

``SyncGraphs`` is the other lever measured for the same job: sync-mode state (growing shapes, the path
NeMo's batch inference takes) with the eager core captured in a CUDA graph per distinct sequence length.
Bit-identical to eager sync mode, but one call per connection (no batching), kept behind
``NEMO_DIAR_SYNC_GRAPHS=1`` for A/B.
"""

import collections
import json
import logging
import math
import os
import queue
import subprocess
import threading
import time
import types

import numpy as np

logger = logging.getLogger(__name__)

SR = 16000
HOP = 160          # 10 ms mel hop (samples)
WIN = 400          # 25 ms mel window (samples)
SUB = 8            # feature stacking: 1 encoder frame = 8 mel frames
LPAD = 3           # left-context mel frames recomputed for exact windowed features
FRAME_S = 0.01     # output frame (high-resolution head: 10 ms)
MIN_TURN_S = 0.08
MERGE_GAP_S = 0.10


def env_bool(name, default):
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "off", "no")


class GraphCache:
    """CUDA-graph replay of a pure function of tensors (+ Python constants), keyed by the input shapes
    and the constants.

    One eager dry run on a fresh key (kernel selection), capture into a graph with static input buffers
    (all graphs of a cache share one memory pool: they are replayed one at a time on one stream, and
    their outputs are copied into persistent buffers inside the graph), then ``copy_`` + replay.
    Inputs may be pinned host tensors -- the copy into the static buffers is then an asynchronous H2D.
    Outputs are the graph's static buffers, valid until the next replay of the same key.
    """

    def __init__(self, torch, fn, name, device, max_graphs=256):
        self.torch, self.fn, self.name, self.max_graphs = torch, fn, name, max_graphs
        self.device = device
        self.graphs = {}
        self.disabled = False
        self.replays = 0
        self.eager_calls = 0
        self.pool = torch.cuda.graph_pool_handle()

    def eager(self, inputs, consts=()):
        self.eager_calls += 1
        dev = [t if t.device == self.device else t.to(self.device, non_blocking=True) for t in inputs]
        return self.fn(*dev, *consts)

    def run(self, inputs, consts=(), capture=True):
        torch = self.torch
        key = (tuple(consts), tuple((tuple(t.shape), str(t.dtype)) for t in inputs))
        g = self.graphs.get(key)
        if g is None:
            if not capture or self.disabled or len(self.graphs) >= self.max_graphs or torch.cuda.is_current_stream_capturing():
                return self.eager(inputs, consts)
            try:
                g = self._capture(key, inputs, consts)
            except Exception:  # noqa: BLE001 - eager is correct, just slower
                logger.exception("%s: graph capture failed for %s; running eager from now on", self.name, key)
                self.disabled = True
                return self.eager(inputs, consts)
        for buf, t in zip(g["in"], inputs):
            buf.copy_(t, non_blocking=True)
        g["graph"].replay()
        self.replays += 1
        return g["out"]

    def _capture(self, key, inputs, consts):
        torch = self.torch
        device = self.device
        static = [t.to(device).clone() for t in inputs]
        outs = self.fn(*static, *consts)                # dry run: shapes + kernel warm-up
        out_bufs = [torch.empty_like(o) for o in outs]
        del outs
        torch.cuda.synchronize(device)
        s = torch.cuda.Stream(device)
        s.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s), torch.cuda.graph(graph, pool=self.pool, stream=s, capture_error_mode="thread_local"):
            outs = self.fn(*static, *consts)
            for b, o in zip(out_bufs, outs):
                b.copy_(o)
        torch.cuda.current_stream(device).wait_stream(s)
        g = {"graph": graph, "in": static, "out": tuple(out_bufs)}
        self.graphs[key] = g
        logger.info("%s: graph captured for %s (%d graphs)", self.name, key[1][0][0], len(self.graphs))
        return g


def concat_and_pad_dev(torch, embs, lengths, output_length):
    """``SortformerModules.concat_and_pad`` with its host-scalar ``torch.tensor`` replaced by a device
    fill, so the function can be captured into a CUDA graph. Everything else identical."""
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


class Chunk:
    """One chunk of one session, as the handler hands it to the stepper (numpy only)."""

    __slots__ = ("sess", "seg", "a", "b", "left_off", "right_off", "f_true", "disc", "flush",
                 "loop", "fut", "done", "result", "error", "t_submit", "t_start", "t_end", "batch")

    def __init__(self, sess, seg, a, b, left_off, right_off, f_true, disc, flush):
        self.sess, self.seg, self.a, self.b = sess, seg, a, b
        self.left_off, self.right_off, self.f_true, self.disc, self.flush = left_off, right_off, f_true, disc, flush
        self.loop = self.fut = self.done = None
        self.result = self.error = None
        self.t_submit = time.perf_counter()
        self.t_start = self.t_end = 0.0
        self.batch = 0

    def key(self):
        # Rows of a group share one mel call and one (lc, rc) update, so the key is the full chunk
        # geometry including the mel left context: a session's first chunk has none (disc 0), every
        # later chunk recomputes LPAD frames, and the checkpoint's chunk_left_context is 0, so the two
        # differ only in window length.
        return (self.left_off, self.right_off, self.f_true, self.disc, len(self.seg))

    def _resolve(self):
        if self.fut.done():
            return
        if self.error is not None:
            self.fut.set_exception(self.error)
        else:
            self.fut.set_result(self.result)


class Release:
    """Queue marker: the session's slot goes back to the free list once every chunk queued before it
    has been issued (stream order then guarantees the slot's last tick precedes its re-use)."""

    __slots__ = ("sess",)

    def __init__(self, sess):
        self.sess = sess


def finish_all(chunks):
    """Resolve a tick's chunks: one loop wake-up for all asyncio futures, one event per blocking waiter."""
    by_loop = {}
    for ch in chunks:
        if ch.loop is not None:
            by_loop.setdefault(ch.loop, []).append(ch)
        elif ch.done is not None:
            ch.done.set()
    for loop, chs in by_loop.items():
        try:
            loop.call_soon_threadsafe(_resolve_many, chs)
        except RuntimeError:   # loop closed (shutdown)
            pass


def _resolve_many(chs):
    for ch in chs:
        ch._resolve()


STATE_FIELDS = ("spkcache", "spkcache_preds", "spkcache_lengths", "spkcache_compressed", "fifo",
                "fifo_lengths", "mean_sil_emb", "n_sil_frames")


class Phases:
    """Accumulated wall time per named phase of the stepper loop (perf_counter marks; timers only)."""

    def __init__(self):
        self.t = collections.Counter()
        self.n = collections.Counter()
        self._last = time.perf_counter()

    def mark(self, name):
        now = time.perf_counter()
        self.t[name] += now - self._last
        self.n[name] += 1
        self._last = now

    def reset(self):
        self._last = time.perf_counter()

    def table(self, ticks):
        tot = sum(self.t.values()) or 1e-9
        return {k: {"ms_per_tick": round(v / max(1, ticks) * 1000, 3), "share": round(v / tot, 3)}
                for k, v in sorted(self.t.items(), key=lambda kv: -kv[1])}


class Stepper(threading.Thread):
    """The batched stepper of one latency profile: queue, thread, per-tick gather/scatter."""

    RING = 3   # pinned buffer sets / events in flight

    def __init__(self, torch, model, profile, core="graphs", b_max=16, wait_ms=0.0, tick_mode="fused",
                 slots=640, prof_sync=False, b_target=None, core_dtype=None, b_min=1):
        super().__init__(daemon=True, name=f"stepper-{profile}")
        from nemo.collections.asr.modules.sortformer_modules import StreamingSortformerState

        self.torch, self.m, self.profile = torch, model, profile
        self._State = StreamingSortformerState
        self.sm = model.sortformer_modules
        self.device = model.device
        self.b_max, self.wait_ms = b_max, wait_ms
        self.b_target = min(b_max, b_target or b_max)   # stop coalescing once this many rows are queued
        # Never replay a steady tick below b_min rows: pad with the reserved scratch slots (zero audio,
        # outputs discarded). The bf16 core's small-row-count graphs carry a one-sided rounding bias.
        self.b_min = max(1, min(b_max, b_min))
        self.pad_rows = 0
        # Core (encoder + head) dtype: the weights were cast once at load; inputs are cast at the core
        # boundary and the outputs come back fp32 for NeMo's update (state tables stay fp32).
        self.core_dtype = core_dtype or torch.float32
        self.tick_mode = tick_mode
        self.prof_sync = prof_sync
        sm = self.sm
        self.chunk, self.rc = sm.chunk_len, sm.chunk_right_context
        self.clc = getattr(sm, "chunk_left_context", 1)
        self.f_steady = (self.clc + self.chunk + self.rc) * SUB
        self.l_max = sm.spkcache_len + sm.fifo_len + self.clc + self.chunk + self.rc
        self.n_spk = sm.n_spk
        self.core_mode = core
        self.q = queue.SimpleQueue()
        self.ticks = self.rows = 0
        self.t_busy = 0.0
        self.t_started = time.time()
        self.b_hist = collections.Counter()
        self.tick_ms = collections.deque(maxlen=4096)
        self.gpu_ms = collections.deque(maxlen=4096)
        self.gpu_rows = collections.deque(maxlen=4096)
        self.gpu_t = collections.deque(maxlen=4096)
        self.wait_ms_hist = collections.deque(maxlen=4096)
        self.phases = Phases()
        self.eager_ticks = 0
        self.slot_fail = 0
        self._inflight = None
        # legacy: core graph per batch size; fused: whole-tick graph per (geometry, batch size)
        if tick_mode == "fused":
            self.n_slots = slots
            self._alloc_tables(slots + b_max)                       # top b_max slots are warm-up scratch
            self.free = list(range(slots))
            self.free_lock = threading.Lock()
            self.live_slots = 0
            self.fused = GraphCache(torch, self._fused, f"{profile} fused", self.device) if core == "graphs" else None
            self.core = None
            self._ring = None
            self._inflight = None
        else:
            self.core = GraphCache(torch, self._core_eager, f"{profile} core", self.device) if core == "graphs" else None
            self.fused = None

    # ------------------------------------------------------------------ slot tables (fused)
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

    def _acquire(self, sess):
        with self.free_lock:
            if not self.free:
                self.slot_fail += 1
                raise RuntimeError(f"{self.profile}: no free session slot ({self.n_slots})")
            slot = self.free.pop()
            self.live_slots += 1
        for t in self.tables.values():
            t[slot].zero_()                                      # stream-ordered after the slot's last tick
        sess.slot = slot
        return slot

    def release(self, sess):
        """Called from any thread when a session ends; processed in queue order by the stepper."""
        if getattr(sess, "slot", None) is not None:
            sess.closed = True
            self.q.put(Release(sess))

    def _release_now(self, sess):
        slot, sess.slot = sess.slot, None
        if slot is not None:
            with self.free_lock:
                self.free.append(slot)
                self.live_slots -= 1

    # ------------------------------------------------------------------ submit / loop
    def submit(self, chunk: Chunk):
        self.q.put(chunk)

    def _drain(self, block):
        """Take up to b_max queued chunks: block for the first one unless a tick is in flight, then
        coalesce for up to wait_ms or until min(b_target, live sessions) rows are queued (per-row GPU
        cost bottoms out at B~16; whatever else is already queued is still taken, up to b_max). Release
        markers are applied as they come."""
        items = []
        wait = self.wait_ms / 1000.0
        # Rows can only come from live sessions: a lone stream never waits for a 16-row tick.
        target = min(self.b_target, max(1, getattr(self, "live_slots", self.b_target)))

        def take(it):
            if isinstance(it, Release):
                self._release_now(it.sess)
                return
            if getattr(it.sess, "closed", False):
                it.error = RuntimeError("session closed")
                finish_all([it])
                return
            items.append(it)

        if block:
            take(self.q.get())
            self.phases.mark("idle")
        deadline = time.perf_counter() + wait
        # While a tick is in flight, poll its event every 2 ms between queue reads so its rows are
        # delivered as soon as the GPU is done (not after the full window) and the next tick keeps
        # coalescing meanwhile.
        while len(items) < self.b_max:
            rem = deadline - time.perf_counter()
            if len(items) >= target:
                timeout = 0.0
            elif self._inflight is not None:
                if wait > 0 and rem <= 0:
                    break
                timeout = 0.002 if wait <= 0 else max(0.0005, min(rem, 0.002))
            elif wait > 0:
                if rem <= 0:
                    break
                timeout = rem
            else:
                timeout = 0.0
            try:
                take(self.q.get(timeout=timeout) if timeout > 0 else self.q.get_nowait())
            except queue.Empty:
                if self._inflight is not None and self._inflight[0]["ev"].query():
                    self._complete()
                    if not items and wait <= 0:
                        break
                    continue
                if self._inflight is None and (wait <= 0 or len(items) >= target):
                    break
                if len(items) >= target:
                    break
        self.phases.mark("coalesce")
        return items

    def _groups(self, items):
        # A session awaits its chunk before submitting the next, so rows are distinct sessions by
        # construction; defer duplicates anyway rather than step one state twice in a tick.
        seen, groups = set(), collections.OrderedDict()
        for it in items:
            if id(it.sess) in seen:
                self.q.put(it)
                continue
            seen.add(id(it.sess))
            groups.setdefault(it.key(), []).append(it)
        return list(groups.values())

    def run(self):
        next_log = 2000
        with self.torch.inference_mode():
            while True:
                try:
                    self._loop_once()
                except Exception as e:  # noqa: BLE001 - an async CUDA fault surfaces at the event wait
                    logger.exception("%s stepper: iteration failed", self.profile)
                    self._fail_inflight(e)
                if self.ticks >= next_log:
                    next_log = self.ticks + 2000
                    logger.info("%s stepper: %s", self.profile, self.summary())

    def _fail_inflight(self, e):
        inflight, self._inflight = self._inflight, None
        if inflight is not None:
            for it in inflight[1]:
                it.error = e
            self._account(inflight[1], inflight[2], time.perf_counter())
            finish_all(inflight[1])

    def _loop_once(self):
        self.phases.reset()
        items = self._drain(block=self._inflight is None)
        if not items:
            if self._inflight is not None:
                self._complete()
            return
        for batch in self._groups(items):
            t0 = time.perf_counter()
            for it in batch:
                it.t_start = t0
                it.batch = len(batch)
                self.wait_ms_hist.append((t0 - it.t_submit) * 1000)
            try:
                if self.tick_mode == "fused":
                    self._issue(batch)
                else:
                    self.tick_legacy(batch)
            except Exception as e:  # noqa: BLE001 - every row of the group fails together
                logger.exception("%s stepper: tick failed (B=%d)", self.profile, len(batch))
                if self._inflight is not None and self._inflight[1] is batch:
                    self._inflight = None
                for it in batch:
                    it.error = e
                self._account(batch, t0, time.perf_counter())
                finish_all(batch)
                continue
            if self.tick_mode != "fused":
                self._account(batch, t0, time.perf_counter())
                finish_all(batch)

    def _account(self, batch, t0, t1):
        self.t_busy += t1 - t0
        self.tick_ms.append((t1 - t0) * 1000)
        self.ticks += 1
        self.rows += len(batch)
        self.b_hist[len(batch)] += 1
        for it in batch:
            it.t_end = t1

    def _recent_span(self):
        return (self.gpu_t[-1] - self.gpu_t[0]) if len(self.gpu_t) > 1 else 0.0

    def summary(self):
        up = max(1e-9, time.time() - self.t_started)
        if not self.tick_ms:
            return {"ticks": 0}
        tm = sorted(self.tick_ms)
        wm = sorted(self.wait_ms_hist)
        gm = sorted(self.gpu_ms)

        def pct(v, p):
            return round(v[min(len(v) - 1, int(len(v) * p))], 2) if v else None
        # busy_frac: the thread's time outside its blocking queue wait (comparable across modes);
        # tick_busy_frac: summed tick wall over uptime (the legacy definition; > 1 possible when ticks overlap).
        out = {"mode": self.tick_mode, "ticks": self.ticks, "rows": self.rows,
               "busy_frac": round(1.0 - self.phases.t["idle"] / up, 3), "tick_busy_frac": round(self.t_busy / up, 3),
               "rows_per_tick": round(self.rows / max(1, self.ticks), 2),
               "tick_ms_p50": pct(tm, .5), "tick_ms_p99": pct(tm, .99), "wait_ms_p50": pct(wm, .5),
               "wait_ms_p99": pct(wm, .99), "b_hist": dict(sorted(self.b_hist.items())),
               "eager_ticks": self.eager_ticks, "phases": self.phases.table(self.ticks)}
        if gm:
            out.update({"gpu_ms_p50": pct(gm, .5), "gpu_ms_p99": pct(gm, .99),
                        "gpu_ms_per_row_recent": round(sum(self.gpu_ms) / max(1, sum(self.gpu_rows)), 3),
                        "gpu_busy_frac_recent": round(sum(self.gpu_ms) / 1000.0 / max(1e-9, self._recent_span()), 3)})
        cache = self.fused if self.tick_mode == "fused" else self.core
        out["graphs"] = len(cache.graphs) if cache is not None else 0
        out["eager_calls"] = cache.eager_calls if cache is not None else None
        if self.tick_mode == "fused":
            out.update({"live_slots": self.live_slots, "slots": self.n_slots, "slot_fail": self.slot_fail,
                        "b_min": self.b_min, "pad_rows": self.pad_rows})
        return out

    # ------------------------------------------------------------------ fused tick
    def _steady_consts(self):
        left_off, right_off, f_true, disc = self.clc * SUB, self.rc * SUB, self.f_steady, LPAD
        return (left_off, right_off, f_true, disc), (f_true - 1 + disc) * HOP + WIN

    def _mel_dev(self, sig, disc, f_true):
        """Preprocessor on device windows [B, n] -> features [B, f_steady, 128] (padded past f_true)."""
        torch = self.torch
        ln = torch.full((sig.shape[0],), sig.shape[1], dtype=torch.int64, device=self.device)
        feats, _ = self.m.preprocessor(input_signal=sig, length=ln)          # [B, 128, T]
        feats_t = feats[:, :, disc: disc + f_true].transpose(1, 2).contiguous()
        if f_true < self.f_steady:
            feats_t = torch.nn.functional.pad(feats_t, (0, 0, 0, self.f_steady - f_true))
        return feats_t

    def _fused(self, idx, sig, flen, left_off, right_off, f_true, disc):
        """Whole step for B rows whose state sits at table rows ``idx``: mel -> gather -> core -> update
        -> write-back. Pure tensor ops, no host sync (capturable)."""
        torch, m, sm = self.torch, self.m, self.sm
        feats_t = self._mel_dev(sig, disc, f_true)
        st = types.SimpleNamespace(**{k: self.tables[k].index_select(0, idx) for k in STATE_FIELDS})
        preds, enc_lens, chunk_embs, chunk_lens = self._core_eager(
            feats_t, flen, st.spkcache, st.spkcache_lengths, st.fifo, st.fifo_lengths)
        sub = m.encoder.subsampling_factor
        lc_enc = round(left_off / sub)
        rc_enc = -(-right_off // sub)
        high_res = None
        if m.high_resolution:
            high_res = preds
            preds = sm.downsample_preds(high_res, m.upsample_factor)
        n_frames = preds.shape[1]
        mask = torch.arange(n_frames, device=preds.device).view(1, -1, 1) < enc_lens.view(-1, 1, 1)
        preds = preds.masked_fill(~mask, 0.0)                                  # == apply_mask_to_preds
        saved_sc, saved_f = st.spkcache_lengths.clone(), st.fifo_lengths.clone()
        max_chunk_len = chunk_embs.shape[1] - lc_enc - rc_enc
        cl = (chunk_lens - lc_enc).clamp(min=0, max=max_chunk_len)
        chunk_preds = self._update_async(st, chunk_embs, cl, preds, lc_enc, rc_enc, max_chunk_len)
        if m.high_resolution:
            chunk_preds = m._extract_async_high_resolution_chunk_preds(
                high_resolution_preds=high_res, spkcache_lengths=saved_sc, fifo_lengths=saved_f,
                chunk_lengths=cl, max_chunk_len=max_chunk_len, lc_enc=lc_enc)
        native = 1 if m.high_resolution else sub
        ds = m.output_subsampling_factor // native
        if ds > 1:
            chunk_preds = sm.downsample_preds(chunk_preds, ds)
        for k in STATE_FIELDS:
            self.tables[k].index_copy_(0, idx, getattr(st, k))
        return (chunk_preds.float(),)

    def _update_async(self, st, chunk, cl, preds, lc, rc, max_chunk_len):
        """``SortformerModules.streaming_update_async`` on a namespace of gathered rows, with the
        speaker-cache compression step made sync-free (see ``_update_spkcache_all``)."""
        sm = self.sm
        max_spkcache_len, max_fifo_len = st.spkcache.shape[1], st.fifo.shape[1]
        max_pop_out_len = max(sm.spkcache_update_period, max_fifo_len, max_chunk_len)
        max_pop_out_len = min(max_pop_out_len, max_chunk_len + max_fifo_len)
        cur_sc_preds, cur_fifo_preds, chunk_preds = sm._gather_async_predictions(
            st, preds, st.spkcache_lengths, st.fifo_lengths, cl, max_spkcache_len, max_fifo_len, max_chunk_len, lc)
        pop_len, new_fifo_len = sm._compute_async_fifo_pop_lengths(st.spkcache_lengths, st.fifo_lengths, cl, max_fifo_len)
        pop_embs, pop_preds, valid_pop = sm._update_async_fifo(
            st, chunk, cur_fifo_preds, chunk_preds, st.fifo_lengths, pop_len, new_fifo_len, max_chunk_len, max_pop_out_len, lc)
        sm._update_async_silence_profile(st, pop_embs, pop_preds, valid_pop)
        self._update_spkcache_all(st, cur_sc_preds, pop_embs, pop_preds, pop_len)
        return chunk_preds

    def _update_spkcache_all(self, st, cur_sc_preds, pop_embs, pop_preds, pop_len):
        """``_update_async_spkcache`` with compression run for every row and selected by ``where``."""
        torch, sm = self.torch, self.sm
        batch_size, max_pop_out_len, emb_dim = pop_embs.shape
        n_spk = pop_preds.shape[2]
        max_spkcache_len = st.spkcache.shape[1]
        sc_len = st.spkcache_lengths
        upd_len = sc_len + pop_len
        need = upd_len > sm.spkcache_len
        first = (~st.spkcache_compressed) & need
        cand_old_preds = torch.where(first.view(-1, 1, 1), cur_sc_preds, st.spkcache_preds)
        cand_embs = torch.cat([st.spkcache, pop_embs, st.spkcache.new_zeros((batch_size, 1, emb_dim))], dim=1)
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
        emb_g = torch.where(disabled.unsqueeze(-1), mean_sil_emb.unsqueeze(1).expand(-1, sm.spkcache_len, -1), emb_g)
        preds_g = torch.gather(preds, 1, top_s.unsqueeze(-1).expand(-1, -1, n_spk))
        preds_g = torch.where(disabled.unsqueeze(-1), 0.0, preds_g)
        return emb_g, preds_g

    def _ring_init(self, out_shape):
        torch = self.torch
        consts, seg_len = self._steady_consts()
        mk = lambda *s, dt=torch.float32: torch.empty(s, dtype=dt, pin_memory=True)  # noqa: E731
        self._ring = [{"sig": mk(self.b_max, seg_len), "idx": mk(self.b_max, dt=torch.long),
                       "flen": mk(self.b_max, dt=torch.long), "out": mk(self.b_max, *out_shape),
                       "ev": torch.cuda.Event(), "ev0": torch.cuda.Event(enable_timing=True),
                       "ev1": torch.cuda.Event(enable_timing=True)} for _ in range(self.RING)]
        self._ring_i = 0

    def _issue(self, items):
        """Pack a group into pinned buffers, enqueue H2D + replay + D2H, and leave it in flight (steady
        geometry); non-steady geometries run eagerly and are delivered at once."""
        torch = self.torch
        ph = self.phases
        for it in items:
            if it.sess.slot is None:
                self._acquire(it.sess)
        it0 = items[0]
        consts = (it0.left_off, it0.right_off, it0.f_true, it0.disc)
        steady = consts == self._steady_consts()[0] and self.fused is not None and not self.fused.disabled
        B = len(items)
        if not steady:
            idx = torch.tensor([it.sess.slot for it in items], dtype=torch.long)
            sig = torch.from_numpy(np.stack([it.seg for it in items]))
            flen = torch.full((B,), it0.f_true, dtype=torch.long)
            ph.mark("pack")
            if self.fused is not None:
                outs = self.fused.run([idx, sig, flen], consts, capture=False)
            else:
                outs = self._fused(idx.to(self.device), sig.to(self.device), flen.to(self.device), *consts)
            out = outs[0].cpu().numpy()
            self.eager_ticks += 1
            ph.mark("eager")
            for i, it in enumerate(items):
                it.result = out[i]
            if self._inflight is not None:
                self._complete()                     # keep delivery order: the earlier tick first
            self._account(items, it0.t_start, time.perf_counter())
            finish_all(items)
            ph.mark("deliver")
            return
        if self._ring is None:
            self._ring_init(self._probe_out_shape())
        r = self._ring[self._ring_i]
        if self._inflight is not None and self._inflight[0] is r:
            self._complete()
        self._ring_i = (self._ring_i + 1) % self.RING
        r["sig"][:B].copy_(torch.from_numpy(np.stack([it.seg for it in items])))
        r["idx"][:B].copy_(torch.tensor([it.sess.slot for it in items], dtype=torch.long))
        r["flen"][:B].fill_(it0.f_true)
        Bp = max(B, self.b_min)
        if Bp > B:
            r["sig"][B:Bp].zero_()
            r["idx"][B:Bp].copy_(torch.arange(self.n_slots, self.n_slots + Bp - B, dtype=torch.long))
            r["flen"][B:Bp].fill_(it0.f_true)
            self.pad_rows += Bp - B
        ph.mark("pack")
        r["ev0"].record()
        outs = self.fused.run([r["idx"][:Bp], r["sig"][:Bp], r["flen"][:Bp]], consts)
        r["ev1"].record()
        r["out"][:B].copy_(outs[0][:B], non_blocking=True)     # pad rows (if any) are discarded
        r["ev"].record()
        ph.mark("issue")
        prev, self._inflight = self._inflight, (r, items, it0.t_start)
        if prev is not None:
            self._complete(prev)

    def _complete(self, inflight=None):
        """Wait for a tick's event (GIL released), copy its rows out of the pinned buffer, resolve."""
        ph = self.phases
        if inflight is None:
            inflight, self._inflight = self._inflight, None
        elif inflight is self._inflight:
            self._inflight = None
        r, items, t0 = inflight
        r["ev"].synchronize()
        ph.mark("wait")
        try:
            self.gpu_ms.append(r["ev0"].elapsed_time(r["ev1"]))
            self.gpu_rows.append(len(items))
            self.gpu_t.append(time.perf_counter())
        except Exception:  # noqa: BLE001 - timing is diagnostics only
            pass
        out = r["out"][: len(items)].numpy()
        for i, it in enumerate(items):
            it.result = out[i].copy()
        self._account(items, t0, time.perf_counter())
        finish_all(items)
        ph.mark("deliver")

    def _probe_out_shape(self):
        """Output frame count of the steady geometry (from the captured graph or one eager call)."""
        torch = self.torch
        consts, seg_len = self._steady_consts()
        scratch = torch.arange(self.n_slots, self.n_slots + 1, device=self.device)
        outs = self._fused(scratch, torch.zeros((1, seg_len), device=self.device),
                           torch.full((1,), self.f_steady, dtype=torch.long, device=self.device), *consts)
        return tuple(outs[0].shape[1:])

    # ------------------------------------------------------------------ legacy tick
    def _mel(self, items):
        """One preprocessor call for the group's raw windows (equal length by construction)."""
        torch = self.torch
        it0 = items[0]
        segs = np.stack([it.seg for it in items])                       # [B, n]
        sig = torch.from_numpy(segs).to(self.device)
        feats_t = self._mel_dev(sig, it0.disc, it0.f_true)
        flen = torch.full((segs.shape[0],), it0.f_true, dtype=torch.int64, device=self.device)
        return feats_t, flen

    def _gather(self, items):
        st = self._State()
        for it in items:
            if it.sess.state is None:
                it.sess.state = self.sm.init_streaming_state(batch_size=1, async_streaming=True, device=self.device)
        for name in STATE_FIELDS:
            setattr(st, name, self.torch.cat([getattr(it.sess.state, name) for it in items]))
        return st

    def _scatter(self, items, st):
        for i, it in enumerate(items):
            s = it.sess.state
            for name in STATE_FIELDS + ("fifo_preds",):
                v = getattr(st, name)
                setattr(s, name, v[i:i + 1] if v is not None else None)

    def _core_eager(self, feats_t, flen, spkcache, spkcache_lengths, fifo, fifo_lengths):
        """pre-encode -> concat+pad (fixed l_max) -> encoder -> head. Capture-safe: no host scalars."""
        m, dt = self.m, self.core_dtype
        if dt != self.torch.float32:
            feats_t, spkcache, fifo = feats_t.to(dt), spkcache.to(dt), fifo.to(dt)
        chunk_embs, chunk_lens = m._call_pre_encode(feats_t, flen)
        embs, lens = concat_and_pad_dev(self.torch, [spkcache, fifo, chunk_embs],
                                        [spkcache_lengths, fifo_lengths, chunk_lens], self.l_max)
        emb_seq, emb_len = m.frontend_encoder(processed_signal=embs, processed_signal_length=lens,
                                              bypass_pre_encode=True)
        preds = m.forward_infer(emb_seq=emb_seq, emb_seq_length=emb_len)
        return preds.float(), emb_len, chunk_embs.float(), chunk_lens

    def run_core(self, *args):
        if self.core is not None:
            return self.core.run(list(args))
        return self._core_eager(*args)

    def tick_legacy(self, items):
        """``forward_streaming_step``'s async branch for B rows at once; per-row results as numpy."""
        torch, m, sm = self.torch, self.m, self.sm
        ph = self.phases
        it0 = items[0]
        feats_t, flen = self._mel(items)
        ph.mark("mel")
        st = self._gather(items)
        ph.mark("gather")
        preds, enc_lens, chunk_embs, chunk_lens = self.run_core(
            feats_t, flen, st.spkcache, st.spkcache_lengths, st.fifo, st.fifo_lengths)
        if self.prof_sync:
            torch.cuda.synchronize()
        ph.mark("core")
        sub = m.encoder.subsampling_factor
        lc_enc = round(it0.left_off / sub)
        rc_enc = -(-it0.right_off // sub)
        high_res = None
        if m.high_resolution:
            high_res = preds
            preds = sm.downsample_preds(high_res, m.upsample_factor).detach()
        preds = sm.apply_mask_to_preds(preds, enc_lens)
        saved_sc, saved_f = st.spkcache_lengths.clone(), st.fifo_lengths.clone()
        ph.mark("post")
        st, chunk_preds = sm.streaming_update_async(streaming_state=st, chunk=chunk_embs, chunk_lengths=chunk_lens,
                                                    preds=preds, lc=lc_enc, rc=rc_enc)
        ph.mark("update")
        if m.high_resolution:
            max_chunk_len = chunk_embs.shape[1] - lc_enc - rc_enc
            cl = (chunk_lens - lc_enc).clamp(min=0, max=max_chunk_len)
            chunk_preds = m._extract_async_high_resolution_chunk_preds(
                high_resolution_preds=high_res, spkcache_lengths=saved_sc, fifo_lengths=saved_f,
                chunk_lengths=cl, max_chunk_len=max_chunk_len, lc_enc=lc_enc)
        native = 1 if m.high_resolution else sub
        ds = m.output_subsampling_factor // native
        if ds > 1:
            chunk_preds = sm.downsample_preds(chunk_preds, ds)
        ph.mark("hr_extract")
        self._scatter(items, st)
        ph.mark("scatter")
        out = chunk_preds.float().cpu().numpy()                         # one D2H for the tick
        ph.mark("d2h")
        for i, it in enumerate(items):
            it.result = out[i]

    # ------------------------------------------------------------------ warm-up
    def warm_graphs(self):
        """Capture the graph for every batch size the stepper can issue (steady geometry)."""
        torch, sm = self.torch, self.sm
        t0 = time.perf_counter()
        with torch.inference_mode():
            if self.tick_mode == "fused":
                if self.fused is None:
                    return 0
                consts, seg_len = self._steady_consts()
                for b in range(1, self.b_max + 1):
                    idx = torch.arange(self.n_slots, self.n_slots + b, dtype=torch.long)
                    sig = torch.zeros((b, seg_len))
                    flen = torch.full((b,), self.f_steady, dtype=torch.long)
                    self.fused.run([idx, sig, flen], consts)
                for t in self.tables.values():
                    t[self.n_slots:].zero_()
                n = len(self.fused.graphs)
            else:
                if self.core is None:
                    return 0
                for b in range(1, self.b_max + 1):
                    st = sm.init_streaming_state(batch_size=b, async_streaming=True, device=self.device)
                    feats = torch.zeros((b, self.f_steady, 128), device=self.device)
                    flen = torch.full((b,), self.f_steady, dtype=torch.int64, device=self.device)
                    self.core.run([feats, flen, st.spkcache, st.spkcache_lengths, st.fifo, st.fifo_lengths])
                n = len(self.core.graphs)
        torch.cuda.synchronize()
        logger.info("%s: %d %s graphs captured in %.1fs", self.profile, n, self.tick_mode, time.perf_counter() - t0)
        return n

    def bench_fused(self, sizes=(1, 4, 8, 16, 32, 64), reps=20):
        """GPU ms per tick of the steady fused graph at each batch size (scratch slots, zero audio)."""
        torch = self.torch
        if self.tick_mode != "fused" or self.fused is None:
            return []
        consts, seg_len = self._steady_consts()
        rows = []
        with torch.inference_mode():
            for b in sizes:
                if b > self.b_max:
                    continue
                idx = torch.arange(self.n_slots, self.n_slots + b, dtype=torch.long)
                sig = torch.randn((b, seg_len)) * 0.02
                flen = torch.full((b,), self.f_steady, dtype=torch.long)
                for _ in range(3):
                    self.fused.run([idx, sig, flen], consts)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(reps):
                    self.fused.run([idx, sig, flen], consts)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / reps * 1000
                rows.append((b, round(ms, 2), round(ms / b, 3)))
            for t in self.tables.values():
                t[self.n_slots:].zero_()
        return rows


class SyncGraphs:
    """Sync-mode diarizer step with the eager core captured in a CUDA graph per distinct sequence length.

    Sync mode's speaker cache and FIFO grow from empty to capacity, so the encoder input walks
    ~(spkcache_len + fifo_len) / chunk_len distinct lengths (plus FIFO cycling) per session; each is
    captured once (at load, by the warm session) and replayed. Replay runs exactly the eager kernels,
    so the output is bit-identical to NeMo's sync path. One connection per call; callers serialise
    calls on the same instance (static buffers).
    """

    def __init__(self, torch, model):
        self.torch, self.m = torch, model
        self.sm = model.sortformer_modules
        self.lock = threading.Lock()
        self.graphs = GraphCache(torch, self._core, "sync core", model.device)

    def _core(self, embs, lens):
        emb_seq, emb_len = self.m.frontend_encoder(processed_signal=embs, processed_signal_length=lens,
                                                   bypass_pre_encode=True)
        return self.m.forward_infer(emb_seq=emb_seq, emb_seq_length=emb_len), emb_len

    def step(self, feats_t, flen, st, left_off, right_off):
        """``forward_streaming_step`` sync branch (no logits); returns (state, chunk_preds)."""
        m, sm = self.m, self.sm
        with self.lock, self.torch.inference_mode():
            chunk_embs, chunk_lens = m._call_pre_encode(feats_t, flen)
            embs = sm.concat_embs([st.spkcache, st.fifo, chunk_embs], dim=1, device=chunk_embs.device)
            lens = st.spkcache.shape[1] + st.fifo.shape[1] + chunk_lens
            preds, enc_lens = self.graphs.run([embs, lens])
            sub = m.encoder.subsampling_factor
            lc_enc = round(left_off / sub)
            rc_enc = -(-right_off // sub)
            high_res = None
            if m.high_resolution:
                high_res = preds
                preds = sm.downsample_preds(high_res, m.upsample_factor).detach()
            preds = sm.apply_mask_to_preds(preds, enc_lens)
            saved_sc, saved_f = st.spkcache.shape[1], st.fifo.shape[1]
            st, chunk_preds = sm.streaming_update(streaming_state=st, chunk=chunk_embs, preds=preds, lc=lc_enc, rc=rc_enc)
            if m.high_resolution:
                chunk_len = chunk_embs.shape[1] - lc_enc - rc_enc
                start = (saved_sc + saved_f + lc_enc) * m.upsample_factor
                chunk_preds = high_res[:, start: start + chunk_len * m.upsample_factor]
            native = 1 if m.high_resolution else sub
            ds = m.output_subsampling_factor // native
            if ds > 1:
                chunk_preds = sm.downsample_preds(chunk_preds, ds)
            return st, chunk_preds.float().cpu().numpy()[0]


class TurnTracker:
    """Incremental speaker turns: same output as thresholding the whole prediction history, finding
    per-speaker runs, merging same-speaker runs across gaps <= MERGE_GAP_S and dropping turns shorter
    than MIN_TURN_S -- but fed one chunk at a time, so the per-step cost does not grow with the call.
    The gap and length tests use the same float expressions as the whole-history version.

    Only the last run of a speaker can still grow, so every earlier run is frozen once with its JSON
    fragment. A frozen run whose start precedes every still-growing run can never be preceded by a
    later one, so it is appended to a committed prefix string once; a replace-style partial is then
    that prefix + a sort of the few uncommitted fragments -- O(1) in the call length, where a
    ``json.dumps`` of the whole (hour-long: ~2300 turns) list was ~3 ms per partial and a sort of the
    whole fragment list ~0.5 ms, both on the GIL at hundreds of partials per second."""

    def __init__(self, n_spk, threshold):
        self.threshold = threshold
        self.n_spk = n_spk
        self.last = [None] * n_spk                # per speaker: [start_frame, end_frame] of the growing run
        self.first = [None] * n_spk               # per speaker: start frame of its first run (tie order)
        self.has_frozen = [False] * n_spk         # per speaker: a kept (>= MIN_TURN_S) frozen run exists
        self.committed = []                       # fragments in final order (source of truth)
        self._committed_text = ""                 # ", ".join(committed) cached; rebuilt only when it grows
        self._committed_n = 0
        self.tail = []                            # (sort key, fragment): frozen, position not yet final
        self.frames = 0

    @staticmethod
    def _frag(s, e, spk):
        return json.dumps({"start": round(s * FRAME_S, 3), "end": round(e * FRAME_S, 3), "speaker": f"speaker_{spk}"})

    def _key(self, s, spk):
        # Whole-history order: sort by start, ties by speaker first appearance (stable sort over
        # speakers listed in that order).
        return (s, self.first[spk], spk)

    def _freeze(self, spk, run):
        s, e = run
        if e * FRAME_S - s * FRAME_S >= MIN_TURN_S:
            self.tail.append((self._key(s, spk), self._frag(s, e, spk)))
            self.has_frozen[spk] = True

    def add(self, preds: np.ndarray):
        """preds: [T, n_spk] probabilities for frames [self.frames, self.frames + T)."""
        active = preds > self.threshold
        base = self.frames
        for spk in range(active.shape[1]):
            a = active[:, spk]
            if not a.any():
                continue
            d = np.diff(np.concatenate(([0], a.astype(np.int8), [0])))
            last = self.last[spk]
            for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                s, e = int(s) + base, int(e) + base
                if last is not None and s * FRAME_S - last[1] * FRAME_S <= MERGE_GAP_S:
                    if e > last[1]:
                        last[1] = e
                else:
                    if last is not None:
                        self._freeze(spk, last)
                    last = [s, e]
                    if self.first[spk] is None:
                        self.first[spk] = s
            self.last[spk] = last
        self.frames += preds.shape[0]
        if self.tail:
            self._commit()

    def _commit(self):
        # Every future fragment is a current growing run or starts at/after self.frames.
        bar = min((r[0] for r in self.last if r is not None), default=self.frames)
        if not any(k[0] < bar for k, _ in self.tail):
            return
        self.tail.sort()
        i = 0
        while i < len(self.tail) and self.tail[i][0][0] < bar:
            self.committed.append(self.tail[i][1])
            i += 1
        del self.tail[:i]

    def committed_text(self):
        """", ".join(committed)", extended incrementally (one concat per newly committed run)."""
        n = len(self.committed)
        if n != self._committed_n:
            new = ", ".join(self.committed[self._committed_n:n])
            self._committed_text = new if not self._committed_text else self._committed_text + ", " + new
            self._committed_n = n
        return self._committed_text

    def turns_json(self):
        """The turns array as JSON text (identical bytes to json.dumps of the dict list) + speaker count."""
        items = list(self.tail)
        speakers = 0
        for spk in range(self.n_spk):
            last = self.last[spk]
            grow = last is not None and last[1] * FRAME_S - last[0] * FRAME_S >= MIN_TURN_S
            if grow:
                items.append((self._key(last[0], spk), self._frag(last[0], last[1], spk)))
            speakers += self.has_frozen[spk] or grow
        items.sort()
        head = self.committed_text()
        tail = ", ".join(f for _, f in items)
        body = head + ", " + tail if head and tail else head or tail
        return "[" + body + "]", speakers

    def turns(self):
        text, _ = self.turns_json()
        return json.loads(text)


def patch_attention_sdpa(torch, encoder):
    """``NEMO_DIAR_ATTN=sdpa``: run the encoder's self-attention through
    ``F.scaled_dot_product_attention`` instead of FlexAttention, with the same semantics.

    NeMo's ``TransformerEncoder`` already materialises the Transformer-XL relative-position bias as a
    dense ``(B, H, T, T)`` tensor (captured by its ``score_mod`` closure) and folds the content bias
    into the query; the padding ``mask_mod`` is ``kv_idx < length[b]``. Here ``create_block_mask`` is
    replaced (in the module namespace ``forward_internal`` resolves it from) by a dense bool mask built
    by evaluating the same ``mask_mod`` on index grids, and every layer's attention forward adds the
    two as one additive ``attn_mask``. Only padding-only attention is supported (``attn_mode='full'``,
    no causal tail); anything else leaves FlexAttention in place. Returns the number of patched layers."""
    import types

    from nemo.collections.asr.modules import transformer_encoder as te

    if not isinstance(encoder, te.TransformerEncoder) or getattr(encoder, "attn_mode", "full") != "full" \
            or getattr(encoder, "causal_tail_len", 0):
        logger.warning("NEMO_DIAR_ATTN=sdpa: encoder %s is not padding-only full attention; keeping FlexAttention",
                       type(encoder).__name__)
        return 0

    def dense_mask(mask_mod, B, H, Q_LEN, KV_LEN, device, **_):
        b = torch.arange(B, device=device).view(B, 1, 1, 1)
        h = torch.zeros((1, 1, 1, 1), dtype=torch.long, device=device)
        q = torch.arange(Q_LEN, device=device).view(1, 1, Q_LEN, 1)
        kv = torch.arange(KV_LEN, device=device).view(1, 1, 1, KV_LEN)
        return mask_mod(b, h, q, kv)                                  # (B, 1, Q, KV) bool: True = attend

    te.create_block_mask = dense_mask

    def forward(self, x, block_mask=None, pos_emb=None):
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        qkv = self.w_qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_norm:
            q = self.q_norm(q).to(v.dtype)
            k = self.k_norm(k).to(v.dtype)
        if self._uses_rope:
            q, k = self.rope(q, k)
        bias = None
        if self._uses_rel_pos:
            score_mod, q = self._build_rel_pos_score_mod(q, pos_emb)
            bias = score_mod._relative_position_bias.to(q.dtype)      # (B, H, T, T), already 1/sqrt(D)-scaled
        if block_mask is not None:
            pad = torch.zeros(block_mask.shape, dtype=q.dtype, device=q.device).masked_fill(~block_mask, float("-inf"))
            bias = pad if bias is None else bias + pad
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)

    n = 0
    for layer in encoder.layers:
        layer.attn.forward = types.MethodType(forward, layer.attn)
        n += 1
    logger.info("attention: F.scaled_dot_product_attention on %d layers (dense rel-pos bias + padding mask)", n)
    return n


class GpuSampler(threading.Thread):
    """Polls nvidia-smi every second; keeps (t, util%, mem MiB) samples (diagnostics, NEMO_DIAR_PROF=1)."""

    def __init__(self, torch):
        super().__init__(daemon=True, name="gpu-sampler")
        self.samples = collections.deque(maxlen=7200)
        self._torch = torch

    def run(self):
        try:
            p = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                  "--format=csv,noheader,nounits", "-l", "1"], stdout=subprocess.PIPE, text=True)
            for line in p.stdout:
                try:
                    u, mem = [float(x) for x in line.strip().split(",")]
                except ValueError:
                    continue
                self.samples.append((round(time.time(), 2), u, mem))
        except Exception as e:  # noqa: BLE001 - fall back to torch's NVML wrapper
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
