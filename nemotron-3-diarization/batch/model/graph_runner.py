"""Whole-file batched diarization with every streaming step replayed from a CUDA graph.

NeMo's ``diarize()`` runs ``forward_streaming``: one ``forward_streaming_step`` per chunk for the
whole batch, each step re-encoding [speaker cache | FIFO | chunk] through the 31-layer encoder and
then updating the cache/FIFO (NeMo's *sync* state: cache and FIFO grow from empty, so their shapes
change from step to step and cycle once the FIFO is full). At bs<=16 that step is launch-bound
(~600 kernels for a few ms of GPU work) and ``torch.compile`` only halves the gap; it also changes
the numbers (Inductor's FlexAttention rounds differently in bf16 and the arrival-ordered speaker
cache amplifies that into confusion errors).

This runner keeps NeMo's arithmetic and removes the launches: the whole step (pre-encode, concat,
encoder, head, mask, cache/FIFO update incl. compression) is captured once per distinct
``(batch, cache_len, fifo_len, chunk_frames, compressed)`` shape and replayed. The step reads and
writes persistent per-batch-size state buffers, so a replay is one launch and the state never moves.
Replay runs exactly the eager kernels, so in fp32 the output is bit-identical to NeMo's eager sync
path; bf16/fp16 cores use the same graphs with the weights cast. ``low`` has 129 steady shapes,
``ultralow`` 237, ``offline`` 2 (``simkeys``), all captured at load for the configured batch sizes;
a batch of n requests runs at the next captured size (rows duplicated), so no request pays a
capture. Non-steady shapes (a file's last chunk) run the same function eagerly.
"""

import contextlib
import logging
import math
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

SUB = 8  # encoder subsampling: 1 pre-encoded frame = 8 mel frames (80 ms)


class _Timers:
    def __init__(self):
        self.t = {}

    @contextlib.contextmanager
    def __call__(self, key):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[key] = self.t.get(key, 0.0) + time.perf_counter() - t0


class GraphRunner:
    """Batched sync-mode inference for one ``SortformerEncLabelModel`` instance.

    ``run(audios)`` -> (list of [T, n_spk] float32 CPU prediction tensors, timing dict). Callers
    serialise calls on one runner (static buffers); different runners (profiles) may run
    concurrently, each on its own CUDA stream.
    """

    def __init__(self, torch, model, *, dtype=None, graphs=True, sizes=(1, 2, 4, 8, 16, 32),
                 compile_encoder=True, attn_fp32=False, attn_mode=None, post_fp32=False, pre_fp32=False,
                 head_fp32=False, name="runner"):
        self.torch, self.m, self.name = torch, model, name
        self.sm = model.sortformer_modules
        self.device = model.device
        self.dtype = dtype or torch.float32          # core (encoder) dtype
        self.use_graphs = graphs
        # post_fp32: autocast only around the encoder; pre-encode, state, head and cache update in fp32
        self.post_fp32 = post_fp32 and self.dtype != torch.float32
        # bisect islands: pre-encode alone / head (final norm, projection, sigmoid, 8-frame pooling) alone
        self.pre_fp32 = (pre_fp32 or self.post_fp32) and self.dtype != torch.float32
        self.head_fp32 = (head_fp32 or self.post_fp32) and self.dtype != torch.float32
        self.state_dtype = torch.float32 if self.post_fp32 else self.dtype
        self.attn_mode = attn_mode or ("fp32" if attn_fp32 else None)
        self.hybrid = self.attn_mode == "hybrid"
        self.attn_fp32 = self.attn_mode == "fp32"
        if self.attn_mode:
            patch_attention(torch, model.encoder, self.attn_mode, self.dtype)
        if compile_encoder:
            # Inductor's fused kernels are what make the step fast on the GPU; the graph replay then
            # removes the launch overhead they leave behind (NeMo's script stops at compile). Compiling
            # the whole core (pre-encode, head, mask) as well was measured slower (2.30 vs 2.08 s).
            torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
            model.encoder = torch.compile(model.encoder, dynamic=True)
        self.sizes = tuple(sorted(sizes))
        self.stream = torch.cuda.Stream(device=self.device)
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs = {}                              # key -> {"graph", "out"}
        self.bufs = {}                                # B -> persistent state / input buffers
        self.eager_steps = self.replays = 0
        self.lock = threading.Lock()
        sm = self.sm
        if not sm.use_learnable_sil_emb:
            raise RuntimeError("GraphRunner assumes use_learnable_sil_emb (this checkpoint); the mean-silence "
                               "profile update has a host sync")
        self.clc = getattr(sm, "chunk_left_context", 0)
        self.chunk_len, self.rc = sm.chunk_len, sm.chunk_right_context
        self.upsample = model.upsample_factor if model.high_resolution else 1
        self.out_ds = model.output_subsampling_factor // (1 if model.high_resolution else SUB)
        self.tc_steady = (self.clc + self.chunk_len + self.rc) * SUB       # mel frames per steady chunk
        self.n_spk = sm.n_spk
        self.enc_autocast = (torch.autocast(device_type="cuda", dtype=self.dtype) if self.dtype != torch.float32
                             else contextlib.nullcontext())
        self.autocast = contextlib.nullcontext() if self.post_fp32 else self.enc_autocast

    # ------------------------------------------------------------------ buffers
    def _bufs(self, B, emb_dtype):
        b = self.bufs.get(B)
        if b is None:
            torch, sm, dev = self.torch, self.sm, self.device
            d = sm.fc_d_model
            fifo_cap = sm.fifo_len + self.chunk_len          # FIFO holds <= fifo_len before the pop
            b = {"feats": torch.zeros((B, self.tc_steady, self.m.encoder._feat_in), device=dev),
                 "flen": torch.zeros((B,), dtype=torch.long, device=dev),
                 "spk": torch.zeros((B, sm.spkcache_len, d), device=dev, dtype=emb_dtype),
                 "fifo": torch.zeros((B, fifo_cap, d), device=dev, dtype=emb_dtype),
                 "spk_preds": torch.zeros((B, sm.spkcache_len, self.n_spk), device=dev),
                 "chunk_out": torch.zeros((B, self.chunk_len * self.upsample, self.n_spk), device=dev)}
            self.bufs[B] = b
        return b

    # ------------------------------------------------------------------ the step (pure tensor ops)
    def _core(self, feats_t, flen, spkcache, fifo):
        """pre-encode -> concat -> encoder -> head -> (downsample, length mask): NeMo's
        forward_streaming_step, sync branch, up to the state update. Returns (preds [B, T, S]
        masked at encoder resolution, high-res preds or None, chunk_embs)."""
        torch, m, sm = self.torch, self.m, self.sm
        with torch.autocast(device_type="cuda", enabled=False) if self.pre_fp32 else contextlib.nullcontext():
            chunk_embs, chunk_lens = m._call_pre_encode(feats_t, flen)
            if self.pre_fp32 and not self.post_fp32:
                chunk_embs = chunk_embs.to(self.dtype)          # state stays in the core dtype
        embs = sm.concat_embs([spkcache, fifo, chunk_embs], dim=1, device=chunk_embs.device)
        lens = spkcache.shape[1] + fifo.shape[1] + chunk_lens
        with self.enc_autocast:
            emb_seq, emb_len = m.frontend_encoder(processed_signal=embs, processed_signal_length=lens,
                                                  bypass_pre_encode=True)
        with torch.autocast(device_type="cuda", enabled=False) if self.head_fp32 else contextlib.nullcontext():
            if self.head_fp32:
                emb_seq = emb_seq.float()
            preds = m.forward_infer(emb_seq=emb_seq, emb_seq_length=emb_len)
            high_res = None
            if m.high_resolution:
                high_res = preds
                preds = sm.downsample_preds(high_res, m.upsample_factor)
        # == sm.apply_mask_to_preds (which uses a CPU scalar in torch.where; masked_fill is capture-safe)
        valid = torch.arange(preds.shape[1], device=preds.device).view(1, -1, 1) < emb_len.view(-1, 1, 1)
        return preds.masked_fill(~valid, 0.0), high_res, chunk_embs

    def _step(self, feats_t, flen, spkcache, fifo, spkcache_preds, lc, rc, compressed):
        """One streaming step: core + NeMo's sync cache/FIFO update. Returns
        (chunk_preds [B, chunk*up, S] high-res, spkcache', fifo', spkcache_preds', compressed')."""
        torch, m, sm = self.torch, self.m, self.sm
        S, F = spkcache.shape[1], fifo.shape[1]
        preds, high_res, chunk_embs = self._core(feats_t, flen, spkcache, fifo)
        chunk_len = chunk_embs.shape[1] - lc - rc
        # --- streaming_update (sync), use_learnable_sil_emb -> no silence-profile update
        fifo_preds = preds[:, S: S + F]
        chunk = chunk_embs[:, lc: chunk_len + lc]
        chunk_preds_lr = preds[:, S + F + lc: S + F + chunk_len + lc]
        fifo = torch.cat([fifo, chunk], dim=1)
        fifo_preds = torch.cat([fifo_preds, chunk_preds_lr], dim=1)
        if F + chunk_len > sm.fifo_len:
            pop = max(sm.spkcache_update_period, chunk_len - sm.fifo_len + F)
            pop = min(pop, F + chunk_len)
            pop_embs, pop_preds = fifo[:, :pop], fifo_preds[:, :pop]
            fifo = fifo[:, pop:]
            spkcache = torch.cat([spkcache, pop_embs], dim=1)
            if compressed:
                spkcache_preds = torch.cat([spkcache_preds, pop_preds], dim=1)
            else:
                spkcache_preds = torch.cat([preds[:, :S], pop_preds], dim=1)
            if spkcache.shape[1] > sm.spkcache_len:
                spkcache, spkcache_preds = self._compress(spkcache, spkcache_preds)
                compressed = True
        if m.high_resolution:
            start = (S + F + lc) * m.upsample_factor
            chunk_preds = high_res[:, start: start + chunk_len * m.upsample_factor]
        else:
            chunk_preds = chunk_preds_lr
        if self.out_ds > 1:
            chunk_preds = sm.downsample_preds(chunk_preds, self.out_ds)
        return chunk_preds.float(), spkcache, fifo, spkcache_preds, compressed

    def _compress(self, emb_seq, preds):
        """``SortformerModules._compress_spkcache(permute_spk=False)`` with device-side indices only
        (the original's boolean index_put and CPU index tensors cannot be captured); same values."""
        torch, sm = self.torch, self.sm
        batch_size, n_frames, n_spk = preds.shape
        dev = preds.device
        mean_sil_emb = sm.learnable_sil_emb.to(dtype=emb_seq.dtype, device=dev).unsqueeze(0).expand(batch_size, -1)
        per_spk = sm.spkcache_len // n_spk - sm.spkcache_sil_frames_per_spk
        strong = math.floor(per_spk * sm.strong_boost_rate)
        weak = math.floor(per_spk * sm.weak_boost_rate)
        min_pos = math.floor(per_spk * sm.min_pos_scores_rate)
        scores = sm._get_log_pred_scores(preds)
        # _disable_low_scores
        is_speech = preds > 0.5
        scores = torch.where(is_speech, scores, float("-inf"))
        is_pos = scores > 0
        replace = (~is_pos) * is_speech * (is_pos.sum(dim=1).unsqueeze(1) >= min_pos)
        scores = torch.where(replace, float("-inf"), scores)
        if sm.scores_boost_latest > 0:
            scores[:, sm.spkcache_len:, :] += sm.scores_boost_latest
        # _boost_topk_scores (strong, weak): distinct indices, so scatter_add == indexed subtract
        for n_boost, scale in ((strong, 2), (weak, 1)):
            _, top = torch.topk(scores, n_boost, dim=1, largest=True, sorted=False)
            scores = scores.scatter_add(1, top, torch.full_like(top, -scale * math.log(0.5), dtype=scores.dtype))
        if sm.spkcache_sil_frames_per_spk > 0:
            pad = torch.full((batch_size, sm.spkcache_sil_frames_per_spk, n_spk), float("inf"), device=dev)
            scores = torch.cat([scores, pad], dim=1)
        # _get_topk_indices
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
        # _gather_spkcache_and_preds
        emb_g = torch.gather(emb_seq, 1, top_s.unsqueeze(-1).expand(-1, -1, emb_seq.shape[2]))
        emb_g = torch.where(disabled.unsqueeze(-1), mean_sil_emb.unsqueeze(1).expand(-1, sm.spkcache_len, -1), emb_g)
        preds_g = torch.gather(preds, 1, top_s.unsqueeze(-1).expand(-1, -1, n_spk))
        preds_g = torch.where(disabled.unsqueeze(-1), 0.0, preds_g)
        return emb_g, preds_g

    # ------------------------------------------------------------------ graph per shape
    def _step_buffered(self, b, S, F, tc, lc, rc, compressed, full=True):
        """The step on the persistent buffers: read state prefixes, write the new prefixes back."""
        if self.hybrid:
            self.m.encoder._attn_full = full       # baked into this graph (part of its key)
        out = self._step(b["feats"][:, :tc], b["flen"], b["spk"][:, :S], b["fifo"][:, :F],
                         b["spk_preds"][:, :S], lc, rc, compressed)
        chunk_preds, spk, fifo, spk_preds, compressed2 = out
        self._writeback(b["spk"], spk)
        self._writeback(b["fifo"], fifo)
        if compressed2:
            self._writeback(b["spk_preds"], spk_preds)
        b["chunk_out"][:, : chunk_preds.shape[1]].copy_(chunk_preds)
        return spk.shape[1], fifo.shape[1], compressed2, chunk_preds.shape[1]

    @staticmethod
    def _writeback(buf, t):
        """buf[:, :n] = t unless t is still the unchanged prefix view of buf (a step with no pop)."""
        if t.data_ptr() != buf.data_ptr() or t.stride() != buf.stride():
            buf[:, : t.shape[1]].copy_(t)

    def _run_step(self, b, B, S, F, tc, lc, rc, compressed, capture, full=True):
        key = (B, S, F, tc, lc, rc, compressed, full)
        g = self.graphs.get(key)
        if g is None:
            if not capture or not self.use_graphs or tc != self.tc_steady:
                self.eager_steps += 1
                with self.autocast:
                    return self._step_buffered(b, S, F, tc, lc, rc, compressed, full)
            g = self._capture(key, b)
        g["graph"].replay()
        self.replays += 1
        return g["out"]

    def _capture(self, key, b):
        torch = self.torch
        B, S, F, tc, lc, rc, compressed, full = key
        # Dry run for kernel selection / lazy compiles (FlexAttention), on a scratch copy of the
        # state so the buffers are unchanged when the capture replays the same step.
        saved = {k: b[k].clone() for k in ("spk", "fifo", "spk_preds")}
        with self.autocast:
            self._step_buffered(b, S, F, tc, lc, rc, compressed, full)
        for k, v in saved.items():
            b[k].copy_(v)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool, stream=self.stream, capture_error_mode="thread_local"), \
                self.autocast:
            out = self._step_buffered(b, S, F, tc, lc, rc, compressed, full)
        for k, v in saved.items():
            b[k].copy_(v)
        g = {"graph": graph, "out": out}
        self.graphs[key] = g
        return g

    # ------------------------------------------------------------------ whole-file inference
    def run(self, audios, *, capture=True, sample_rate=16000):
        """audios: list of 1-D float32 (numpy or CPU/pinned torch) waveforms at 16 kHz."""
        torch, m = self.torch, self.m
        tm = _Timers()
        n = len(audios)
        B = next((s for s in self.sizes if s >= n), None)
        if B is None or not self.use_graphs:
            B = n
        with self.lock, torch.inference_mode(), torch.cuda.stream(self.stream):
            t_all = time.perf_counter()
            with tm("h2d"):
                lens = [len(a) for a in audios]
                sig = torch.zeros((B, max(lens)), device=self.device)
                for i, a in enumerate(audios):
                    t = a if isinstance(a, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
                    sig[i, : lens[i]].copy_(t, non_blocking=True)
                for i in range(n, B):                       # pad rows: duplicate a real row, result dropped
                    sig[i].copy_(sig[i % n])
                sig_len = torch.tensor([lens[i % n] for i in range(B)], device=self.device)
            with tm("mel"):
                feats, feat_len = m.process_signal(audio_signal=sig, audio_signal_length=sig_len)
                feat_len_max = int(feat_len.max())
                feats = feats[:, :, :feat_len_max]
            with tm("steps"):
                preds_hr = self._stream(feats, feat_len, B, capture, n)
            with tm("tail"):
                out = self._finish(preds_hr, feats.shape[2], feat_len)
                f = m.output_subsampling_factor
                fl = feat_len.tolist()
                out_lens = [min(out.shape[1], -(-fl[i] // f)) for i in range(n)]
                # Numerical health of a low-precision core: non-finite predictions or state, and the
                # probability range (a NaN on a few frames clears a set-level DER gate; this does not).
                b = self.bufs[B]
                health = torch.stack([(~torch.isfinite(out[:n])).sum(), (~torch.isfinite(b["spk"])).sum()
                                      + (~torch.isfinite(b["fifo"])).sum(), out[:n].max().float(), out[:n].min().float()])
            torch.cuda.synchronize(self.device)
            tm.t["fwd"] = time.perf_counter() - t_all
            nf_p, nf_s, p_max, p_min = health.tolist()
        tm.t.update(B=B, nonfinite_preds=int(nf_p), nonfinite_state=int(nf_s), pred_max=round(p_max, 4), pred_min=round(p_min, 4))
        return out[:n], out_lens, tm.t

    def to_cpu(self, out, out_lens):
        """Per-file [T_i, S] CPU tensors (NeMo's `include_tensor_outputs` shape, trimmed)."""
        with self.torch.cuda.stream(self.stream):
            return [out[i, : out_lens[i]].cpu() for i in range(len(out_lens))]

    def segments(self, out, out_lens, pp_params):
        """Speaker segment lines for every file of a batch: NeMo's post-processing, but the
        binarization runs once on the GPU for the whole [n, T, S] tensor (see `binarize_batch`);
        `filtering` is the identity for the default parameters, so the output is the same strings.
        Non-default parameters (padding, minimum durations, hysteresis) fall back to NeMo per file."""
        from nemo.collections.asr.parts.utils.speaker_utils import generate_diarization_output_lines
        from nemo.collections.asr.parts.utils.vad_utils import predlist_to_timestamps
        m = self.m
        pp = {k: float(pp_params.get(k, d)) for k, d in (("onset", 0.5), ("offset", 0.5), ("pad_onset", 0.0),
                                                        ("pad_offset", 0.0), ("min_duration_on", 0.0),
                                                        ("min_duration_off", 0.0))}
        frame_s = 0.01 * m.output_subsampling_factor
        if pp["onset"] >= pp["offset"] and not any(pp[k] > 0 for k in ("pad_onset", "pad_offset", "min_duration_on",
                                                                          "min_duration_off")):
            with self.torch.cuda.stream(self.stream):
                per_file = binarize_batch(self.torch, out, out_lens, pp["onset"], pp["offset"], frame_s)
        else:
            per_file = [predlist_to_timestamps(batch_preds_list=[p.unsqueeze(0)], audio_rttm_map_dict={"x": {"offset": 0.0}},
                                               cfg_vad_params=pp_params, unit_10ms_frame_count=m.output_subsampling_factor,
                                               bypass_postprocessing=False)[0] for p in self.to_cpu(out, out_lens)]
        return [generate_diarization_output_lines(speaker_timestamps=ts, model_spk_num=len(ts)) for ts in per_file]

    def _stream(self, feats, feat_len, B, capture, n_real=None):
        """``forward_streaming`` (sync) over the batch, chunks from NeMo's streaming_feat_loader."""
        torch, sm = self.torch, self.sm
        feat_len_max = feats.shape[2]
        fl = feat_len.tolist()[: n_real or B]          # host copy for the hybrid attention decision
        stt = 0
        n_out = math.ceil(feat_len_max / self.m.output_subsampling_factor)
        chunk_out_frames = self.chunk_len * self.upsample // max(1, self.out_ds)
        total = torch.zeros((B, n_out + chunk_out_frames, self.n_spk), device=self.device)
        offset = torch.zeros((B,), dtype=torch.long, device=self.device)
        loader = sm.streaming_feat_loader(feat_seq=feats, feat_seq_length=feat_len, feat_seq_offset=offset)
        b = self._bufs(B, self.state_dtype)
        for k in ("spk", "fifo", "spk_preds"):        # fresh session
            b[k].zero_()
        S = F = 0
        compressed = False
        pos = 0
        for _, chunk_t, flen, left_off, right_off in loader:
            tc = chunk_t.shape[1]
            lc, rc = round(left_off / SUB), math.ceil(right_off / SUB)
            # unmasked attention is exact unless a still-active row is in its final partial chunk
            # (rows whose file has ended are discarded and row-independent)
            full = (not self.hybrid) or all(f <= stt or f - stt + left_off >= tc for f in fl)
            b["feats"][:, :tc].copy_(chunk_t)
            b["flen"].copy_(flen)
            S, F, compressed, n_chunk = self._run_step(b, B, S, F, tc, lc, rc, compressed, capture, full)
            stt += tc - left_off - right_off
            total[:, pos: pos + n_chunk].copy_(b["chunk_out"][:, :n_chunk])
            pos += n_chunk
        return total[:, :n_out]

    def _finish(self, preds, feat_len_max, feat_len):
        """Tail of ``SortformerEncLabelModel.forward``: length mask (+ optional downsample)."""
        torch, m, sm = self.torch, self.m, self.sm
        f = m.output_subsampling_factor
        max_out = min(preds.shape[1], math.ceil(feat_len_max / f))
        out_len = torch.div(feat_len + f - 1, f, rounding_mode="floor").clamp(max=max_out)
        preds = preds[:, :max_out]
        mask = sm.length_to_mask(out_len, max_out)
        return preds * mask.unsqueeze(-1)

    # ------------------------------------------------------------------ warm-up
    def warm(self, seconds=100.0, sizes=None, sample_rate=16000):
        """Capture every steady shape for each batch size by streaming synthetic audio long enough
        to walk the cache/FIFO cycle (``low``: 129 shapes in ~85 s of audio)."""
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        before = len(self.graphs)
        for B in sizes or self.sizes:
            wav = (rng.standard_normal(int(seconds * sample_rate)) * 0.01).astype(np.float32)
            self.run([wav] * B)
        self.torch.cuda.synchronize(self.device)
        logger.info("%s: %d graphs captured (%d total, sizes %s) in %.1fs", self.name,
                    len(self.graphs) - before, len(self.graphs), list(sizes or self.sizes), time.perf_counter() - t0)
        return len(self.graphs)

    def stats(self):
        return {"graphs": len(self.graphs), "replays": self.replays, "eager_steps": self.eager_steps,
                "sizes": list(self.sizes), "dtype": str(self.dtype).replace("torch.", ""), "attn_mode": self.attn_mode, "post_fp32": self.post_fp32,
                "pre_fp32": self.pre_fp32, "head_fp32": self.head_fp32}


def binarize_batch(torch, preds, lens, onset, offset, frame_s, precision=2):
    """`vad_utils.binarization_vectorized` (onset >= offset branch) for every (file, speaker) row of a
    [n, T, S] device tensor at once, then NeMo's `+ offset(0) -> tolist -> round(…, precision)`.
    Frames past a file's length are zero, exactly as in NeMo's padded batch (an "off" event), so the
    per-row result equals the per-file call. Returns [[[start, end], ...] per speaker] per file."""
    n, T, S = preds.shape
    seq = preds.permute(0, 2, 1).reshape(n * S, T)                     # rows = (file, speaker)
    positions = torch.arange(1, T + 1, device=seq.device)
    force_on = seq > onset
    has_event = force_on | (seq < offset)
    event_positions = torch.where(has_event, positions.unsqueeze(0), 0)
    last_event = torch.cummax(event_positions, dim=1).values
    event_states = torch.cat([torch.zeros((n * S, 1), dtype=torch.bool, device=seq.device), force_on], dim=1)
    above = torch.gather(event_states, 1, last_event)
    padded = torch.nn.functional.pad(above.float(), (1, 1), value=0.0)
    diff = padded[:, 1:] - padded[:, :-1]
    starts = torch.nonzero(diff > 0.5)                                 # (row, frame) in row-major order
    ends = torch.nonzero(diff < -0.5)
    start_times = torch.clamp(starts[:, 1].float() * frame_s, min=0.0)
    end_times = ends[:, 1].float() * frame_s
    rows = starts[:, 0].tolist()
    st, en = start_times.tolist(), end_times.tolist()
    out = [[[] for _ in range(S)] for _ in range(n)]
    for r, a, b in zip(rows, st, en):
        if b > a:
            out[r // S][r % S].append([round(a, precision), round(b, precision)])
    return out


def fp32_layers(torch, encoder, indices, core_dtype):
    """Run the given encoder blocks in fp32 inside a low-precision core (bisect island): their weights
    stay fp32 and their forward runs with autocast off; the residual stream is fp32 under autocast
    already, so only the block's GEMMs/attention change precision."""
    import types

    for i in indices:
        layer = encoder.layers[i]
        layer.float()
        orig = layer.forward

        def forward(self, x, block_mask=None, pos_emb=None, _orig=orig):
            with torch.autocast(device_type="cuda", enabled=False):
                return _orig(x.float(), block_mask=block_mask, pos_emb=pos_emb)

        layer.forward = types.MethodType(forward, layer)
    logger.info("encoder layers %s in fp32", list(indices))


def patch_rope_fp32(torch, encoder):
    """Rotary embedding in fp32 inside a low-precision core: rebuild the cos/sin tables in fp32 (they were
    cast with the weights) and rotate upcast q/k, casting back after (bisect island; negligible cost)."""
    import types

    rope = encoder.layers[0].attn.rope
    if rope is None:
        return
    n = rope.cos.size(0) if hasattr(rope, "cos") else encoder.max_audio_length
    rope.create_pe(torch.arange(0, n, dtype=torch.float32, device=next(encoder.parameters()).device), torch.float32)
    orig = rope.forward

    def forward(self, q, k, _orig=orig):
        q32, k32 = _orig(q.float(), k.float())
        return q32.to(q.dtype), k32.to(k.dtype)

    rope.forward = types.MethodType(forward, rope)
    logger.info("RoPE tables and rotation in fp32")


def patch_gelu_fp32(torch, encoder):
    """FFN activation in fp32 inside a low-precision core (bisect island): GEMMs stay bf16/fp16."""
    import types

    def forward(self, x):
        h = self.net[0](x)
        h = torch.nn.functional.gelu(h.float()).to(h.dtype)
        return self.net[3](self.net[2](h))

    for layer in encoder.layers:
        layer.ffn.forward = types.MethodType(forward, layer.ffn)
    logger.info("FFN GELU in fp32 on %d layers", len(encoder.layers))


def patch_attention(torch, encoder, mode, core_dtype):
    """Override every attention layer's FlexAttention call.

    ``fp32``: q/k/v upcast, kernel in fp32 (the bf16 drift root-cause test; -17 % speed). ``direct``:
    q/k/v cast to the core dtype and the compiled kernel called with autocast disabled — bypasses the
    HOP's autocast wrapper, which under fp16 autocast routed to the eager math path (a CPU tensor
    mid-step, not capturable). The override is the module's own ``forward``
    (``transformer_encoder.MultiHeadAttention.forward``) with the kernel call changed.
    """
    import types

    from nemo.collections.asr.modules import transformer_encoder_utils as _tu

    kernel_dtype = torch.float32 if mode == "fp32" else core_dtype

    flex_opts = {"flex64": {"BLOCK_M": 64, "BLOCK_N": 64}, "flex32": {"BLOCK_M": 32, "BLOCK_N": 32}}.get(mode)

    def forward_sdpa(self, x, block_mask=None, pos_emb=None):
        """Dense `F.scaled_dot_product_attention` instead of FlexAttention. ``sdpa``: the same padding mask
        (keys past a row's valid length; the encoder's `_build_mask_mod` records the lengths — this routes
        SDPA off the flash kernel). ``sdpa_nomask``: no mask — exact in NeMo's sync state, where every row
        of a batch shares the cache/FIFO fill, except for a file's final partial chunk (finished rows are
        discarded anyway). ``sdpa_cudnn``: the mask, cuDNN backend only."""
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim
        qkv = self.w_qkv(x).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_norm:
            q = self.q_norm(q).to(v.dtype)
            k = self.k_norm(k).to(v.dtype)
        if self._uses_rope:
            q, k = self.rope(q, k)
        score_mod = None
        if self._uses_rel_pos:
            score_mod, q = self._build_rel_pos_score_mod(q, pos_emb)
        if mode == "sdpa_nomask" or (mode == "hybrid" and encoder._attn_full):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        elif mode == "hybrid":
            with torch.autocast(device_type="cuda", enabled=False):
                attn_fn = _tu._get_flex_attention(q)
                out = attn_fn(q, k, v, block_mask=block_mask, score_mod=score_mod)
        else:
            lengths = encoder._sdpa_lengths
            key_ok = torch.arange(T, device=x.device).view(1, 1, 1, T) < lengths.view(B, 1, 1, 1)
            if mode == "sdpa_cudnn":
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
                    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=key_ok)
            else:
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=key_ok)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)

    if mode == "hybrid":
        encoder._attn_full = True
    if mode in ("sdpa", "sdpa_cudnn"):
        orig_bmm = encoder._build_mask_mod

        def build_mask_mod(self, length, _orig=orig_bmm):
            self._sdpa_lengths = length
            return _orig(length)

        encoder._build_mask_mod = types.MethodType(build_mask_mod, encoder)
        encoder._sdpa_lengths = None

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
        score_mod = None
        if self._uses_rel_pos:
            score_mod, q = self._build_rel_pos_score_mod(q, pos_emb)
        out_dtype = v.dtype
        with torch.autocast(device_type="cuda", enabled=False):
            attn_fn = _tu._get_flex_attention(q)
            kw = {"kernel_options": flex_opts} if flex_opts else {}
            out = attn_fn(q.to(kernel_dtype), k.to(kernel_dtype), v.to(kernel_dtype), block_mask=block_mask,
                          score_mod=score_mod, **kw)
        out = out.to(out_dtype).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)

    n = 0
    for layer in encoder.layers:
        layer.attn.forward = types.MethodType(forward_sdpa if mode.startswith("sdpa") or mode == "hybrid" else forward,
                                              layer.attn)
        n += 1
    logger.info("attention kernel in %s on %d layers (mode %s)", kernel_dtype, n, mode)


def patch_linears_fp8(torch, encoder):
    """Encoder linears in FP8 (e4m3): weights quantised once with a per-tensor absmax scale, activations
    quantised per call with a per-tensor absmax scale, `torch._scaled_mm` with bf16 output. Attention and
    head untouched. A speed-ceiling measurement first; quality only if the step time earns it."""
    import types

    f8 = torch.float8_e4m3fn
    fmax = torch.finfo(f8).max

    def forward(self, x):
        shp = x.shape
        a = x.reshape(-1, shp[-1]).to(torch.bfloat16)
        sa = (a.abs().amax().float() / fmax).clamp(min=1e-12)
        a8 = (a / sa).to(f8)
        out = torch._scaled_mm(a8, self.w8.t(), scale_a=sa, scale_b=self.s8, bias=None, out_dtype=torch.bfloat16)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out.reshape(*shp[:-1], out.shape[-1])

    n = 0
    for layer in encoder.layers:
        for lin in (layer.attn.w_qkv, layer.attn.out_proj, layer.ffn.net[0], layer.ffn.net[3]):
            w = lin.weight.detach().float()
            s8 = (w.abs().amax() / fmax).clamp(min=1e-12)
            lin.w8 = (w / s8).to(f8).contiguous()
            lin.s8 = s8.float()
            lin.forward = types.MethodType(forward, lin)
            n += 1
    logger.info("%d encoder linears in FP8 e4m3 (per-tensor scales)", n)
