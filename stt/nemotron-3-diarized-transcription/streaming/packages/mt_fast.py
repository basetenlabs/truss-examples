"""Step-time levers for the multitalker streaming truss.

Every lever is an env flag (see ``Flags``) and, when ``MT_ALLOW_CONTROL=1``, can also be flipped at
runtime through the WebSocket handshake ``{"control": {...}}`` while no session is live, so one
deployment serves the whole A/B matrix. The levers do not change the algorithm; they change how
the same NeMo calls are dispatched:

* mel       -- ``MelChunker``: log-mel computed ONCE per 112-frame step over exactly the frames the
               chunk needs (9 pre-encode cache frames + 112, plus 3 frames of recomputed left
               context for STFT exactness) from a bounded raw ring, then NeMo's per-chunk
               ``normalize_batch`` -- the same tensor ``CacheAwareStreamingAudioBuffer`` yields.
* asr_dtype -- ASR encoder + speaker kernels in bf16 WEIGHTS (or fp32) instead of fp32 weights under
               bf16 autocast: removes ~900 autocast weight casts per step. The RNNT decoder always
               runs under bf16 autocast exactly as before, so decoder numerics are unchanged.
* enc_graphs-- NeMo's ``CudaGraphsStreamingEncoderStep`` for the cache-aware FastConformer step. It
               requires no autocast and no ``bypass_pre_encode``, so the pre-encoder runs inside the
               graphed step at batch k (identical math) and the speaker-kernel hook reads its
               targets from static per-k buffers we ``copy_`` into before every replay.
* diar_graphs -- (default) SYNC-mode diarizer with the eager core (pre-encode -> concat -> 31-layer
               encoder -> head) captured in a CUDA graph per distinct sequence length (``DiarGraphs``):
               bit-identical to NeMo's sync path, one replay per step. ~150 lengths, captured at load.
* pad_rows  -- (default 8) pad the ASR step to a constant speaker-row count: cuBLAS rounding depends on
               the GEMM M-dimension, so a changing k re-rolls every speaker's numerics; constant rows
               make the transcript deterministic, leave one encoder graph and a fixed decoder state.
* diar_async / diar_compile -- A/B only (off): Sortformer async fixed-shape state + ``torch.compile``
               of the core. 2.5x faster diarizer but NOT output-preserving (bf16 rounding of the
               padded/fused kernels flips speaker gates; cpWER +1..+5 pt), see FASTSTEP.md s.5.
* no_deepcopy -- ``get_active_speakers_info`` copies hypotheses shallowly: the decoder returns a
               freshly cloned state every call (RLL:792-800), so the deepcopy only protected
               ``merge_``'s in-place update of an object that is written back anyway.
* inc_detok -- (default) ``mt_incremental.IncrementalDetok`` replaces ``decode_hypothesis``: the
               committed prefix of every speaker's transcript is cached on the hypothesis and only
               the new tail is detokenised, so the step stops growing with session length.
               Byte-identical text (``MT_INC_DETOK_VERIFY=1`` checks it against NeMo's full decode).
"""

import logging
import math
import os
import time
from contextlib import nullcontext

import numpy as np

logger = logging.getLogger(__name__)

SR = 16000
HOP = 160      # 10 ms mel hop (samples)
HALF = 256     # n_fft/2: reach of one centered STFT frame
LPAD = 3       # mel frames of real left context recomputed per chunk (>= 2 makes frames exact)


def _env_bool(name, default):
    return os.environ.get(name, default).strip().lower() not in ("0", "", "false", "off", "no")


class Flags:
    """Lever switches; env defaults, mutable at runtime through ``FastPath.apply_control``."""

    def __init__(self):
        self.mel = _env_bool("MT_MEL_PER_STEP", "1")
        self.asr_dtype = os.environ.get("MT_ASR_DTYPE", "bf16").strip().lower()   # bf16 | fp32 | autocast
        self.enc_graphs = _env_bool("MT_ENC_CUDA_GRAPHS", "1")
        self.diar_async = _env_bool("MT_DIAR_ASYNC", "0")
        self.diar_compile = os.environ.get("MT_DIAR_COMPILE", "0").strip().lower()  # 0|default|reduce-overhead
        # SYNC-mode diarizer with the eager core (pre-encode -> concat -> encoder -> head) captured in a
        # CUDA graph per distinct sequence length: bit-identical to eager, one launch per step.
        self.diar_graphs = _env_bool("MT_DIAR_GRAPHS", "1")
        self.no_deepcopy = _env_bool("MT_NO_DEEPCOPY", "1")
        # Pad the active-speaker rows of every ASR step to a constant count (0 = off). cuBLAS picks
        # different (split-K) kernels per GEMM M-dimension, so a changing speaker count k changes the
        # rounding of every speaker's stream (results_mt_td/B_GT_1_ROOTCAUSE.md); constant rows make
        # the step deterministic, give ONE encoder graph and a fixed decoder state.
        self.pad_rows = int(os.environ.get("MT_PAD_ROWS", "8"))
        # TF32 for the ASR encoder's fp32 GEMMs (encoder-scoped: the mel filterbank stays exact).
        self.tf32 = _env_bool("MT_TF32", "0")
        self.mel_verify = _env_bool("MT_MEL_VERIFY", "0")
        # Diagnostics: run a shadow SYNC-mode diarizer state per session next to the async one and
        # log the per-step divergence of the chunk predictions.
        self.diar_verify = _env_bool("MT_DIAR_VERIFY", "0")
        # Warm the ASR path at k=max..1 at load (sizes the decoder state once; captures encoder graphs).
        self.warm_asr = _env_bool("MT_WARM_ASR", "1")
        # Incremental detokenisation of the per-speaker hypotheses (flat step over hour-long calls);
        # VERIFY runs NeMo's full decode alongside and logs any mismatch (diagnostics only).
        self.inc_detok = _env_bool("MT_INC_DETOK", "1")
        self.inc_detok_verify = _env_bool("MT_INC_DETOK_VERIFY", "0")
        if self.asr_dtype not in ("bf16", "fp32", "autocast"):
            raise ValueError(f"MT_ASR_DTYPE must be bf16|fp32|autocast, got {self.asr_dtype}")

    def as_dict(self):
        return {k: v for k, v in vars(self).items()}


class MelChunker:
    """Per-step mel from a bounded raw ring: one preprocessor call per 112-frame chunk.

    Reproduces ``CacheAwareStreamingAudioBuffer.__iter__`` with ``online_normalization`` and
    ``pad_and_drop_preencoded``: chunk = [9 pre-encode cache frames | 112 new frames] (zeros for
    the cache on step 0), per-feature normalised over the 121 frames, length 121. Raw mel frames
    are frame-local (``normalize: None``, centred STFT, no dither), so a frame computed on a
    trailing window with LPAD frames of left context equals the whole-file frame; frames whose
    window runs past the audio end are held back until more audio arrives (or the flush).
    """

    def __init__(self, torch, preprocessor, normalize_type, device, chunk_frames, cache_frames):
        from nemo.collections.asr.parts.preprocessing.features import normalize_batch

        self.torch = torch
        self._normalize_batch = normalize_batch
        self.pre = preprocessor
        self.norm = normalize_type
        self.device = device
        self.chunk = chunk_frames
        self.cache = cache_frames
        self.raw = np.zeros(0, dtype=np.float32)
        self.raw_off = 0      # absolute sample index of raw[0]; consumed audio is dropped
        self.idx = 0          # mel frames consumed so far (== NeMo buffer_idx)
        self.lengths = torch.full((1,), chunk_frames + cache_frames, dtype=torch.int64, device=device)
        self._cpu_len = torch.tensor([chunk_frames + cache_frames])

    def add(self, samples):
        self.raw = np.concatenate([self.raw, samples])

    @property
    def total(self):
        return self.raw_off + len(self.raw)

    def frames_complete(self, final):
        L = self.total
        return L // HOP + 1 if final else max(0, (L - HALF) // HOP + 1)

    def has_chunk(self, final):
        return self.frames_complete(final) >= self.idx + self.chunk

    def pad_for_flush(self):
        """Zero-pad the tail so the remaining frames form whole chunks (same rule as the buffer path)."""
        pending = self.total // HOP + 1 - self.idx
        k = (-pending) % self.chunk
        if k:
            self.raw = np.concatenate([self.raw, np.zeros(k * HOP, dtype=np.float32)])

    def next_window(self, final):
        """The raw window the next chunk's mel needs, without computing it (MT_XMEL: the stepper
        computes one batched mel per tick). Returns ``(samples, geometry)`` with geometry =
        (first_chunk, disc, frames, n_samples); rows of equal geometry share one preprocessor call.
        Consumes the chunk like ``next_chunk``."""
        b = self.idx + self.chunk
        first = self.idx == 0
        a = 0 if first else self.idx - self.cache
        disc = min(a, LPAD)
        start = (a - disc) * HOP
        end = (b - 1) * HOP + HALF
        L = self.total
        if end > L:
            if not final:
                raise RuntimeError(f"mel chunk [{a},{b}) needs {end} samples, have {L}")
            end = L
        seg = np.ascontiguousarray(self.raw[start - self.raw_off: end - self.raw_off])
        self.idx = b
        self._trim()
        return seg, (first, disc, b - a, len(seg))

    def next_chunk(self, final):
        torch = self.torch
        b = self.idx + self.chunk
        if self.idx == 0:
            feats = self._feats(0, b, final)
            feats = torch.cat([feats.new_zeros((1, feats.shape[1], self.cache)), feats], dim=-1)
        else:
            feats = self._feats(self.idx - self.cache, b, final)
        with torch.inference_mode():
            chunk, _, _ = self._normalize_batch(x=feats.contiguous(), seq_len=self._cpu_len, normalize_type=self.norm)
        self.idx = b
        self._trim()
        return chunk, self.lengths

    def _feats(self, a, b, final):
        """Raw log-mel for global frames [a, b): trailing-window compute with LPAD frames of context."""
        torch = self.torch
        disc = min(a, LPAD)
        start = (a - disc) * HOP
        end = (b - 1) * HOP + HALF
        L = self.total
        if end > L:
            if not final:
                raise RuntimeError(f"mel chunk [{a},{b}) needs {end} samples, have {L}")
            end = L   # flush: the tail frames see the reflect padding of the audio end, as whole-file does
        seg = self.raw[start - self.raw_off: end - self.raw_off]
        sig = torch.as_tensor(seg, device=self.device)[None, :]
        ln = torch.tensor([sig.shape[1]], device=self.device)
        with torch.inference_mode():
            feats, _ = self.pre(input_signal=sig, length=ln)   # [1, F, T]
        return feats[:, :, disc: disc + (b - a)]

    def _trim(self):
        keep_from = max(0, (self.idx - self.cache - LPAD) * HOP)
        if keep_from > self.raw_off:
            self.raw = self.raw[keep_from - self.raw_off:]
            self.raw_off = keep_from


class DiarGraphs:
    """CUDA-graph replay of the diarizer core for SYNC streaming mode, keyed by input shape.

    Same idea as NeMo's ``CudaGraphsStreamingEncoderStep``: the eager core is run once for a new
    shape (kernel selection warm-up), captured on its next occurrence into a private graph with
    static input buffers, then replayed. Replay runs exactly the eager kernels, so the outputs are
    bit-identical to the shipped sync path (unlike padding to a fixed shape or torch.compile, which
    change bf16 rounding and, through the speaker gate, the transcript). Sync mode walks ~55
    distinct lengths (cache 0->264, FIFO cycling in 14-frame steps), each graph pool is a few MB.
    """

    def __init__(self, torch, diar, warmup_steps=0, max_graphs=256):
        # warmup_steps=0: capture a length the first time it is seen (the dry run inside _capture is
        # the kernel warm-up), so the 70 s load-time session captures every length of the schedule
        # instead of leaving the ramp lengths to be captured mid-way through the first live session.
        self.torch, self.diar = torch, diar
        self.warmup_steps, self.max_graphs = warmup_steps, max_graphs
        self.graphs, self.counts = {}, {}
        self.disabled = False
        self.replays = 0

    def _ctx(self):
        # bf16 weights: autocast only casts activations; the weight-cast cache must be off for capture.
        return self.torch.amp.autocast("cuda", dtype=self.torch.bfloat16, cache_enabled=False)

    def eager(self, embs, lens):
        with self._ctx():
            emb_seq, emb_len = self.diar.frontend_encoder(processed_signal=embs, processed_signal_length=lens,
                                                          bypass_pre_encode=True)
            preds = self.diar.forward_infer(emb_seq=emb_seq, emb_seq_length=emb_len)
        return preds, emb_len

    def run(self, embs, lens):
        torch = self.torch
        key = (tuple(embs.shape), str(embs.dtype))
        g = self.graphs.get(key)
        if g is None:
            if self.disabled or torch.cuda.is_current_stream_capturing():
                return self.eager(embs, lens)
            self.counts[key] = self.counts.get(key, 0) + 1
            if self.counts[key] <= self.warmup_steps or len(self.graphs) >= self.max_graphs:
                if len(self.graphs) >= self.max_graphs and self.counts[key] == self.warmup_steps + 1:
                    logger.warning("diarizer graphs: cap %d reached, shape %s runs eager", self.max_graphs, key[0])
                return self.eager(embs, lens)
            try:
                g = self._capture(key, embs, lens)
            except Exception:  # noqa: BLE001 — eager is correct, just slower
                logger.exception("diarizer graph capture failed for %s; diarizer runs eager from now on", key[0])
                self.disabled = True
                return self.eager(embs, lens)
        with torch.inference_mode():
            g["embs"].copy_(embs)
            g["lens"].copy_(lens)
            g["graph"].replay()
            self.replays += 1
            return g["preds"].clone(), g["plen"].clone()

    def _capture(self, key, embs, lens):
        torch = self.torch
        device = embs.device
        with torch.inference_mode():
            static_embs, static_lens = embs.clone(), lens.clone()
            preds, plen = self.eager(static_embs, static_lens)      # dry run: shapes + kernel warm-up
            out_preds, out_len = torch.empty_like(preds), torch.empty_like(plen)
            del preds, plen
            torch.cuda.synchronize(device)
            s = torch.cuda.Stream(device)
            s.wait_stream(torch.cuda.default_stream(device))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(s), torch.cuda.graph(graph, stream=s, capture_error_mode="thread_local"):
                p, pl = self.eager(static_embs, static_lens)
                out_preds.copy_(p)
                out_len.copy_(pl)
        g = {"graph": graph, "embs": static_embs, "lens": static_lens, "preds": out_preds, "plen": out_len}
        self.graphs[key] = g
        logger.info("diarizer graph captured for L=%d (%d graphs)", key[0][1], len(self.graphs))
        return g


class FastPath:
    """Installs the levers on the shared (ASR, diarizer) pair and owns their runtime state."""

    def __init__(self, torch, asr, diar, cfg, flags: Flags, lock, max_speakers: int):
        from nemo.collections.asr.parts.utils import multispk_transcribe_utils as U

        self.torch, self.asr, self.diar, self.cfg, self.flags, self.lock = torch, asr, diar, cfg, flags, lock
        self.max_speakers = max_speakers
        self.U = U
        self.device = next(asr.parameters()).device
        self.scfg = asr.encoder.streaming_cfg
        self._enc_dtype = torch.float32
        self._spk_static = {}          # k -> (spk_targets, bg_spk_targets) static buffers
        self._tgt_full = None          # the last full (unpadded) speaker targets, for slabbed calls
        self._diar_core = None         # compiled diarizer core or None
        self._diar_compile_mode = "0"
        self._graphs_after_warmup = None
        self.warm_diar_fn = None       # set by the model: drives warm sessions (takes the step lock)
        self._shadows = {}             # MT_DIAR_VERIFY: id(async state) -> shadow sync state
        self._orig = {}
        self._install()

    # ------------------------------------------------------------------ install / patches
    def _install(self):
        torch, asr, diar, U = self.torch, self.asr, self.diar, self.U
        fast = self
        o = self._orig

        o["set_speaker_targets"] = asr.set_speaker_targets
        asr.set_speaker_targets = self._set_speaker_targets

        o["enc_step"] = asr.encoder.cache_aware_stream_step
        asr.encoder.cache_aware_stream_step = self._enc_step

        o["init_cache"] = asr.encoder.get_initial_cache_state
        asr.encoder.get_initial_cache_state = self._init_cache

        o["dec"] = asr.decoding.rnnt_decoder_predictions_tensor
        asr.decoding.rnnt_decoder_predictions_tensor = self._dec

        o["css"] = asr.conformer_stream_step
        asr.conformer_stream_step = self._conformer_stream_step

        # Incremental detokenisation: decode_hypothesis is looked up on the decoding instance by
        # rnnt_decoder_predictions_tensor, so an instance attribute overrides it (no NeMo edit).
        from mt_incremental import IncrementalDetok
        o["decode_hypothesis"] = asr.decoding.decode_hypothesis
        self.inc = IncrementalDetok(asr.decoding, asr.tokenizer, verify=self.flags.inc_detok_verify)

        def decode_hypothesis(hyps):
            if fast.flags.inc_detok:
                fast.inc.verify = fast.flags.inc_detok_verify
                return fast.inc.decode_hypothesis(hyps)
            return o["decode_hypothesis"](hyps)

        asr.decoding.decode_hypothesis = decode_hypothesis

        # Driver-level patches are class-level: SpeakerTaggedASR/InstanceManager are per session.
        o["fwd_pre"] = U.SpeakerTaggedASR.forward_pre_encoded

        def forward_pre_encoded(streamer, audio_signal, length, drop_extra_pre_encoded=0):
            if fast.no_bypass:
                return audio_signal, length        # pre-encoder runs inside the (graphed) encoder step
            with fast._enc_ctx():
                if fast.flags.asr_dtype == "bf16":
                    audio_signal = audio_signal.to(torch.bfloat16)
                return o["fwd_pre"](streamer, audio_signal, length, drop_extra_pre_encoded)

        U.SpeakerTaggedASR.forward_pre_encoded = forward_pre_encoded

        o["gasi"] = U.MultiTalkerInstanceManager.get_active_speakers_info

        def get_active_speakers_info(im, active_speakers, chunk_audio, chunk_lengths):
            if not fast.flags.no_deepcopy:
                return o["gasi"](im, active_speakers, chunk_audio, chunk_lengths)
            return fast._get_active_speakers_info(im, active_speakers, chunk_audio, chunk_lengths)

        U.MultiTalkerInstanceManager.get_active_speakers_info = get_active_speakers_info

        o["diar_step"] = diar.forward_streaming_step
        diar.forward_streaming_step = self._diar_step
        sm = diar.sortformer_modules
        o["init_state"] = sm.init_streaming_state
        sm.init_streaming_state = self._init_streaming_state

        # RNNT label-looping decoder graph mode. NeMo's default "full_graph" (conditional nodes via
        # cuda-python) intermittently hits cudaErrorIllegalAddress on its first replay after the encoder
        # graph is captured (every load at 64 rows; some pods at 32 -- allocation-dependent, not shape-
        # bound). "no_while_loops" replays the same kernels as partial graphs under a host loop.
        mode = os.environ.get("MT_DEC_GRAPHS_MODE", "").strip().lower()
        if mode:
            try:
                asr.decoding.decoding.decoding_computer.force_cuda_graphs_mode(mode)
                logger.info("RNNT decoder cuda_graphs_mode forced to %s", mode)
            except Exception:  # noqa: BLE001 - leave NeMo's default
                logger.exception("could not force decoder cuda_graphs_mode=%s", mode)

        self.set_asr_dtype(self.flags.asr_dtype, warm=False)
        self.set_enc_graphs(self.flags.enc_graphs, warm=False)
        self.set_diar_async(self.flags.diar_async)
        self.set_diar_compile(self.flags.diar_compile)
        self._diar_graphs = DiarGraphs(torch, diar) if self.flags.diar_graphs else None

    # ------------------------------------------------------------------ ASR encoder / decoder
    @property
    def no_bypass(self):
        return self.flags.enc_graphs

    def _enc_ctx(self):
        if self.flags.asr_dtype == "autocast":
            return self.torch.amp.autocast("cuda", dtype=self.torch.bfloat16)
        return nullcontext()

    def _pad_rows(self, t, rows, dim=0):
        n = t.shape[dim]
        if n >= rows:
            return t
        shape = list(t.shape)
        shape[dim] = rows - n
        return self.torch.cat([t, t.new_zeros(shape)], dim=dim)

    def _conformer_stream_step(self, **kw):
        if self.no_bypass:
            kw["bypass_pre_encode"] = False
        n = kw["processed_signal"].shape[0]
        rows = self.flags.pad_rows
        if rows <= 0 or n == rows:
            return self._orig["css"](**kw)
        if n > rows:
            return self._slabbed_stream_step(kw, n, rows)
        # Pad the speaker rows to a constant count (nemo_patches/multitalker_batch_fix.py semantics):
        # pad rows carry zero features, length 0 (inactive in the label-looping decoder), zero caches,
        # no hypotheses; the speaker targets are already padded by _set_speaker_targets.
        pad = rows - n
        kw["processed_signal"] = self._pad_rows(kw["processed_signal"], rows)
        if kw.get("processed_signal_length") is not None:
            kw["processed_signal_length"] = self._pad_rows(kw["processed_signal_length"], rows)
        if kw.get("cache_last_channel") is not None:
            kw["cache_last_channel"] = self._pad_rows(kw["cache_last_channel"], rows, dim=1)
            kw["cache_last_time"] = self._pad_rows(kw["cache_last_time"], rows, dim=1)
            kw["cache_last_channel_len"] = self._pad_rows(kw["cache_last_channel_len"], rows)
        if kw.get("previous_hypotheses") is not None:
            kw["previous_hypotheses"] = list(kw["previous_hypotheses"]) + [None] * pad
        if kw.get("previous_pred_out") is not None:
            kw["previous_pred_out"] = list(kw["previous_pred_out"]) + [None] * pad
        out = list(self._orig["css"](**kw))
        out[0] = out[0][:n] if out[0] is not None else None
        out[1] = out[1][:n] if out[1] is not None else None
        if out[2] is not None:
            out[2], out[3], out[4] = out[2][:, :n], out[3][:, :n], out[4][:n]
        if out[5] is not None:
            out[5] = out[5][:n]
        if len(out) > 6:
            out[6], out[7] = out[6][:n], out[7][:n]
        return tuple(out)

    def _slabbed_stream_step(self, kw, n, rows):
        """More speaker rows than ``pad_rows``: run them as ceil(n / rows) slabs of ``rows`` (each padded),
        so every model call keeps the one graphed shape. The first call at 64 rows faults with
        ``cudaErrorIllegalAddress`` on this NeMo build (encoder graph capture + label-looping decoder
        state at that width, fp32 or bf16; 8/16/32 never), so a 64-row ASR call is never issued.
        Numerics: fp32 is row- and shape-invariant, so slabs == one 64-row call; bf16 is a shape draw
        per slab like any other constant-row choice. Speaker targets were set for all n rows by
        ``_set_speaker_targets`` (kept in ``_tgt_full``); each slab re-sets its own."""
        torch = self.torch
        full = self._tgt_full
        outs = []
        for a in range(0, n, rows):
            b = min(n, a + rows)
            sk = dict(kw)
            sk["processed_signal"] = kw["processed_signal"][a:b]
            if kw.get("processed_signal_length") is not None:
                sk["processed_signal_length"] = kw["processed_signal_length"][a:b]
            if kw.get("cache_last_channel") is not None:
                sk["cache_last_channel"] = kw["cache_last_channel"][:, a:b]
                sk["cache_last_time"] = kw["cache_last_time"][:, a:b]
                sk["cache_last_channel_len"] = kw["cache_last_channel_len"][a:b]
            if kw.get("previous_hypotheses") is not None:
                sk["previous_hypotheses"] = list(kw["previous_hypotheses"][a:b])
            if kw.get("previous_pred_out") is not None:
                sk["previous_pred_out"] = list(kw["previous_pred_out"][a:b])
            if full is not None and full[0] is not None:
                self._set_speaker_targets(full[0][a:b], None if full[1] is None else full[1][a:b])
            outs.append(list(self._conformer_stream_step(**sk)))   # slab <= rows: the padded single call
        out = list(outs[0])
        for o in outs[1:]:
            for i in (0, 1, 5):
                if out[i] is not None:
                    out[i] = list(out[i]) + list(o[i])
            if out[2] is not None:
                out[2] = torch.cat([out[2], o[2]], dim=1)
                out[3] = torch.cat([out[3], o[3]], dim=1)
                out[4] = torch.cat([out[4], o[4]], dim=0)
            if len(out) > 6:
                out[6] = torch.cat([out[6], o[6]], dim=0)
                out[7] = torch.cat([out[7], o[7]], dim=0)
        return tuple(out)

    def _enc_step(self, processed_signal, processed_signal_length=None, cache_last_channel=None,
                  cache_last_time=None, cache_last_channel_len=None, keep_all_outputs=True,
                  drop_extra_pre_encoded=None, bypass_pre_encode=False):
        torch = self.torch
        if self.flags.asr_dtype != "autocast":
            processed_signal = processed_signal.to(self._enc_dtype)
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        if self.flags.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True   # encoder-scoped; the mel filterbank stays exact
        try:
            with self._enc_ctx():
                return self._orig["enc_step"](
                    processed_signal, processed_signal_length=processed_signal_length,
                    cache_last_channel=cache_last_channel, cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len, keep_all_outputs=keep_all_outputs,
                    drop_extra_pre_encoded=drop_extra_pre_encoded, bypass_pre_encode=bypass_pre_encode)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

    def _init_cache(self, batch_size=1, dtype=None, device=None, max_dim=0):
        # NeMo defaults the cache dtype to fp32; caches must match the encoder weights.
        return self._orig["init_cache"](batch_size=batch_size, dtype=self._enc_dtype, device=device, max_dim=max_dim)

    def _dec(self, *args, **kwargs):
        # NeMo runs the label-looping decoder with autocast DISABLED (label_looping_base.py:323), so
        # in the baseline the joint projected an fp32 encoder output with fp32 weights. Feed it fp32
        # whatever the encoder dtype is, so the decoder graph and its numerics are untouched.
        enc = kwargs.get("encoder_output")
        if enc is not None and enc.dtype != self.torch.float32:
            kwargs["encoder_output"] = enc.float()
        elif enc is None and args and args[0].dtype != self.torch.float32:
            args = (args[0].float(),) + tuple(args[1:])
        return self._orig["dec"](*args, **kwargs)

    def _set_speaker_targets(self, spk_targets=None, bg_spk_targets=None):
        if spk_targets is None:
            self._tgt_full = None
            return self._orig["set_speaker_targets"](None, None)
        if self.flags.pad_rows > 0 and spk_targets.shape[0] > self.flags.pad_rows:
            self._tgt_full = (spk_targets, bg_spk_targets)     # sliced per slab by _slabbed_stream_step
            return None
        self._tgt_full = (spk_targets, bg_spk_targets)
        if self.flags.pad_rows > 0 and spk_targets.shape[0] < self.flags.pad_rows:
            spk_targets = self._pad_rows(spk_targets, self.flags.pad_rows)
            bg_spk_targets = self._pad_rows(bg_spk_targets, self.flags.pad_rows) if bg_spk_targets is not None else None
        k = spk_targets.shape[0]
        bufs = self._spk_static.get(k)
        if bufs is None or bufs[0].shape != spk_targets.shape:
            dt = self._enc_dtype if self.flags.asr_dtype != "autocast" else self.torch.float32
            bufs = (self.torch.empty(spk_targets.shape, dtype=dt, device=spk_targets.device),
                    self.torch.empty(spk_targets.shape, dtype=dt, device=spk_targets.device))
            self._spk_static[k] = bufs
        bufs[0].copy_(spk_targets)
        bufs[1].copy_(bg_spk_targets if bg_spk_targets is not None else spk_targets)
        return self._orig["set_speaker_targets"](bufs[0], bufs[1])

    def set_asr_dtype(self, mode, warm=True):
        torch, asr = self.torch, self.asr
        self.flags.asr_dtype = mode
        dt = torch.bfloat16 if mode == "bf16" else torch.float32
        asr.encoder.to(dt)
        for name in ("spk_kernels", "bg_spk_kernels"):
            mod = getattr(asr, name, None)
            if mod is not None:
                mod.to(dt)
        self._enc_dtype = dt
        self._spk_static.clear()
        helper = getattr(asr.encoder, "_stream_step_cuda_graphs", None)
        if helper is not None:
            helper.reset_cuda_graphs_state()
        logger.info("ASR encoder dtype mode=%s (weights %s)", mode, dt)
        if warm:
            self.warm_asr()

    def set_enc_graphs(self, on, warm=True):
        self.flags.enc_graphs = bool(on)
        self.asr.encoder.set_streaming_cuda_graphs(enabled=bool(on), warmup_steps=1, max_graphs=self.max_speakers)
        if on and self.flags.asr_dtype == "autocast":
            logger.warning("enc_graphs=1 with asr_dtype=autocast: NeMo's wrapper refuses autocast, encoder runs eager")
        if warm:
            self.warm_asr()

    def enc_graphs_captured(self):
        helper = getattr(self.asr.encoder, "_stream_step_cuda_graphs", None)
        return len(helper._graphs) if helper is not None else 0

    def warm_asr(self):
        """Drive the ASR encoder+decoder at k=max..1 rows with injected speaker targets.

        Bypasses the diarizer gate (white-noise warmup never activates a speaker), so the encoder
        graphs for every k are captured, the RNNT decoder state is sized for k=max once (no
        recapture on speaker growth under traffic) and cuDNN/cuBLAS are warm for every shape.
        """
        torch, asr = self.torch, self.asr
        F = getattr(asr.encoder, "_feat_in", None) or asr.cfg.preprocessor.features
        cs = self.scfg.chunk_size[1] + self.scfg.pre_encode_cache_size[1]
        t0 = time.perf_counter()
        with torch.inference_mode():
            for k in list(range(self.max_speakers, 0, -1)):
                sig = torch.randn(k, F, cs, device=self.device) * 0.5
                ln = torch.full((k,), cs, dtype=torch.int64, device=self.device)
                caches = asr.encoder.get_initial_cache_state(batch_size=k)
                tgt = torch.ones(k, self.scfg.valid_out_len, device=self.device)
                asr.set_speaker_targets(tgt, torch.zeros_like(tgt))
                for _ in range(3):
                    x, xl = sig, ln
                    if not self.no_bypass:
                        with self._enc_ctx():
                            if self.flags.asr_dtype == "bf16":
                                x = x.to(torch.bfloat16)
                            x, xl = asr.encoder.pre_encode(x=x.transpose(1, 2), lengths=xl)
                            x = x[:, self.scfg.drop_extra_pre_encoded:, :]
                            xl = (xl.to(torch.int64) - self.scfg.drop_extra_pre_encoded).clamp(min=0)
                    out = asr.conformer_stream_step(
                        processed_signal=x, processed_signal_length=xl,
                        cache_last_channel=caches[0], cache_last_time=caches[1], cache_last_channel_len=caches[2],
                        keep_all_outputs=False, previous_hypotheses=[None] * k, previous_pred_out=[None] * k,
                        drop_extra_pre_encoded=self.scfg.drop_extra_pre_encoded, return_transcription=True,
                        bypass_pre_encode=not self.no_bypass)
                    caches = (out[2], out[3], out[4])
        torch.cuda.synchronize()
        asr.set_speaker_targets(None, None)
        logger.info("ASR warm: k_active=%s exercised (%s), 3 steps each, in %.1fs; encoder graphs captured=%d, "
                    "decoder mode=%s", list(range(self.max_speakers, 0, -1)),
                    f"padded to {self.flags.pad_rows} rows" if self.flags.pad_rows > 0 else "unpadded",
                    time.perf_counter() - t0, self.enc_graphs_captured(), self.decoder_mode())

    def decoder_mode(self):
        try:
            return str(self.asr.decoding.decoding.decoding_computer.cuda_graphs_mode)
        except Exception:  # noqa: BLE001
            return "unknown"

    # ------------------------------------------------------------------ instance manager
    def _get_active_speakers_info(self, im, active_speakers, chunk_audio, chunk_lengths):
        """NeMo's get_active_speakers_info without the per-step deepcopy of hypotheses."""
        torch = self.torch
        im._reset_active_speaker_buffers()
        for batch_idx, speaker_ids in enumerate(active_speakers):
            st = im.batch_asr_states[batch_idx]
            for speaker_id in speaker_ids:
                im._active_chunk_audio.append(chunk_audio[batch_idx, :])
                im._active_chunk_lengths.append(chunk_lengths[batch_idx])
                im._active_speaker_targets.append(im.diar_states.previous_chunk_preds[batch_idx, :, speaker_id])
                inactive = [o for o in speaker_ids if o != speaker_id]
                im._inactive_speaker_targets.append(
                    (im.diar_states.previous_chunk_preds[batch_idx, :, inactive] > 0.5).sum(dim=-1) > 0)
                if speaker_id not in st.get_speakers():
                    im.add_speaker(batch_idx, speaker_id)
                im._active_previous_hypotheses.append(st.previous_hypothesis[speaker_id])
                im._active_asr_pred_out_stream.append(st.previous_pred_out[speaker_id])
                im._active_cache_last_channel.append(st.cache_last_channel[:, speaker_id])
                im._active_cache_last_time.append(st.cache_last_time[:, speaker_id])
                im._active_cache_last_channel_len.append(st.cache_last_channel_len[speaker_id])
        if len(im._active_chunk_audio) == 0:
            return None, None, None, None
        active_chunk_audio = torch.stack(im._active_chunk_audio)
        active_chunk_lengths = torch.stack(im._active_chunk_lengths)
        active_speaker_targets = torch.stack(im._active_speaker_targets)
        inactive_speaker_targets = torch.stack(im._inactive_speaker_targets)
        im.active_previous_hypotheses = list(im._active_previous_hypotheses)
        im.active_asr_pred_out_stream = list(im._active_asr_pred_out_stream)
        im.active_cache_last_channel = torch.stack(im._active_cache_last_channel).transpose(0, 1)
        im.active_cache_last_time = torch.stack(im._active_cache_last_time).transpose(0, 1)
        im.active_cache_last_channel_len = torch.stack(im._active_cache_last_channel_len)
        return active_chunk_audio, active_chunk_lengths, active_speaker_targets, inactive_speaker_targets

    # ------------------------------------------------------------------ diarizer
    def set_diar_async(self, on):
        self.flags.diar_async = bool(on)
        self.diar.async_streaming = bool(on)
        self.diar.async_pad_to_max = bool(on)
        sm = self.diar.sortformer_modules
        # The async FIFO update can randomise the FIRST pop length (async_desync_updates, meant to
        # stagger offline batches); that makes the cache schedule nondeterministic and different
        # from sync mode. Parity with the shipped sync schedule is what we want.
        if getattr(sm, "async_desync_updates", False):
            logger.warning("diarizer async_desync_updates was True in the checkpoint config; forcing False")
        sm.async_desync_updates = False

    def _init_streaming_state(self, batch_size=1, async_streaming=False, device=None):
        return self._orig["init_state"](batch_size=batch_size, async_streaming=self.flags.diar_async,
                                        device=device if device is not None else self.device)

    def set_diar_compile(self, mode):
        mode = str(mode).strip().lower()
        self.flags.diar_compile = mode
        self._diar_compile_mode = mode
        if mode in ("0", "off", "false", ""):
            self._diar_core = None
            return
        if not self.flags.diar_async:
            logger.warning("diar_compile needs diar_async=1 (fixed shapes); leaving the diarizer eager")
            self._diar_core = None
            return
        torch, diar, sm = self.torch, self.diar, self.diar.sortformer_modules
        torch._dynamo.config.cache_size_limit = 64

        def core(chunk_embs, chunk_lens, spkcache, spkcache_lengths, fifo, fifo_lengths, output_length: int):
            embs, lens = sm.concat_and_pad([spkcache, fifo, chunk_embs],
                                           [spkcache_lengths, fifo_lengths, chunk_lens], output_length=output_length)
            emb_seq, emb_len = diar.frontend_encoder(processed_signal=embs, processed_signal_length=lens,
                                                     bypass_pre_encode=True)
            preds = diar.forward_infer(emb_seq=emb_seq, emb_seq_length=emb_len)
            return preds, emb_len

        kw = {"dynamic": False}
        if mode == "reduce-overhead":
            kw["mode"] = "reduce-overhead"
        self._diar_core = torch.compile(core, **kw)
        logger.info("diarizer core torch.compile(%s) armed; compiles on first step", kw)

    def compiled_graphs(self):
        try:
            return int(self.torch._dynamo.utils.counters["stats"]["unique_graphs"])
        except Exception:  # noqa: BLE001
            return -1

    def _diar_step(self, processed_signal, processed_signal_length, streaming_state, total_preds,
                   drop_extra_pre_encoded=0, left_offset=0, right_offset=0, **kw):
        args = dict(processed_signal=processed_signal, processed_signal_length=processed_signal_length,
                    drop_extra_pre_encoded=drop_extra_pre_encoded, left_offset=left_offset,
                    right_offset=right_offset)
        graphs_path = (self._diar_graphs is not None and self.flags.diar_graphs and not self.flags.diar_async
                       and not kw.get("return_logits"))
        with self.torch.amp.autocast("cuda", dtype=self.torch.bfloat16):
            if graphs_path:
                out = self._diar_step_sync_graphs(processed_signal, processed_signal_length, streaming_state,
                                                  total_preds, drop_extra_pre_encoded, left_offset, right_offset)
            elif self._diar_core is None or not self.flags.diar_async or kw.get("return_logits"):
                out = self._orig["diar_step"](streaming_state=streaming_state, total_preds=total_preds, **args, **kw)
            else:
                out = self._diar_step_async_compiled(processed_signal, processed_signal_length, streaming_state,
                                                     total_preds, drop_extra_pre_encoded, left_offset, right_offset)
            if self.flags.diar_verify and (self.flags.diar_async or graphs_path) and not kw.get("return_logits"):
                self._diar_shadow_compare(streaming_state, total_preds, out, args)
        return out

    def _diar_step_sync_graphs(self, processed_signal, processed_signal_length, st, total_preds,
                               drop, left_offset, right_offset):
        """``forward_streaming_step`` sync branch (no logits) with the core replayed from a CUDA graph.

        Everything outside the core (pre-encode, concat, mask, FIFO/cache update, high-resolution
        slicing) is the same eager code the original runs; the core replay is bit-identical to eager.
        """
        torch, diar, sm = self.torch, self.diar, self.diar.sortformer_modules
        chunk_embs, chunk_lens = diar._call_pre_encode(processed_signal, processed_signal_length)
        if drop > 0:
            chunk_embs = chunk_embs[:, drop:, :]
            chunk_lens = chunk_lens - drop
        embs = sm.concat_embs([st.spkcache, st.fifo, chunk_embs], dim=1, device=chunk_embs.device)
        lens = st.spkcache.shape[1] + st.fifo.shape[1] + chunk_lens
        preds, enc_lens = self._diar_graphs.run(embs, lens)
        sub = diar.encoder.subsampling_factor
        lc_enc = round(left_offset / sub)
        rc_enc = math.ceil(right_offset / sub)
        high_res = None
        if diar.high_resolution:
            high_res = preds
            preds = sm.downsample_preds(high_res, diar.upsample_factor).detach()
        preds = sm.apply_mask_to_preds(preds, enc_lens)
        saved_sc, saved_f = st.spkcache.shape[1], st.fifo.shape[1]
        st, chunk_preds = sm.streaming_update(streaming_state=st, chunk=chunk_embs, preds=preds, lc=lc_enc, rc=rc_enc)
        if diar.high_resolution:
            chunk_len = chunk_embs.shape[1] - lc_enc - rc_enc
            start = (saved_sc + saved_f + lc_enc) * diar.upsample_factor
            chunk_preds = high_res[:, start: start + chunk_len * diar.upsample_factor]
        native = 1 if diar.high_resolution else sub
        ds = diar.output_subsampling_factor // native
        if ds > 1:
            chunk_preds = sm.downsample_preds(chunk_preds, ds)
        return st, torch.cat([total_preds, chunk_preds], dim=1)

    def _diar_shadow_compare(self, st_in, total_in, out, args):
        """MT_DIAR_VERIFY: drive a shadow SYNC-mode state on the same chunks; log the divergence
        of this step's chunk predictions (max |dp|, flipped 0.5-decisions) against the async path."""
        torch, diar = self.torch, self.diar
        key = id(st_in)
        shadow = self._shadows.get(key)
        if total_in.shape[1] == 0 or shadow is None:
            sst = self._orig["init_state"](batch_size=st_in.spkcache.shape[0], async_streaming=False, device=self.device)
            shadow = {"state": sst, "total": torch.zeros((st_in.spkcache.shape[0], 0, total_in.shape[2]),
                                                          device=self.device), "step": 0, "n_diff": 0}
            self._shadows[key] = shadow
        was = (diar.async_streaming, diar.async_pad_to_max)
        diar.async_streaming = diar.async_pad_to_max = False
        try:
            # The shadow is NeMo's own sync path (eager); note the shadow shares the pre-encode
            # inputs only, its cache/FIFO evolve independently.
            shadow["state"], shadow["total"] = self._orig["diar_step"](
                streaming_state=shadow["state"], total_preds=shadow["total"], **args)
        finally:
            diar.async_streaming, diar.async_pad_to_max = was
        n = shadow["total"].shape[1] - total_in.shape[1]     # frames added this step
        a = out[1][:, -n:]
        b = shadow["total"][:, -n:]
        if a.shape == b.shape:
            d = (a - b).abs().max().item()
            flips = ((a > 0.5) != (b > 0.5)).sum().item()
        else:
            d, flips = float("inf"), -1
        shadow["n_diff"] += int(flips > 0)
        if flips or d > 0 or shadow["step"] % 20 == 0:
            sst = shadow["state"]
            if getattr(st_in, "spkcache_lengths", None) is not None:
                mine = f"async sc={int(st_in.spkcache_lengths[0])} fifo={int(st_in.fifo_lengths[0])} comp={bool(st_in.spkcache_compressed[0])}"
            else:
                mine = f"sync-graphs sc={st_in.spkcache.shape[1]} fifo={st_in.fifo.shape[1]} comp={bool(st_in.spkcache_compressed)}"
            logger.info("DIARVERIFY step=%d n=%d max_abs=%.6f flips=%d | %s | shadow-sync sc=%d fifo=%d comp=%s",
                        shadow["step"], n, d, flips, mine, sst.spkcache.shape[1], sst.fifo.shape[1],
                        bool(sst.spkcache_compressed))
        shadow["step"] += 1
        if len(self._shadows) > 64:   # sessions end without notice; keep the dict bounded
            self._shadows.pop(next(iter(self._shadows)))

    def _diar_step_async_compiled(self, processed_signal, processed_signal_length, st, total_preds,
                                  drop, left_offset, right_offset):
        """``SortformerEncLabelModel.forward_streaming_step`` (async branch, no logits) with the
        pre-encode -> concat -> encoder -> head section replaced by the compiled core."""
        torch, diar, sm = self.torch, self.diar, self.diar.sortformer_modules
        output_length = st.spkcache.shape[1] + st.fifo.shape[1] + sm.chunk_left_context + sm.chunk_len \
            + sm.chunk_right_context
        chunk_embs, chunk_lens = diar._call_pre_encode(processed_signal, processed_signal_length)
        if drop > 0:
            chunk_embs = chunk_embs[:, drop:, :]
            chunk_lens = chunk_lens - drop
        try:
            if self._diar_compile_mode == "reduce-overhead":
                torch.compiler.cudagraph_mark_step_begin()
            preds, enc_lens = self._diar_core(chunk_embs, chunk_lens, st.spkcache, st.spkcache_lengths,
                                              st.fifo, st.fifo_lengths, int(output_length))
        except Exception:  # noqa: BLE001 — the core is side-effect free; eager is correct, just slower
            logger.exception("diarizer compiled core failed; running the diarizer eager from now on")
            self._diar_core = None
            return self._orig["diar_step"](
                processed_signal=processed_signal, processed_signal_length=processed_signal_length,
                streaming_state=st, total_preds=total_preds, drop_extra_pre_encoded=drop,
                left_offset=left_offset, right_offset=right_offset)
        base = self._graphs_after_warmup
        if base is not None:
            g = self.compiled_graphs()
            if g > base:
                logger.warning("diarizer recompiled under traffic: unique_graphs %d -> %d", base, g)
                self._graphs_after_warmup = g

        sub = diar.encoder.subsampling_factor
        lc_enc = round(left_offset / sub)
        rc_enc = math.ceil(right_offset / sub)
        high_res = None
        if diar.high_resolution:
            high_res = preds
            preds = sm.downsample_preds(high_res, diar.upsample_factor).detach()
        preds = sm.apply_mask_to_preds(preds, enc_lens)
        saved_sc, saved_f = st.spkcache_lengths.clone(), st.fifo_lengths.clone()
        st, chunk_preds = sm.streaming_update_async(streaming_state=st, chunk=chunk_embs, chunk_lengths=chunk_lens,
                                                    preds=preds, lc=lc_enc, rc=rc_enc)
        if diar.high_resolution:
            max_chunk_len = chunk_embs.shape[1] - lc_enc - rc_enc
            cl = (chunk_lens - lc_enc).clamp(min=0, max=max_chunk_len)
            chunk_preds = diar._extract_async_high_resolution_chunk_preds(
                high_resolution_preds=high_res, spkcache_lengths=saved_sc, fifo_lengths=saved_f,
                chunk_lengths=cl, max_chunk_len=max_chunk_len, lc_enc=lc_enc)
        native = 1 if diar.high_resolution else sub
        ds = diar.output_subsampling_factor // native
        if ds > 1:
            chunk_preds = sm.downsample_preds(chunk_preds, ds)
        total_preds = torch.cat([total_preds, chunk_preds], dim=1)
        return st, total_preds

    def mark_warm(self):
        self._graphs_after_warmup = self.compiled_graphs()
        return self._graphs_after_warmup

    def diar_graphs_captured(self):
        return len(self._diar_graphs.graphs) if self._diar_graphs is not None else 0

    # ------------------------------------------------------------------ runtime A/B control
    def apply_control(self, ctl: dict, live: int) -> dict:
        """Flip levers while no session is live, then re-warm. Returns the applied flag set.

        Warm-up runs outside the step lock: ``warm_diar_fn`` (set by the model) drives sessions
        that take the lock themselves, and nothing else is live by construction.
        """
        if live > 0:
            raise ValueError(f"control refused: {live} live session(s)")
        warm_asr = warm_diar = False
        with self.lock:
            if "asr_dtype" in ctl and str(ctl["asr_dtype"]).lower() != self.flags.asr_dtype:
                # Re-allocating the encoder parameters under a live decoder CUDA graph corrupted the
                # CUDA context once (illegal memory access on the next encoder step); dtype is
                # fixed per deployment (MT_ASR_DTYPE).
                raise ValueError("asr_dtype cannot be changed at runtime; redeploy with MT_ASR_DTYPE")
            if "enc_graphs" in ctl:
                on = bool(int(ctl["enc_graphs"]))
                if on and not self.flags.enc_graphs:
                    raise ValueError("enc_graphs can only be turned OFF at runtime (capture happens at load)")
                if not on and self.flags.enc_graphs:
                    self.set_enc_graphs(False, warm=False)
                    warm_asr = True
            if "diar_async" in ctl:
                self.set_diar_async(bool(int(ctl["diar_async"])))
                if self._diar_core is not None and not self.flags.diar_async:
                    self._diar_core = None
                warm_diar = True
            if "diar_compile" in ctl:
                self.set_diar_compile(ctl["diar_compile"])
                warm_diar = True
            if "diar_graphs" in ctl:
                on = bool(int(ctl["diar_graphs"]))
                if on and self._diar_graphs is None:
                    self._diar_graphs = DiarGraphs(self.torch, self.diar)   # captures during the re-warm
                    warm_diar = True
                self.flags.diar_graphs = on
            if "no_deepcopy" in ctl:
                self.flags.no_deepcopy = bool(int(ctl["no_deepcopy"]))
            if "inc_detok" in ctl:
                self.flags.inc_detok = bool(int(ctl["inc_detok"]))
            if "inc_detok_verify" in ctl:
                self.flags.inc_detok_verify = bool(int(ctl["inc_detok_verify"]))
                self.inc.stats = {k: 0 for k in self.inc.stats}
            if "mel" in ctl:
                self.flags.mel = bool(int(ctl["mel"]))
            if "mel_verify" in ctl:
                self.flags.mel_verify = bool(int(ctl["mel_verify"]))
            if "diar_verify" in ctl:
                self.flags.diar_verify = bool(int(ctl["diar_verify"]))
                self._shadows.clear()
            if "tf32" in ctl and bool(int(ctl["tf32"])) != self.flags.tf32:
                raise ValueError("tf32 is baked into the captured encoder graphs; redeploy with MT_TF32")
            if "pad_rows" in ctl and int(ctl["pad_rows"]) != self.flags.pad_rows:
                raise ValueError("pad_rows is baked into the captured encoder graphs; redeploy with MT_PAD_ROWS")
        if warm_asr:
            self.warm_asr()
        if warm_diar and self.warm_diar_fn is not None:
            self.warm_diar_fn()
        self.mark_warm()
        return self.flags.as_dict()
