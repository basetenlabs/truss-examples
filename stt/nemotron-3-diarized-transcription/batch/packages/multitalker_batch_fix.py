"""Batch-shape pinning for NeMo's multitalker lockstep streaming step (pinned NeMo 3c2d62ae7eb4).

Why: on RTX PRO 6000 with bf16 autocast, identical rows inside ONE batched call come back
different at bf16-ulp level because cuBLAS picks split-K/stream-K GEMM kernels for some
(M, N, K) shapes (first seen in the ASR encoder's ``self_attn.linear_q``, M = SigmaS*14,
N = K = 1024) whose reduction order depends on the output tile. The 24-layer streaming
conformer and greedy RNNT amplify one ulp into different tokens, and every GEMM's rounding also
depends on M, so a B=8 session can never bit-match B=1 while the shapes differ. There is no
indexing bug in NeMo's driver; the divergence is numerical and shape-dependent.

Fix: run every batched model call at a FIXED batch shape by zero-padding rows and slicing the
results back, so B=1 and B=8 (and any SigmaS) execute the same kernels and produce bit-identical
per-row results (rows at 64 ASR rows / 8 diarizer rows were verified row-invariant):
  * ``conformer_stream_step``: pad SigmaS active-speaker rows to ``asr_rows`` (default 64 = 8
    sessions x 8 speakers). Pad rows have length 0 (inactive in the label-looping decoder).
  * ``SpeakerTaggedASR.forward_pre_encoded`` and ``_forward_diarization_streaming_step``: pad the
    session batch to ``diar_rows`` (default 8). The diarizer streaming state keeps ``diar_rows``
    rows across steps; ``diar_pred_out_stream`` is sliced back to the real batch.
Optionally ``asr_fp32=True`` runs the ASR step without autocast (also row-invariant; changes
numerics vs the bf16 checkpoint reference, so it is off by default).

Usage (before the first session):
    from multitalker_batch_fix import apply
    apply(asr_model, diar_model, asr_rows=64, diar_rows=8)
"""

import torch

_APPLIED = {}


def _pad_rows(t, rows, dim=0):
    n = t.shape[dim]
    if n >= rows:
        return t
    shape = list(t.shape)
    shape[dim] = rows - n
    return torch.cat([t, t.new_zeros(shape)], dim=dim)


def patch_conformer_stream_step(asr_model, asr_rows: int, asr_fp32: bool = False):
    cls = type(asr_model)
    if getattr(cls, "_mt_batch_fix_css", None):
        return
    orig = cls.conformer_stream_step

    def conformer_stream_step(self, processed_signal, processed_signal_length=None, cache_last_channel=None,
                              cache_last_time=None, cache_last_channel_len=None, keep_all_outputs=True,
                              previous_hypotheses=None, previous_pred_out=None, drop_extra_pre_encoded=None,
                              return_transcription=True, return_log_probs=False, bypass_pre_encode=False):
        n = processed_signal.shape[0]
        rows = max(asr_rows, n)
        pad = rows - n
        ac = torch.autocast("cuda", enabled=False) if asr_fp32 else _nullctx()
        if pad == 0 and not asr_fp32:
            return orig(self, processed_signal, processed_signal_length, cache_last_channel, cache_last_time,
                        cache_last_channel_len, keep_all_outputs, previous_hypotheses, previous_pred_out,
                        drop_extra_pre_encoded, return_transcription, return_log_probs, bypass_pre_encode)
        if asr_fp32:
            processed_signal = processed_signal.float()
        spk_t, bg_t = getattr(self, "spk_targets", None), getattr(self, "bg_spk_targets", None)
        try:
            if pad:
                processed_signal = _pad_rows(processed_signal, rows)
                if processed_signal_length is not None:
                    processed_signal_length = _pad_rows(processed_signal_length, rows)  # pad rows: length 0
                if cache_last_channel is not None:
                    cache_last_channel = _pad_rows(cache_last_channel, rows, dim=1)
                    cache_last_time = _pad_rows(cache_last_time, rows, dim=1)
                    cache_last_channel_len = _pad_rows(cache_last_channel_len, rows)
                if previous_hypotheses is not None:
                    previous_hypotheses = list(previous_hypotheses) + [None] * pad
                if previous_pred_out is not None:
                    previous_pred_out = list(previous_pred_out) + [None] * pad
                if spk_t is not None:
                    self.spk_targets = _pad_rows(spk_t, rows)
                if bg_t is not None:
                    self.bg_spk_targets = _pad_rows(bg_t, rows)
            with ac:
                out = orig(self, processed_signal, processed_signal_length, cache_last_channel, cache_last_time,
                           cache_last_channel_len, keep_all_outputs, previous_hypotheses, previous_pred_out,
                           drop_extra_pre_encoded, return_transcription, return_log_probs, bypass_pre_encode)
        finally:
            if pad:
                if spk_t is not None:
                    self.spk_targets = spk_t
                if bg_t is not None:
                    self.bg_spk_targets = bg_t
        if not pad:
            return out
        out = list(out)
        out[0] = out[0][:n] if out[0] is not None else None            # greedy_predictions (list)
        out[1] = out[1][:n] if out[1] is not None else None            # hyps / texts (list)
        if out[2] is not None:
            out[2] = out[2][:, :n]                                     # cache_last_channel_next [L, rows, ...]
            out[3] = out[3][:, :n]
            out[4] = out[4][:n]
        if out[5] is not None:
            out[5] = out[5][:n]                                        # best_hyp (list)
        if return_log_probs and len(out) > 6:
            out[6] = out[6][:n]
            out[7] = out[7][:n]
        return tuple(out)

    cls.conformer_stream_step = conformer_stream_step
    cls._mt_batch_fix_css = orig


def patch_speaker_tagged_asr(diar_rows: int):
    from nemo.collections.asr.parts.utils import multispk_transcribe_utils as U

    cls = U.SpeakerTaggedASR
    if getattr(cls, "_mt_batch_fix_diar", None):
        return
    orig_pre = cls.forward_pre_encoded
    orig_diar = cls._forward_diarization_streaming_step

    def forward_pre_encoded(self, audio_signal, length, drop_extra_pre_encoded=0):
        # asr_fp32 covers the whole ASR path, pre-encode included (the caller's autocast is bf16)
        ac = torch.autocast("cuda", enabled=False) if _APPLIED.get("asr_fp32") else _nullctx()
        n = audio_signal.shape[0]
        with ac:
            if n >= diar_rows:
                return orig_pre(self, audio_signal, length, drop_extra_pre_encoded)
            # pad rows carry the same length as row 0 so the conv subsampling sees one uniform shape
            a = _pad_rows(audio_signal, diar_rows)
            lens = torch.cat([length, length[:1].expand(diar_rows - n)])
            x, xl = orig_pre(self, a, lens, drop_extra_pre_encoded)
        return x[:n], xl[:n]

    def _forward_diarization_streaming_step(self, diar_chunk_audio, diar_chunk_lengths, drop_extra_pre_encoded):
        if _APPLIED.get("diar_fp32"):
            # fp32 diarizer is also row-POSITION invariant (bf16 is not at some sync-mode shapes); ~4x cost.
            with torch.autocast("cuda", enabled=False):
                return _forward_diarization_streaming_step_padded(self, diar_chunk_audio.float(), diar_chunk_lengths, drop_extra_pre_encoded)
        return _forward_diarization_streaming_step_padded(self, diar_chunk_audio, diar_chunk_lengths, drop_extra_pre_encoded)

    def _forward_diarization_streaming_step_padded(self, diar_chunk_audio, diar_chunk_lengths, drop_extra_pre_encoded):
        n = diar_chunk_audio.shape[0]
        if n == diar_rows:
            return orig_diar(self, diar_chunk_audio, diar_chunk_lengths, drop_extra_pre_encoded)
        if n > diar_rows:
            return _forward_diarization_slabs(self, diar_chunk_audio, diar_chunk_lengths, drop_extra_pre_encoded)
        ds = self.instance_manager.diar_states
        st = ds.streaming_state
        # first step: state was initialised with n rows -> grow it to diar_rows once
        if st.spkcache is not None and st.spkcache.shape[0] < diar_rows:
            for f in ("spkcache", "spkcache_preds", "spkcache_lengths", "fifo", "fifo_lengths", "fifo_preds",
                      "mean_sil_emb", "n_sil_frames"):
                v = getattr(st, f, None)
                if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == n:
                    setattr(st, f, _pad_rows(v, diar_rows))
            if isinstance(st.spkcache_compressed, torch.Tensor) and st.spkcache_compressed.shape[0] == n:
                st.spkcache_compressed = _pad_rows(st.spkcache_compressed, diar_rows)
        total = ds.diar_pred_out_stream
        ds.diar_pred_out_stream = _pad_rows(total, diar_rows)
        try:
            new_state, new_total = orig_diar(
                self, _pad_rows(diar_chunk_audio, diar_rows),
                torch.cat([diar_chunk_lengths, diar_chunk_lengths[:1].expand(diar_rows - n)]),
                drop_extra_pre_encoded)
        finally:
            ds.diar_pred_out_stream = total
        return new_state, new_total[:n]

    def _forward_diarization_slabs(self, diar_chunk_audio, diar_chunk_lengths, drop_extra_pre_encoded):
        """A batch wider than ``diar_rows`` runs the diarizer as slabs of ``diar_rows`` sessions, each
        with its own slice of the sync streaming state, so the bf16 diarizer sees exactly the shapes a
        ``diar_rows``-session batch sees (a 16-row diarizer call is a different rounding draw: eval-30
        +0.56 cpWER). Rows are in lockstep, so every slab's state tensors have equal shapes and the
        slabs concatenate back; the sync mode's scalar ``spkcache_compressed`` is shared."""
        n = diar_chunk_audio.shape[0]
        ds = self.instance_manager.diar_states
        st, total = ds.streaming_state, ds.diar_pred_out_stream
        names = [k for k in vars(type(st)) if not k.startswith("_") and not callable(getattr(type(st), k))]
        new_states, totals = [], []
        try:
            for a in range(0, n, diar_rows):
                b = min(n, a + diar_rows)
                sub = type(st)()
                for k in names:
                    v = getattr(st, k)
                    setattr(sub, k, v[a:b] if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == n else v)
                ds.streaming_state, ds.diar_pred_out_stream = sub, total[a:b]
                ns, nt = _forward_diarization_streaming_step_padded(
                    self, diar_chunk_audio[a:b], diar_chunk_lengths[a:b], drop_extra_pre_encoded)
                if b - a < diar_rows:   # a short last slab came back padded: keep its real rows only
                    for k in names:
                        v = getattr(ns, k)
                        if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == diar_rows:
                            setattr(ns, k, v[: b - a])
                new_states.append(ns)
                totals.append(nt)
        finally:
            ds.streaming_state, ds.diar_pred_out_stream = st, total
        merged = type(st)()
        for k in names:
            vs = [getattr(s, k) for s in new_states]
            v0 = vs[0]
            if isinstance(v0, torch.Tensor) and v0.dim() >= 1 and sum(v.shape[0] for v in vs) == n:
                setattr(merged, k, torch.cat(vs, dim=0))
            else:
                setattr(merged, k, v0)
        return merged, torch.cat(totals, dim=0)

    cls.forward_pre_encoded = forward_pre_encoded
    cls._forward_diarization_streaming_step = _forward_diarization_streaming_step
    cls._mt_batch_fix_diar = (orig_pre, orig_diar)


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def apply(asr_model, diar_model=None, asr_rows: int = 64, diar_rows: int = 8, asr_fp32: bool = False,
          diar_fp32: bool = False, pin_asr: bool = True):
    """Install the shape-pinning patches. Idempotent. Call once before any streaming session.

    diar_fp32=True additionally casts ``diar_model`` to fp32 and runs the diarizer step without autocast:
    bf16 kernels are still row-position dependent at a few sync-mode shapes (a session alone vs the same
    session as row b of a batch can differ by ~1e-4 in speaker posteriors; 2/30 eval sessions changed a
    segment boundary), fp32 removes that at ~4x diarizer cost. Not validated for cpWER; opt-in.
    pin_asr=False leaves ``conformer_stream_step`` alone (``mt_fast.FastPath`` pads the ASR rows and owns
    the encoder dtype itself); the diarizer-row pinning still applies.
    """
    if pin_asr:
        patch_conformer_stream_step(asr_model, asr_rows, asr_fp32)
    patch_speaker_tagged_asr(diar_rows)
    if diar_fp32 and diar_model is not None:
        diar_model.float()
    _APPLIED.update(asr_rows=asr_rows, diar_rows=diar_rows, asr_fp32=asr_fp32, diar_fp32=diar_fp32)
    return dict(_APPLIED)
