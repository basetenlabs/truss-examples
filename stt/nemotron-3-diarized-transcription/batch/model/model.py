"""Batch truss for coupled multitalker speaker-attributed transcription (NVIDIA NeMo).

Nemotron 3 Diarization (OpenMDW-1.1) + multitalker-parakeet-streaming-0.6b-v1 (NVIDIA Open Model License). English-only.

Pipeline: Nemotron-3-Diarization (streaming Sortformer) emits per-speaker activity; the
multitalker-parakeet-streaming-0.6b-v1 ASR runs ONE instance per speaker on the same mixed
audio, adapting to each speaker via speaker-kernel injection — so fully overlapped speech is
transcribed per speaker with no enrollment. Mirrors NeMo's
``speech_to_text_multitalker_streaming_infer.py``: a cache-aware streaming buffer over the
whole file, ``SpeakerTaggedASR.perform_parallel_streaming_stt_spk`` per chunk, SegLST out.

Request micro-batching: concurrent requests coalesce into ONE coupled session with B files on
the batch dimension (`instance_manager.batch_asr_states[b]`), all rows in lockstep from step 0.
The step is launch-bound (~50 ms for ~11 ms of GPU work at B=1), so B files cost about the same
wall as one until the GPU fills. Rows are zero-padded to the longest file so every row sees the
K=1 chunk geometry at every step; each row's output is cut at its true duration. Model calls are
shape-pinned (packages/multitalker_batch_fix.py) so B=1..cap are bit-identical per file — bf16
GEMM rounding here depends on the batch shape, and the pipeline amplifies one ulp into words.

Request:  {"transcription_input": {"audio": {"url" | "audio_b64"}, "max_speakers": 8}}
Response: {"segments": [{"speaker", "start", "end", "text"}], "speakers": n,
           "text_by_speaker": {"speaker_0": "..."}, "compute_s", "peak_gpu_gb", "batch_n"}
"""

import base64
import concurrent.futures
import contextlib
import copy
import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ASR encoder attention context [left, right] frames — ~1.12 s streaming advance (model card).
ATT_CTX = [70, 13]
# Diarizer streaming knobs paired with the ASR chunk geometry (model card recommendation).
SPKCACHE_LEN, FIFO_LEN, SPKCACHE_UPDATE_PERIOD = 264, 264, 222
MAX_SPEAKERS = int(os.environ.get("MT_MAX_SPEAKERS", "8"))
SAMPLE_RATE = 16000
# Micro-batching: collect for up to the window, at most cap files per coupled session.
MB_WINDOW_S = float(os.environ.get("MT_MB_WINDOW_MS", "100")) / 1000
MB_CAP = int(os.environ.get("MT_MB_CAP", "8"))
# Split a window's requests into more sessions when zero-padding to the longest file would
# waste more than this fraction of the batch's rows. Costs steps (one extra session's worth of
# the shorter group), saves latency for the short files; ≥1 disables.
MB_PAD_WASTE = float(os.environ.get("MT_MB_PAD_WASTE", "0.3"))
# Batch sizes to warm at load ("cap" = MT_MB_CAP). With shapes pinned (below) every batch size
# runs the same kernels, so B=1 alone warms everything.
MB_WARMUP_SIZES = [MB_CAP if s.strip() == "cap" else int(s)
                   for s in os.environ.get("MT_MB_WARMUP_SIZES", "1").split(",") if s.strip()]
# Shape pinning (see packages/multitalker_batch_fix.py): bf16 GEMM rounding depends on the batch
# shape, so every model call is zero-padded to a fixed row count — ASR to cap x max_speakers
# active-speaker rows, diarizer to cap sessions — making B=1..cap bit-identical per row.
ASR_ROWS = int(os.environ.get("MT_ASR_ROWS", str(MB_CAP * MAX_SPEAKERS)))
DIAR_ROWS = int(os.environ.get("MT_DIAR_ROWS", str(MB_CAP)))
# Diarizer step in fp32 (no autocast) instead of bf16. The ASR dtype is MT_ASR_DTYPE (mt_fast.Flags)
# with MT_FAST=1, MT_ASR_FP32 with the eager path.
DIAR_FP32 = os.environ.get("MT_DIAR_FP32", "0") == "1"
# Step-time levers (packages/mt_fast.py, shared with ../streaming): sync-mode diarizer CUDA graphs,
# encoder CUDA graphs, no per-step deepcopy, incremental detokenisation. MT_FAST=0 = NeMo's eager step
# with the batch fix's own ASR-row pinning (the e98c7e8 build), kept for A/B.
FAST = os.environ.get("MT_FAST", "1") == "1"
# Warm-session length (s) at load: the sync diarizer walks ~150 sequence lengths, each a graph capture.
WARMUP_SECS = int(os.environ.get("MT_WARMUP_SECS", "180"))


@dataclass
class _Job:
    wav: str
    n_samples: int
    max_speakers: int
    future: concurrent.futures.Future


class _Batcher:
    """Single consumer thread: coalesce queued requests, run one coupled session per group.

    A single consumer also serialises the non-reentrant NeMo session (it keeps per-request
    state on the shared models), replacing the old per-request lock.
    """

    def __init__(self, model):
        self.model = model
        self.q = queue.Queue()
        threading.Thread(target=self._run, name="mt-mb", daemon=True).start()

    def submit(self, wav: str, n_samples: int, max_speakers: int) -> concurrent.futures.Future:
        fut = concurrent.futures.Future()
        self.q.put(_Job(wav, n_samples, max_speakers, fut))
        return fut

    def _collect(self) -> list:
        batch = [self.q.get()]
        deadline = time.time() + MB_WINDOW_S
        while len(batch) < MB_CAP:
            rem = deadline - time.time()
            if rem <= 0:
                break
            try:
                batch.append(self.q.get(timeout=rem))
            except queue.Empty:
                break
        return batch

    @staticmethod
    def _groups(jobs: list) -> list:
        """Sessions to run: `max_speakers` binds per session, so key on it; within a key sort by
        duration and split where padding to the longest file would waste > MB_PAD_WASTE."""
        by_key = {}
        for j in jobs:
            by_key.setdefault(j.max_speakers, []).append(j)
        groups = []
        for js in by_key.values():
            js.sort(key=lambda j: j.n_samples)
            cur = []
            for j in js:
                if cur:
                    total = j.n_samples * (len(cur) + 1)  # j is the longest so far
                    waste = 1 - (sum(x.n_samples for x in cur) + j.n_samples) / total
                    if waste > MB_PAD_WASTE:
                        groups.append(cur)
                        cur = []
                cur.append(j)
            groups.append(cur)
        return groups

    def _run(self):
        while True:
            jobs = self._collect()
            for group in self._groups(jobs):
                t0 = time.time()
                try:
                    rows, compute_s, peak_gb, util = self.model._transcribe_batch(
                        [j.wav for j in group], group[0].max_speakers)
                    for j, seglst in zip(group, rows):
                        j.future.set_result((seglst, compute_s, peak_gb, len(group)))
                except Exception as e:  # noqa: BLE001 — fail the whole group, callers see the error
                    logger.exception("batched session failed (n=%d)", len(group))
                    for j in group:
                        j.future.set_exception(e)
                    continue
                longest = max(j.n_samples for j in group)
                pad = 1 - sum(j.n_samples for j in group) / (longest * len(group))
                logger.info("batched %d req (max_spk=%d, %.0fs longest, pad %.0f%%, of %d in window) "
                            "in %.2fs compute %.2fs peak %.1fGB gpu-util %s",
                            len(group), group[0].max_speakers, longest / SAMPLE_RATE, 100 * pad,
                            len(jobs), time.time() - t0, compute_s, peak_gb,
                            f"{util:.0f}%" if util is not None else "n/a")


class _GpuUtilSampler:
    """Mean SM utilisation over the sampled window (NVML via torch; None if unavailable)."""

    def __init__(self, torch):
        self._torch = torch
        self._samples = []
        self._last = 0.0

    def tick(self, every_s: float = 0.5):
        now = time.time()
        if now - self._last < every_s:
            return
        self._last = now
        try:
            self._samples.append(self._torch.cuda.utilization())
        except Exception:  # noqa: BLE001 — pynvml missing or NVML unavailable
            pass

    def mean(self):
        return sum(self._samples) / len(self._samples) if self._samples else None


class Model:
    def __init__(self, **kwargs):
        self._secrets = kwargs.get("secrets", {})
        self._batcher = None

    def load(self):
        import nemo.collections.asr as nemo_asr
        import torch
        from multitalker_transcript_config import MultitalkerTranscriptionConfig
        from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
        from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
            configure_diar_streaming,
            validate_feature_frame_strides,
        )
        from nemo.utils import logging as nemo_logging
        from omegaconf import OmegaConf

        self._torch = torch
        torch.set_float32_matmul_precision("highest")
        asr = nemo_asr.models.ASRModel.restore_from(restore_path=os.environ["MT_ASR_PATH"])
        asr.encoder.set_default_att_context_size(att_context_size=ATT_CTX)
        self._asr = asr.to("cuda").eval()
        scfg = asr.encoder.streaming_cfg
        logger.info("ASR streaming_cfg: %s", scfg)

        diar = SortformerEncLabelModel.restore_from(
            restore_path=os.environ["MT_DIAR_PATH"], map_location="cuda")
        self._diar = diar.to(dtype=torch.bfloat16).eval()

        cfg = OmegaConf.structured(MultitalkerTranscriptionConfig())
        OmegaConf.set_struct(cfg, False)
        cfg.max_num_of_spks = MAX_SPEAKERS
        cfg.att_context_size = ATT_CTX
        cfg.parallel_speaker_strategy = True
        cfg.masked_asr = False
        cfg.cache_gating = True
        cfg.binary_diar_preds = True
        cfg.spkcache_len = SPKCACHE_LEN
        cfg.fifo_len = FIFO_LEN
        cfg.spkcache_update_period = SPKCACHE_UPDATE_PERIOD
        cfg.diar_right_context = 0
        cfg.precision = "bf16"
        cfg.log = False
        cfg.print_time = False
        cfg.colored_text = False
        validate_feature_frame_strides(asr_model=self._asr, diar_model=self._diar)
        # The diarizer's chunk must equal the ASR's per-step output span so activity and
        # audio stay aligned; FeatureStacking pre-encoders need cache-free padded chunks.
        configure_diar_streaming(
            diar_model=self._diar, cfg=cfg,
            output_subsampling_factor=self._asr.encoder.subsampling_factor,
            diar_chunk_len=scfg.valid_out_len + scfg.cache_drop_size)
        cfg.pad_and_drop_preencoded = isinstance(self._diar.encoder.pre_encode, FeatureStacking)
        self._cfg = cfg
        from multitalker_batch_fix import apply as apply_batch_fix
        fix = apply_batch_fix(self._asr, self._diar, asr_rows=ASR_ROWS, diar_rows=DIAR_ROWS,
                              asr_fp32=os.environ.get("MT_ASR_FP32", "0") == "1", diar_fp32=DIAR_FP32,
                              pin_asr=not FAST)
        self._fast = None
        if FAST:
            # Same lever set as the streaming preset, minus the per-step mel (this path feeds whole-file
            # features through NeMo's buffer): the diarizer core replayed from a CUDA graph per sequence
            # length (bit-identical to eager sync mode), NeMo's CudaGraphsStreamingEncoderStep at the
            # pinned row count, shallow hypothesis copies and incremental detokenisation. The ASR dtype is
            # MT_ASR_DTYPE (fp32 weights = the shipped "fp32 ASR, no autocast" numerics; bf16 = the
            # checkpoint reference's). No step-wide autocast: FastPath scopes the dtype per call.
            from mt_fast import FastPath, Flags
            flags = Flags()          # MT_PAD_ROWS (32): rows per ASR call; SigmaS above it runs as slabs
            flags.mel = False
            self._fast = FastPath(torch, self._asr, self._diar, cfg, flags, threading.Lock(), MAX_SPEAKERS)
            if flags.warm_asr:
                self._fast.warm_asr()
        # Output path (packages/mt_turns.py, shared with ../streaming): segments rebuilt from each speaker's
        # token ids and RNNT emission frames. NeMo's own seglst appends `text.strip()` -- the speaker's whole
        # transcript so far -- whenever the new hypothesis text is not a string-prefix extension of the previous
        # one (a punctuation piece removes the space before it), which duplicated ~870 words over eval-30
        # (insertions 2,192 vs 1,322 for the same audio through the streaming preset). MT_TURN_SEGMENTS=0
        # keeps NeMo's seglst for A/B.
        import mt_turns
        from nemo.collections.asr.parts.utils import multispk_transcribe_utils as U
        self._turns_flags = mt_turns.TurnFlags()
        self._turns_on = bool(self._turns_flags.turn_segments and self._fast is not None)
        mt_turns.install(U, on=self._turns_on)
        nemo_logging.setLevel(logging.WARNING)
        logger.info("multitalker T+D ready (max_speakers=%d, pad_and_drop=%s, micro-batch window=%.0fms "
                    "cap=%d pad_waste=%.2f, shape-pin %s, fast=%s)", MAX_SPEAKERS, cfg.pad_and_drop_preencoded,
                    MB_WINDOW_S * 1000, MB_CAP, MB_PAD_WASTE, fix,
                    json.dumps(self._fast.flags.as_dict()) if self._fast is not None else "off")
        self._warmup()
        if self._fast is not None:
            logger.info("warm: encoder graphs=%d diarizer graphs=%d decoder=%s, gpu %.1f GB allocated",
                        self._fast.enc_graphs_captured(), self._fast.diar_graphs_captured(),
                        self._fast.decoder_mode(), torch.cuda.memory_allocated() / 2**30)
        self._batcher = _Batcher(self)

    def _warmup(self):
        """Passes at B=1 (and any MT_MB_WARMUP_SIZES) so kernels, the diarizer's per-length CUDA graphs
        (a WARMUP_SECS session walks FIFO fill -> pop -> cache compression -> every steady length) and the
        decoder's batch-sized state are hot before the first real request."""
        import numpy as np
        import soundfile as sf

        secs = WARMUP_SECS if self._fast is not None else 20
        with tempfile.TemporaryDirectory() as td:
            wavs = []
            for _ in range(max(MB_WARMUP_SIZES, default=1)):
                wav = os.path.join(td, f"{uuid.uuid4().hex}.wav")
                sf.write(wav, (np.random.randn(SAMPLE_RATE * secs) * 0.01).astype(np.float32), SAMPLE_RATE)
                wavs.append(wav)
            for n in sorted(set(MB_WARMUP_SIZES)):
                try:
                    _, compute_s, _, _ = self._transcribe_batch(wavs[:n], 2)
                    logger.info("warmup B=%d done in %.1fs", n, compute_s)
                except Exception as e:  # noqa: BLE001 — warmup is best-effort
                    logger.warning("warmup B=%d failed: %s", n, e)

    def _fetch_audio(self, audio: dict) -> bytes:
        if not isinstance(audio, dict):
            raise ValueError("transcription_input.audio must be an object with 'url' or 'audio_b64'")
        if audio.get("audio_b64"):
            return base64.b64decode(audio["audio_b64"])
        if audio.get("url"):
            with urllib.request.urlopen(audio["url"], timeout=120) as r:
                return r.read()
        raise ValueError("audio requires 'url' or 'audio_b64'")

    def _transcribe_batch(self, wavs: list, max_speakers: int):
        """Run ONE coupled streaming session with len(wavs) rows in lockstep.

        Rows are zero-padded to the longest file so every row presents identical
        `chunk_lengths` at every step (exactly the K=1 geometry; the diarizer's sync-mode
        shape decisions are shared but identical per row). Returns per-row SegLST cut at the
        row's true duration, plus session compute, peak GPU and mean GPU util.
        """
        import numpy as np
        from nemo.collections.asr.parts.preprocessing.segment import get_samples
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

        torch = self._torch
        cfg = copy.deepcopy(self._cfg)
        cfg.audio_file = wavs[0]  # only sizes an unused per-session list in the parallel path
        cfg.batch_size = len(wavs)
        cfg.max_num_of_spks = max_speakers
        util = _GpuUtilSampler(torch)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        buf = CacheAwareStreamingAudioBuffer(
            model=self._asr, online_normalization=cfg.online_normalization,
            pad_and_drop_preencoded=cfg.pad_and_drop_preencoded)
        audios = [get_samples(w) for w in wavs]  # same reader as append_audio_file → K=1 byte-identical
        n_samples = [len(a) for a in audios]
        longest = max(n_samples)
        for audio, n in zip(audios, n_samples):
            if n < longest:
                audio = np.concatenate([audio, np.zeros(longest - n, dtype=audio.dtype)])
            buf.append_audio(audio, stream_id=-1)
        streamer = SpeakerTaggedASR(cfg, self._asr, self._diar)
        drop_size = self._asr.encoder.streaming_cfg.drop_extra_pre_encoded
        # MT_FAST: no step-wide autocast -- mt_fast wraps each NeMo call with the dtype it needs (diarizer
        # + RNNT decoder under bf16 autocast as before, encoder per MT_ASR_DTYPE). The eager A/B path
        # keeps the reference script's whole-step bf16 autocast.
        ac = (torch.amp.autocast("cuda", dtype=torch.bfloat16) if self._fast is None
              else contextlib.nullcontext())
        for step, (chunk_audio, chunk_lengths) in enumerate(iter(buf)):
            drop = 0 if (step == 0 and not cfg.pad_and_drop_preencoded) else drop_size
            with torch.inference_mode(), ac:
                streamer.perform_parallel_streaming_stt_spk(
                    step_num=step, chunk_audio=chunk_audio, chunk_lengths=chunk_lengths,
                    is_buffer_empty=buf.is_buffer_empty(), drop_extra_pre_encoded=drop)
            util.tick()
        rows = [[] for _ in wavs]
        if self._turns_on:
            from mt_turns import SessionTurns
            processed_s = float(streamer._offset_chunk_start_time)
            for i, st in enumerate(streamer.instance_manager.batch_asr_states):
                turns = SessionTurns(self._fast.inc, self._turns_flags, words=False)
                turns.pull(st, processed_s, True)
                rows[i] = [{"session_id": f"row{i}", "speaker": t["speaker"], "start_time": t["start"],
                            "end_time": t["end"], "words": t["text"]} for t in turns.segments(True)]
        else:
            # session_id = the sample's basename stem; use a synthetic per-row id to split rows.
            seglst = streamer.generate_seglst_dicts_from_parallel_streaming(
                samples=[{"audio_filepath": f"row{i}.wav"} for i in range(len(wavs))])
            for s in seglst:
                rows[int(str(s["session_id"])[3:])].append(s)
        for i, n in enumerate(n_samples):  # cut the padded tail: drop late segments, clip the last
            end = n / SAMPLE_RATE
            kept = []
            for s in rows[i]:
                if float(s["start_time"]) >= end:
                    continue
                if float(s["end_time"]) > end:
                    s = {**s, "end_time": end}
                kept.append(s)
            rows[i] = kept
        return rows, time.time() - t0, torch.cuda.max_memory_allocated() / 1e9, util.mean()

    def predict(self, request: dict) -> dict:
        from fastapi import HTTPException

        try:
            return self._predict(request)
        except ValueError as e:  # bad input → 400, not a generic 500
            raise HTTPException(status_code=400, detail=str(e)) from e

    def _predict(self, request: dict) -> dict:
        import soundfile as sf

        ti = request.get("transcription_input")
        if not isinstance(ti, dict):
            raise ValueError("request requires 'transcription_input'")
        try:
            max_speakers = int(ti.get("max_speakers", MAX_SPEAKERS))
        except (TypeError, ValueError) as e:
            raise ValueError("max_speakers must be an integer") from e
        if not 1 <= max_speakers <= MAX_SPEAKERS:
            raise ValueError(f"max_speakers must be in [1, {MAX_SPEAKERS}]")
        raw = self._fetch_audio(ti.get("audio"))

        # The tempdir must outlive the batched session that reads the wav.
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in")
            wav = os.path.join(td, f"{uuid.uuid4().hex}.wav")
            with open(src, "wb") as f:
                f.write(raw)
            try:  # normalize any container/codec to 16 kHz mono wav
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1", "-ar", "16000", wav],
                               check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                raise ValueError(f"could not decode audio: {e.stderr[-300:].decode(errors='replace')}") from e
            n_samples = sf.info(wav).frames
            if n_samples == 0:
                raise ValueError("audio is empty")
            seglst, compute_s, peak_gb, batch_n = self._batcher.submit(
                wav, n_samples, max_speakers).result(timeout=3600)

        segments = sorted(
            ({"speaker": str(s["speaker"]), "start": round(float(s["start_time"]), 3),
              "end": round(float(s["end_time"]), 3), "text": str(s.get("words", "")).strip()}
             for s in seglst if str(s.get("words", "")).strip()),
            key=lambda s: s["start"])
        by_spk = {}
        for s in segments:
            by_spk.setdefault(s["speaker"], []).append(s["text"])
        return {"segments": segments, "speakers": len(by_spk),
                "text_by_speaker": {k: " ".join(v) for k, v in by_spk.items()},
                "compute_s": round(compute_s, 2), "peak_gpu_gb": round(peak_gb, 2), "batch_n": batch_n}
