"""Streaming WebSocket truss for coupled multitalker speaker-attributed transcription (NeMo).

Nemotron 3 Diarization (OpenMDW-1.1) + multitalker-parakeet-streaming-0.6b-v1 (NVIDIA Open Model License). English-only.

Nemotron-3-Diarization (streaming Sortformer) emits per-speaker activity; the
multitalker-parakeet-streaming-0.6b-v1 ASR runs ONE instance per speaker on the same mixed
audio (speaker-kernel injection). Every connection owns its ``SpeakerTaggedASR`` (deploy
mode), audio buffer and diarizer streaming state; per ASR chunk (~1.12 s at att_context
[70, 13]) we call ``perform_parallel_streaming_stt_spk`` and emit the current segments.

The replica holds ONE (ASR, diarizer) pair and serialises every step on one lock. NeMo's
step is not thread-safe (decoder CUDA graphs and batched scratch state are shared). The eager
step was CPU launch-bound (~50 ms for ~11 ms of GPU work); ``packages/mt_fast.py`` dispatches
the same NeMo calls with far fewer launches (per-step mel, bf16 weights, encoder CUDA graphs,
per-length CUDA graphs of the sync-mode diarizer, constant 8 speaker rows, no per-step deepcopy)
for a ~13 ms step — each lever behind an ``MT_*`` flag (see config.yaml and BENCHMARK.md).

Wire protocol (client -> server, text frames):
  {"session_id": "abc", "max_speakers": 8}                      # optional handshake
    (+ optional per-session overrides of the MT_TURN_SEGMENTS / MT_PARTIALS / MT_TURN_WORDS / MT_TURN_OVERLAP
     defaults: "turn_segments": 0/1, "partials": 0/1, "words": 0/1, "overlap": 0/1)
  {"type": "input_audio_buffer.append", "audio": "<b64 pcm16 16 kHz mono>"}  # repeat
  {"type": "input_audio_buffer.commit"}                         # finalize + close
server -> client (replace-style):
  {"type": "transcription", "is_final": false, "session_id": "abc", "processed_s": t,
   "num_speakers": n,
   "segments": [{"speaker": "speaker_0", "start": s, "end": e, "text": "...",
                 "overlap": false, "overlaps_with": [],                  # MT_TURN_OVERLAP: parallel turns
                 "words": [{"w": "so", "start": s, "end": e}, ...]}],   # closed turns (MT_TURN_SEGMENTS)
   "partial": [{"speaker": "speaker_1", "text": "right, first", "start": s, "end": e}]}  # open tails (MT_PARTIALS)
  {"type": "transcription", "is_final": true, ...}              # every word is in "segments"; "partial" is []
  {"type": "error", "error": "..."}                             # then the socket closes
With MT_TURN_SEGMENTS=0 the segments are NeMo's own per-speaker sentences (one running block per speaker,
broken only by that speaker's own 30 s silence) and "partial"/"words" are absent -- see packages/mt_turns.py.
"""

import base64
import copy
import gc
import json
import logging
import os
import threading
import time
import uuid

import numpy as np

import mt_profiling as P  # MT_PROFILE=1 instrumentation (sync-bounded phase timers)

logger = logging.getLogger(__name__)

# ASR encoder attention context [left, right] frames — ~1.12 s streaming advance (model card).
ATT_CTX = [70, 13]
# Diarizer streaming knobs paired with the ASR chunk geometry (model card recommendation).
SPKCACHE_LEN, FIFO_LEN, SPKCACHE_UPDATE_PERIOD = 264, 264, 222
MAX_SPEAKERS = int(os.environ.get("MT_MAX_SPEAKERS", "8"))
ALLOW_CONTROL = os.environ.get("MT_ALLOW_CONTROL", "0") == "1"
WARMUP_SECS = int(os.environ.get("MT_WARMUP_SECS", "180"))
# A finished session's SpeakerTaggedASR / instance-manager / ASR caches / diarizer streaming state
# sit in reference cycles (session <-> streamer <-> instance_manager), so CPython's refcount does
# not free them at scope exit and their CUDA tensors are held until the cyclic collector runs. Under
# call churn (sessions starting and ending continuously, as in production) that accumulates ~1 GB of
# GPU per ended session and OOMs the replica; hour-long streams never showed it because they end
# together at the hour. Collect every MT_GC_EVERY session-ends and return the freed blocks to CUDA.
GC_EVERY = int(os.environ.get("MT_GC_EVERY", "4"))
# Finalize a session that sends nothing for this long (a live 16 kHz stream sends 10 frames/s).
IDLE_TIMEOUT_S = float(os.environ.get("MT_IDLE_TIMEOUT_S", "120"))
# Per-session output-shape / end-of-turn-policy keys a handshake may carry (packages/mt_turns.py).
HANDSHAKE_OPTS = ("turn_segments", "partials", "words", "overlap", "turn_pause_s", "turn_handover_s",
                  "turn_handover_words", "turn_vad_s", "turn_max_open_s", "turn_diar_eot_s", "speech_events", "turn_freeze_closed",
                  "turn_sentence_break")
SR = 16000
HOP = 160      # 10 ms mel hop (samples)
HALF = 256     # n_fft/2: reach of one centered STFT frame
LMARGIN = 3    # mel frames of real left context recomputed at each append seam (legacy path)


def _nemo_build_id() -> str:
    """Resolved NeMo overlay commit (pip records it for VCS installs), for the startup log."""
    try:
        import importlib.metadata as md
        import json as _json
        info = _json.loads(md.distribution("nemo_toolkit").read_text("direct_url.json") or "{}")
        return f"{md.version('nemo_toolkit')}@{info.get('vcs_info', {}).get('commit_id', '?')[:12]}"
    except Exception:  # noqa: BLE001
        return "unknown"


def _parse_max_speakers(value) -> int:
    if value is None:
        return MAX_SPEAKERS
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("max_speakers must be an integer")
    try:
        n = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError("max_speakers must be an integer") from e
    if not 1 <= n <= MAX_SPEAKERS:
        raise ValueError(f"max_speakers must be in [1, {MAX_SPEAKERS}]")
    return n


class Model:
    def __init__(self, **kwargs):
        self._secrets = kwargs.get("secrets", {})
        # Replica-wide step lock: every streaming step on the shared model pair goes through it.
        self._lock = threading.Lock()
        self._live = 0
        self._ended = 0                 # cumulative finished sessions (drives periodic GC)
        self._gc_lock = threading.Lock()

    def load(self):
        import concurrent.futures

        import nemo.collections.asr as nemo_asr
        import torch
        from mt_fast import FastPath, Flags
        from multitalker_transcript_config import MultitalkerTranscriptionConfig
        from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
        from nemo.collections.asr.parts.submodules.subsampling import FeatureStacking
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import (
            configure_diar_streaming,
            validate_feature_frame_strides,
        )
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
        from nemo.utils import logging as nemo_logging
        from omegaconf import OmegaConf

        t_all = time.perf_counter()
        self._torch = torch
        from mt_xsession import XFlags
        self.flags = XFlags.attach(Flags())
        torch.set_float32_matmul_precision("high" if self.flags.tf32 else "highest")
        # The only CPU tensor work is the per-tick window stack + H2D copy; PyTorch's OpenMP workers
        # otherwise spin between ticks on every core and compete with the event loop for the GIL-free
        # CPU it needs (the standalone diarizer fork measured 3+ cores of spin at 320 streams).
        torch.set_num_threads(int(os.environ.get("MT_TORCH_THREADS", "1")))
        # GPU steps are blocking CUDA calls; run them off the event loop so many connections
        # are served concurrently (the lock still serialises the GPU work).
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("MT_WORKERS", "32")))
        # One dedicated thread for the per-step GPU mel (MT_MEL_THREAD): keeps cuFFT/cuDNN off the
        # many connection pool threads so their per-thread CUDA workspaces do not accumulate at high
        # concurrency. Numerically identical to computing mel on the connection thread.
        self._mel_pool = (concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.environ.get("MT_MEL_THREADS", "1")), thread_name_prefix="mel")
            if os.environ.get("MT_MEL_THREAD", "0") == "1" else None)

        # Decoding is left at the checkpoint default (CUDA-graph label-looping RNNT decoder):
        # ~1.5x the stream ceiling of plain PyTorch decoding, and safe because steps are locked.
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
        cfg.deploy_mode = True                 # no manifest/audio_file: live session
        cfg.online_normalization = True        # per-chunk normalisation (frame-local mel)
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
        # One mel preprocessor shared by all sessions (normalize=None, dither=0, pad_to=0 — the
        # buffer's extraction rules) for the per-step mel path.
        probe = CacheAwareStreamingAudioBuffer(model=self._asr, online_normalization=True,
                                               pad_and_drop_preencoded=cfg.pad_and_drop_preencoded)
        self._pre, self._norm_type = probe.preprocessor, probe.model_normalize_type
        if self.flags.mel and not cfg.pad_and_drop_preencoded:
            logger.warning("MT_MEL_PER_STEP needs pad_and_drop_preencoded; using the legacy buffer path")
            self.flags.mel = False
        nemo_logging.setLevel(logging.WARNING)  # NeMo logs one INFO line per streaming step

        self._fast = FastPath(torch, self._asr, self._diar, cfg, self.flags, self._lock, MAX_SPEAKERS)
        self._fast.warm_diar_fn = self._warmup
        # Turn-level segments + live partials (packages/mt_turns.py): output formatting only. The step
        # path records three ints per speaker per step; words/turns are built in message().
        import mt_turns
        self.turns_flags = mt_turns.TurnFlags()
        mt_turns.install(self._fast.U, on=self.turns_flags.turn_segments)
        self._gpu = None
        if P.ENABLED:
            P.wrap_shared(self._asr, self._diar, torch)
            self._gpu = P.GpuSampler(torch)
            self._gpu.start()
            logger.info("MT_PROFILE on: torch threads=%d, profile every %d steps",
                        torch.get_num_threads(), P.PROFILE_EVERY)
        # Cross-session batched stepper (MT_XSESSION, packages/mt_xsession.py): one thread runs every
        # ready session's step in one pass per tick. Started before warm-up so the warm sessions take
        # the same path a connection takes. MT_XSESSION=0 keeps the per-session lock path.
        self._xstep = None
        self._warm_buckets = []
        if self.flags.xsession:
            from mt_xsession import XStepper, warm_asr_buckets
            self._xstep = XStepper(torch, self._fast, self.flags, lambda: self._live,
                                   pre=self._pre, norm_type=self._norm_type)
            self._xstep.start()
            if self.flags.warm_asr:
                self._warm_buckets = warm_asr_buckets(self._fast)
            self._xstep.warm_core(self.flags.xsession_b)
        elif self.flags.warm_asr:
            self._fast.warm_asr()
        self._warmup()
        graphs = self._fast.mark_warm()
        logger.info("multitalker streaming T+D ready in %.0fs (max_speakers=%d, pad_and_drop=%s, "
                    "chunk=%s shift=%s, gpu %.1f GB allocated / %.1f GB reserved, nemo=%s, torch=%s, "
                    "flags=%s, enc_graphs_captured=%d, diar_graphs_captured=%d, dynamo_graphs=%d, decoder=%s, "
                    "xsession=%s, turns=%s)",
                    time.perf_counter() - t_all, MAX_SPEAKERS, cfg.pad_and_drop_preencoded,
                    scfg.chunk_size, scfg.shift_size, torch.cuda.memory_allocated() / 2**30,
                    torch.cuda.memory_reserved() / 2**30, _nemo_build_id(), torch.__version__,
                    json.dumps(self.flags.as_dict()), self._fast.enc_graphs_captured(),
                    self._fast.diar_graphs_captured(), graphs, self._fast.decoder_mode(),
                    json.dumps(self._xstep.summary()) if self._xstep is not None else "off",
                    json.dumps(self.turns_flags.as_dict()))

    def _apply_control(self, control):
        """Runtime lever set: xsession scheduler knobs first (live-safe), the rest through FastPath."""
        from mt_xsession import apply_xcontrol
        rest = apply_xcontrol(self.flags, self._warm_buckets, control)
        buckets = rest.pop("_row_buckets", None)
        if "xdiar" in rest:
            if self._live > 0:
                raise ValueError(f"control refused: {self._live} live session(s)")
            self.flags.xdiar = bool(int(rest.pop("xdiar")))
        if buckets is not None:
            if self._live > 0:
                raise ValueError(f"control refused: {self._live} live session(s)")
            self.flags.xsession_row_buckets = buckets
        if rest:
            return self._fast.apply_control(rest, self._live)
        return self.flags.as_dict()

    def _warmup(self):
        """Drive synthetic sessions through the live path before the first real connection.

        Same code path a connection takes — 100 ms appends, a step + partial whenever a full
        chunk is available, then commit (flush, final step with the buffer empty, SegLST). The
        first session runs WARMUP_SECS so the diarizer state walks FIFO fill -> pop -> speaker
        cache compression (the compiled core and its CUDA graph are recorded on the way); the
        ASR path at k=1..max is warmed separately in ``FastPath.warm_asr`` because white noise
        never activates a speaker. A second short session at a smaller max_speakers exercises
        the per-session config copy.
        """
        rng = np.random.default_rng(0)
        for sid, n_spk, secs in (("warmup-8", MAX_SPEAKERS, WARMUP_SECS), ("warmup-4", min(4, MAX_SPEAKERS), 6)):
            t0 = time.perf_counter()
            try:
                s = _Session(self, sid, n_spk)
                partials = 0
                for _ in range(secs * 10):
                    s.add_audio((rng.standard_normal(SR // 10) * 0.01).astype(np.float32))
                    if s.ready(False) and s.step(flush=False):
                        s.message(False)
                        partials += 1
                s.step(flush=True)
                s.message(True)
                logger.info("%s ok: %d steps, %d partials, max_speakers=%d, %.1fs",
                            sid, s.step_num, partials, n_spk, time.perf_counter() - t0)
            except Exception as e:  # noqa: BLE001 — warmup is best-effort
                logger.exception("%s failed: %s: %s", sid, type(e).__name__, e)

    async def websocket(self, ws):
        import asyncio

        loop = asyncio.get_event_loop()
        sess = None
        # Partials are replace-style, so a partial the socket has not taken yet is superseded by
        # the next one: outbound frames go through a one-slot mailbox drained by a sender task,
        # and receiving never waits on sending. Otherwise a slow reader of the growing partials
        # (hour-long sessions reach ~90 KB per partial) stalls this connection's next audio frames
        # behind ws.send_text and every one of its chunks reaches the stepper late (SWEEP_XSESSION.md).
        # Finals, profile records and errors are never dropped.
        box = {"partial": None, "must": [], "dropped": 0, "closed": False}
        wake = asyncio.Event()

        def text_of(m):
            return m if isinstance(m, str) else json.dumps(m)   # message() pre-serialises the turns path

        async def sender():
            while True:
                await wake.wait()
                wake.clear()
                while box["must"] or box["partial"] is not None:
                    if box["must"]:
                        await ws.send_text(text_of(box["must"].pop(0)))
                    elif box["partial"] is not None:
                        m, box["partial"] = box["partial"], None
                        await ws.send_text(text_of(m))
                if box["closed"]:
                    return

        def post(msg, must=False):
            if must:
                box["must"].append(msg)
            else:
                if box["partial"] is not None:
                    box["dropped"] += 1
                box["partial"] = msg
            wake.set()

        sender_task = loop.create_task(sender())

        async def close_with(msg):
            box["partial"] = None
            post(msg, must=True)
            box["closed"] = True
            wake.set()
            await sender_task

        try:
            while True:
                try:
                    frame = await asyncio.wait_for(ws.receive_text(), timeout=IDLE_TIMEOUT_S)
                except asyncio.TimeoutError:
                    # A client that vanished without a close frame (crash, network drop) otherwise lingers
                    # behind the gateway for ping_timeout_seconds, holding its slot and a concurrency unit:
                    # 13 of 160 k6 sessions killed mid-stream were still "live" minutes later. Finalize.
                    logger.info("ws session idle for %.0fs: finalizing", IDLE_TIMEOUT_S)
                    if sess is not None:
                        try:
                            await sess.step_async(True)
                            msg = await loop.run_in_executor(self._pool, sess.message, True)
                            await close_with(msg)
                        except Exception:  # noqa: BLE001 - the socket is most likely gone
                            pass
                    return
                try:
                    data = json.loads(frame)
                    if not isinstance(data, dict):
                        raise ValueError("frame must be a JSON object")
                    mtype = data.get("type")
                    handshake = None
                    if sess is None:
                        handshake = (str(data.get("session_id") or uuid.uuid4().hex),
                                     _parse_max_speakers(data.get("max_speakers")),
                                     data.get("mel"),
                                     {k: data[k] for k in HANDSHAKE_OPTS if k in data})
                        prof_opts = {"sync": bool(data.get("prof_sync", True)),
                                     "threads": data.get("prof_threads")}
                        control = data.get("control")
                    pcm = None
                    if mtype == "input_audio_buffer.append" and data.get("audio"):
                        pcm = np.frombuffer(base64.b64decode(data["audio"], validate=True), dtype=np.int16)
                except (ValueError, TypeError) as e:  # malformed client frame: tell them, then close
                    logger.info("rejected client frame: %s", e)
                    await close_with({"type": "error", "error": str(e)})
                    return
                if sess is None:
                    if data.get("stats"):
                        # Read-only replica snapshot for smoke tests / operators: live session count,
                        # GPU memory, stepper and detokeniser counters. Closes the socket; no session.
                        await close_with({"type": "stats", "live": self._live, "ended": self._ended,
                                          "gpu_alloc_gb": round(self._torch.cuda.memory_allocated() / 2**30, 2),
                                          "gpu_reserved_gb": round(self._torch.cuda.memory_reserved() / 2**30, 2),
                                          "xsession": self._xstep.summary() if self._xstep is not None else None,
                                          "detok": self._fast.inc.summary(), "flags": self.flags.as_dict()})
                        return
                    if control:
                        if not ALLOW_CONTROL:
                            await ws.send_text(json.dumps({"type": "error", "error": "control disabled"}))
                            return
                        try:
                            applied = await loop.run_in_executor(self._pool, self._apply_control, control)
                        except Exception as e:  # noqa: BLE001
                            logger.exception("control failed")
                            await ws.send_text(json.dumps({"type": "error", "error": f"control: {e}"}))
                            return
                        logger.info("control applied: %s", json.dumps(applied))
                        await ws.send_text(json.dumps({"type": "control", "flags": applied}))
                    sess = await loop.run_in_executor(self._pool, _Session, self, *handshake)
                    self._live += 1
                    if P.ENABLED:
                        sess.prof_sync = prof_opts["sync"]
                        # OpenMP's thread count is per-thread: apply it on the pool thread that
                        # runs the step (see _run_step), not here on the event loop.
                        sess.prof_threads = int(prof_opts["threads"] or 0)
                    if mtype not in ("input_audio_buffer.append", "input_audio_buffer.commit"):
                        continue  # bare handshake frame
                if pcm is not None:
                    samples = pcm.astype(np.float32) / 32768.0
                    sess.add_audio(samples)
                    if sess.vad is not None:
                        # End-of-turn policy between steps: the handler's energy VAD sees the pause up to
                        # 1.12 s before the next step does. Closed tails go out as a regular replace-style
                        # frame (never dropped: it carries new finals); speech onsets as a tiny frame.
                        onset = sess.vad.update(samples)
                        if onset and sess.speech_events:
                            post({"type": "speech", "event": "start", "session_id": sess.sid,
                                  "t": round(sess.vad.now_s, 2)}, must=True)
                        if not sess.vad.in_speech and sess.turns is not None and sess.step_num > 0:
                            if sess.turns.close_silent(sess.vad.silence_start_s, sess.vad.now_s, sess.processed_s()):
                                post(sess.message(False), must=True)
                    if not sess.ready(False):
                        continue      # no full chunk yet: no thread hop for this 100 ms frame
                    ran = await sess.step_async(False)
                    if ran:
                        # The turns-path message is a cached-fragment string join (~0.1 ms): build it
                        # here; the dict/profile paths keep the pool.
                        msg = (sess.message(False) if sess.windowed and not P.ENABLED
                               else await loop.run_in_executor(self._pool, sess.message, False))
                        if sess.pending_profile is not None:
                            post(sess.pending_profile, must=True)
                            sess.pending_profile = None
                        post(msg)
                elif mtype == "input_audio_buffer.commit":
                    await sess.step_async(True)
                    msg = await loop.run_in_executor(self._pool, sess.message, True)
                    if sess.pending_profile is not None:
                        post(sess.pending_profile, must=True)
                        sess.pending_profile = None
                    await close_with(msg)          # the final supersedes any unsent partial
                    return
        except Exception as e:  # noqa: BLE001
            # Client disconnects land here and are routine; anything else is a server fault
            # (e.g. CUDA) and must be visible in logs, with a best-effort error frame.
            if "Disconnect" in type(e).__name__ or "ConnectionClosed" in type(e).__name__:
                logger.info("ws session ended: %s", type(e).__name__)
                return
            logger.exception("ws session failed")
            try:
                await close_with({"type": "error", "error": f"internal error: {type(e).__name__}"})
            except Exception:  # noqa: BLE001 — socket already gone
                pass
        finally:
            if not sender_task.done():
                sender_task.cancel()
            if sess is not None:
                self._live = max(0, self._live - 1)
                tb = sess.turns.tb if sess.turns is not None else None
                logger.info("session %s: %d steps, %.1fs audio, %d live, mel=%s%s, partials_coalesced=%d, "
                            "turns=%s vad_closed=%s diar_closed=%s frozen=%s, "
                            "gpu_alloc_gb=%.2f gpu_reserved_gb=%.2f, detok=%s, xsession=%s",
                            sess.sid, sess.step_num, sess.audio_s(), self._live, sess.use_mel,
                            sess.verify_summary(), box["dropped"],
                            len(tb.closed) if tb else None, tb.n_vad_closed if tb else None, tb.n_diar_closed if tb else None,
                            tb.n_frozen if tb else None,
                            self._torch.cuda.memory_allocated() / 2**30, self._torch.cuda.memory_reserved() / 2**30,
                            json.dumps(self._fast.inc.summary()),
                            json.dumps(self._xstep.summary()) if self._xstep is not None else "off")
                # Break the session's reference cycles now (see GC_EVERY) so its CUDA tensors are
                # unreferenced, then collect+empty periodically off the event loop.
                sess.teardown()
                self._ended += 1
                if self._ended % GC_EVERY == 0:
                    await loop.run_in_executor(self._pool, self._reclaim)

    def _reclaim(self):
        """Run the cyclic collector and return the freed CUDA blocks to the driver. Cheap (~ms) and
        serialised so overlapping session-ends do not stack collects."""
        if not self._gc_lock.acquire(blocking=False):
            return
        try:
            gc.collect()
            try:
                self._torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._gc_lock.release()


def _spk(x):
    x = str(x)
    return x if x.startswith("speaker") else f"speaker_{x}"


class _Session:
    """Per-connection state: audio buffer, SpeakerTaggedASR (ASR caches + diar state)."""

    def __init__(self, model: "Model", sid: str, max_speakers: int, mel=None, opts=None):
        from mt_fast import MelChunker
        from mt_turns import EnergyVAD, SessionTurns
        from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
        from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

        self.m = model
        self.torch = model._torch
        self.sid = sid.split(".")[0]
        # Output shape for this session: env defaults, handshake overrides (0/1). Partials are the open
        # tail of the turn builder, so they need turn segments.
        tf, opts = model.turns_flags, opts or {}

        def opt(k, default):
            v = opts.get(k)
            return default if v is None else bool(int(v))

        self.turn_segments = opt("turn_segments", tf.turn_segments)
        self.partials_on = self.turn_segments and opt("partials", tf.partials)
        tf = tf.override(opts)       # end-of-turn policy knobs (pause / hand-over / VAD / max-open) per session
        self.turns = (SessionTurns(model._fast.inc, tf, words=opt("words", tf.words), overlap=opt("overlap", tf.overlap))
                      if self.turn_segments else None)
        self.speech_events = tf.speech_events
        # Handler-side energy VAD: drives between-step end of turn (tf.vad_s) and speech-onset frames.
        self.vad = EnergyVAD() if (tf.vad_s > 0 or tf.speech_events) else None
        self.last_active = []        # speaker indices the diarizer marked active in the latest step
        self.tail_inactive = None    # per speaker, trailing OFF frames of the latest chunk (xsession)
        self.cfg = copy.deepcopy(model._cfg)
        self.cfg.max_num_of_spks = max_speakers
        self.max_speakers = max_speakers
        self.offset = 0.0            # xsession: this session's chunk start time (NeMo shares one float)
        self.slot = None             # xsession MT_XDIAR: row of the diarizer slot tables
        self.lock = model._lock
        self.xstep = model._xstep if model.flags.xsession else None
        self.asr, self.diar = model._asr, model._diar
        self.use_mel = bool(int(mel)) if mel is not None else model.flags.mel
        self.verify = self.use_mel and model.flags.mel_verify
        with self.lock:  # construction reads shared module state
            self.buf = None
            if not self.use_mel or self.verify:
                self.buf = CacheAwareStreamingAudioBuffer(
                    model=self.asr, online_normalization=self.cfg.online_normalization,
                    pad_and_drop_preencoded=self.cfg.pad_and_drop_preencoded)
            self.streamer = SpeakerTaggedASR(self.cfg, self.asr, self.diar)
        self.scfg = self.asr.encoder.streaming_cfg
        self.mel = None
        if self.use_mel:
            self.mel = MelChunker(self.torch, model._pre, model._norm_type, "cuda",
                                  self.scfg.chunk_size[1], self.scfg.pre_encode_cache_size[1])
        # Profiling state (MT_PROFILE=1): per-step records shipped in the next message.
        self.prof_sync = True
        self.prof_threads = 0
        self.recs = []
        self.pending_profile = None
        self._mel_s = 0.0
        self._gpu_t = time.time()
        if P.ENABLED and self.xstep is None:
            P.wrap_session(self.streamer, self.torch)
        if self.xstep is not None:
            self.xstep.admit(self)     # per-row step-0 reset; the tick never resets a batch
        self.raw = np.zeros(0, dtype=np.float32)   # legacy path / verify: whole-session raw
        self.feat_appended = 0   # mel frames already appended to the buffer
        self.stream_id = -1      # -1 creates stream 0 on first append, then extend it
        self.step_num = 0
        self.t0 = time.time()
        self.vstats = {"steps": 0, "equal": 0, "max_abs": 0.0}

    def teardown(self):
        """Drop the heavy per-session state so its CUDA tensors are collectable. The streamer,
        instance manager, ASR caches and diarizer streaming state form reference cycles, so this
        just nulls the session's own references; the model's periodic ``_reclaim`` runs the cyclic
        collector. Also clears the last speaker-target tensors the shared ASR model still points at."""
        try:
            if self.xstep is not None:
                self.xstep.release(self)       # free the diarizer slot (MT_XDIAR)
            self.streamer = None
            self.turns = None
            self.buf = None
            self.mel = None
            self.raw = None
        except Exception:  # noqa: BLE001
            pass

    def audio_s(self):
        return (self.mel.total if self.mel is not None else len(self.raw)) / SR

    def processed_s(self):
        """Audio time the last completed step covered (0 before the first step)."""
        return float(self.streamer._offset_chunk_start_time) if self.step_num > 0 else 0.0

    def verify_summary(self):
        if not self.verify:
            return ""
        return f", mel_verify steps={self.vstats['steps']} equal={self.vstats['equal']} max_abs={self.vstats['max_abs']:.3e}"

    def add_audio(self, samples: np.ndarray):
        if self.mel is not None:
            self.mel.add(samples)
        if self.buf is not None:
            self.raw = np.concatenate([self.raw, samples])

    def ready(self, flush: bool) -> bool:
        """A full chunk is available (cheap, no GPU; the legacy buffer path always says yes)."""
        return self.mel.has_chunk(flush) if (self.use_mel and self.mel is not None) else True

    # ------------------------------------------------------------------ legacy (per-append) mel path
    def _append_new_feats(self, final: bool):
        """Append mel for newly complete frames (centered STFT: frame i spans i*HOP +- NFFT/2).

        Frames whose window runs past the audio end are held back until more audio arrives,
        so seams match whole-file features exactly; on final, the tail is emitted as-is.
        """
        L = len(self.raw)
        n = L // HOP + 1 if final else max(0, (L - HALF) // HOP + 1)
        start = self.feat_appended
        if n <= start:
            return
        a = max(0, start - LMARGIN)
        seg = self.raw[a * HOP: min(L, (n - 1) * HOP + HALF)]
        t_mel = time.perf_counter()
        sig = self.torch.as_tensor(seg, device="cuda")[None, :]
        ln = self.torch.tensor([sig.shape[1]], device="cuda")
        with self.torch.inference_mode():
            feats, _ = self.buf.preprocessor(input_signal=sig, length=ln)   # [1, F, T]
        feats = feats[:, :, start - a: start - a + (n - start)]
        if feats.shape[-1] == 0:
            return
        self.buf.append_processed_signal(feats, stream_id=self.stream_id)
        if P.ENABLED and self.prof_sync:
            self.torch.cuda.synchronize()
        if not self.use_mel:
            self._mel_s += time.perf_counter() - t_mel
        self.stream_id = 0
        self.feat_appended = start + feats.shape[-1]

    def _chunk_frames_needed(self):
        cs = self.scfg.chunk_size
        if isinstance(cs, list):
            if self.buf.buffer_idx == 0:
                return cs[1] if self.cfg.pad_and_drop_preencoded else cs[0]
            return cs[1]
        return cs

    def _legacy_pad_for_flush(self):
        ss = self.scfg.shift_size
        shift = ss[1] if isinstance(ss, list) else ss
        idx = int(self.buf.buffer_idx) if self.buf.buffer is not None else 0
        pending = len(self.raw) // HOP + 1 - idx      # frames left after padding-free finish
        k = (-pending) % shift
        if k:
            self.raw = np.concatenate([self.raw, np.zeros(k * HOP, dtype=np.float32)])

    def _legacy_next_chunk(self, flush):
        """One chunk from the NeMo buffer, or None when no full chunk is available."""
        if self.buf.buffer is None:
            return None
        avail = int(self.buf.buffer.size(-1) - self.buf.buffer_idx)
        if avail <= 0 or (avail < self._chunk_frames_needed() and not flush):
            return None
        try:
            return next(iter(self.buf))
        except StopIteration:  # tail shorter than the pre-encoder needs
            return None

    # ------------------------------------------------------------------ step driver
    def _run_step(self, chunk_audio, chunk_lengths, last):
        if self.xstep is not None:
            # Hand ready features to the batched stepper (legacy / verify path) and wait for the tick
            # that carries them; the stepper appends this step's profile record itself.
            self.xstep.submit(self, None, None, last, chunk=chunk_audio).result()
            self.step_num += 1
            return
        drop = 0 if (self.step_num == 0 and not self.cfg.pad_and_drop_preencoded) \
            else self.scfg.drop_extra_pre_encoded

        def call():
            # No step-wide autocast: mt_fast wraps each NeMo call with the dtype context it needs
            # (diarizer + RNNT decoder bf16 autocast as before; encoder per MT_ASR_DTYPE).
            with self.torch.inference_mode():
                self.streamer.perform_parallel_streaming_stt_spk(
                    step_num=self.step_num, chunk_audio=chunk_audio, chunk_lengths=chunk_lengths,
                    is_buffer_empty=last, drop_extra_pre_encoded=drop)

        if not P.ENABLED:
            with self.lock:
                call()
            self.step_num += 1
            return
        torch = self.torch
        if self.prof_threads and torch.get_num_threads() != self.prof_threads:
            torch.set_num_threads(self.prof_threads)  # this pool thread runs the step
            logger.info("torch.set_num_threads(%d) on %s", self.prof_threads, threading.current_thread().name)
        rec = {"sid": self.sid, "step": self.step_num, "last": last, "_sync": self.prof_sync,
               "mel": self._mel_s, "live": self.m._live, "t": round(time.time(), 3)}
        self._mel_s = 0.0
        profile_this = self.step_num > 0 and self.step_num % P.PROFILE_EVERY == 0
        t_wait = time.perf_counter()
        self.lock.acquire()
        rec["wait"] = time.perf_counter() - t_wait
        try:
            P.set_current(rec)
            if profile_this:
                rec["_profiling"] = True
                t0 = time.perf_counter()
                tables, counters = P.run_profiled(call, torch)
                rec["step_wall"] = time.perf_counter() - t0
                self.pending_profile = {"type": "profile", "session_id": self.sid, "step": self.step_num,
                                        "k_active": rec.get("k_active"), "tables": tables,
                                        "counters": counters}
                logger.info("MTPROF profile step=%d k=%s %s", self.step_num, rec.get("k_active"),
                            json.dumps(counters))
            else:
                if self.prof_sync:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                call()
                if self.prof_sync:
                    torch.cuda.synchronize()
                rec["step_wall"] = time.perf_counter() - t0
        finally:
            P.set_current(None)
            self.lock.release()
        states = getattr(self.streamer.instance_manager, "batch_asr_states", None)
        rec["k_known"] = len(states[0].get_speakers()) if states else 0
        rec["threads"] = torch.get_num_threads()
        out = {k: (round(v * 1000, 3) if isinstance(v, float) and not k.startswith("_") and k != "t" else v)
               for k, v in rec.items() if not k.startswith("_") or k == "_profiling"}
        self.recs.append(out)
        if self.step_num % P.LOG_EVERY == 0:
            logger.info("MTPROF %s", json.dumps(out))
        self.step_num += 1

    def step(self, flush: bool) -> bool:
        """Process every FULL chunk available; on flush, zero-pad so the tail is a full last chunk."""
        if self.use_mel:
            return self._step_mel(flush)
        if flush:
            self._legacy_pad_for_flush()
        self._append_new_feats(final=flush)
        ran = False
        while True:
            got = self._legacy_next_chunk(flush)
            if got is None:
                break
            chunk_audio, chunk_lengths = got
            last = flush and self.buf.is_buffer_empty()
            self._run_step(chunk_audio, chunk_lengths, last)
            ran = True
        return ran

    @property
    def windowed(self) -> bool:
        """MT_XMEL on the stepper path: hand raw windows to the stepper, which computes the mel."""
        return self.xstep is not None and self.use_mel and self.m.flags.xmel and not self.verify

    async def step_async(self, flush: bool) -> bool:
        """Connection-side step: slice the next chunk's raw window (numpy only) and await the tick
        that carries it. No pool thread parks; the stepper resolves the future from its thread."""
        import asyncio
        if not self.windowed:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.m._pool, self.step, flush)
        if flush:
            self.mel.pad_for_flush()
        loop = asyncio.get_event_loop()
        ran = False
        while self.mel.has_chunk(flush):
            seg, geom = self.mel.next_window(flush)
            last = flush and not self.mel.has_chunk(flush)
            await self.xstep.submit_async(self, seg, geom, last, loop)
            self.step_num += 1
            ran = True
        return ran

    def _step_mel(self, flush: bool) -> bool:
        if flush:
            self.mel.pad_for_flush()
            if self.verify:
                self._legacy_pad_for_flush()
        if self.verify:
            self._append_new_feats(final=flush)
        ran = False
        while self.mel.has_chunk(flush):
            if self.windowed:      # blocking form of step_async (warm-up sessions)
                seg, geom = self.mel.next_window(flush)
                last = flush and not self.mel.has_chunk(flush)
                self.xstep.submit(self, seg, geom, last).result()
                self.step_num += 1
                ran = True
                continue
            t_mel = time.perf_counter()
            # MT_MEL_THREAD: run the per-step GPU mel on ONE shared thread instead of on this
            # connection's pool thread. Numerically identical (same next_chunk, same per-session
            # order), but only that one thread ever touches cuFFT/cuDNN, so N connections do not each
            # cache a preprocessor plan / workspace — the per-thread context blow-up that let a
            # high-MT_WORKERS replica OOM at ~160 concurrent streams. Lets MT_WORKERS scale with
            # concurrency (each parks a thread waiting for its tick) without GPU memory growing with it.
            mp = self.m._mel_pool
            chunk_audio, chunk_lengths = (mp.submit(self.mel.next_chunk, flush).result() if mp is not None
                                          else self.mel.next_chunk(flush))   # outside the lock
            if P.ENABLED and self.prof_sync:
                self.torch.cuda.synchronize()
            self._mel_s += time.perf_counter() - t_mel
            if self.verify:
                self._verify_chunk(chunk_audio, chunk_lengths, flush)
            last = flush and not self.mel.has_chunk(flush)
            self._run_step(chunk_audio, chunk_lengths, last)
            ran = True
        return ran

    def _verify_chunk(self, chunk_audio, chunk_lengths, flush):
        """MT_MEL_VERIFY: compare the per-step chunk with the legacy buffer's chunk for this step."""
        got = self._legacy_next_chunk(flush)
        self.vstats["steps"] += 1
        if got is None:
            logger.warning("mel_verify step %d: legacy buffer had no chunk", self.step_num)
            return
        ref, ref_len = got
        eq = bool(self.torch.equal(ref, chunk_audio)) and bool(self.torch.equal(ref_len, chunk_lengths))
        diff = float((ref - chunk_audio).abs().max()) if ref.shape == chunk_audio.shape else float("inf")
        self.vstats["equal"] += int(eq)
        self.vstats["max_abs"] = max(self.vstats["max_abs"], diff)
        if not eq:
            logger.info("mel_verify step %d: equal=%s max_abs=%.3e shapes %s/%s len %s/%s", self.step_num, eq,
                        diff, tuple(ref.shape), tuple(chunk_audio.shape), ref_len.tolist(), chunk_lengths.tolist())

    def _segments(self):
        states = getattr(self.streamer.instance_manager, "batch_asr_states", None)
        if not states:
            return []
        out = []
        for seg in states[0].seglsts:
            text = str(seg.get("words", "")).strip()
            if text:
                out.append({"speaker": _spk(seg["speaker"]), "start": round(float(seg["start_time"]), 2),
                            "end": round(float(seg["end_time"]), 2), "text": text})
        return sorted(out, key=lambda s: s["start"])

    def message(self, is_final: bool):
        """The outbound frame: a JSON string on the turns path (segments are cached per-turn fragments,
        so a replace-style partial costs a string join, not a re-serialisation of every word), a dict
        elsewhere (NeMo-seglst output, MT_PROFILE records)."""
        if self.turns is not None and not P.ENABLED:
            return self._message_str(is_final)
        t0 = time.perf_counter()
        msg = self._message(is_final)
        if P.ENABLED:
            recs, self.recs = self.recs, []
            if recs:
                recs[-1]["msg"] = round((time.perf_counter() - t0) * 1000, 3)
            msg["prof"] = recs
            if self.m._gpu is not None:
                samples = self.m._gpu.since(self._gpu_t)
                if samples:
                    self._gpu_t = samples[-1][0]
                msg["gpu"] = samples
        return msg

    def _message_str(self, is_final: bool):
        from mt_turns import dumps
        states = getattr(self.streamer.instance_manager, "batch_asr_states", None)
        processed_s = round(float(self.streamer._offset_chunk_start_time), 2)
        if states:
            self.turns.pull(states[0], processed_s, is_final)
            if not is_final:
                self.turns.close_inactive(self.tail_inactive, processed_s)
            if is_final and self.m.turns_flags.verify:
                self.turns.verify(states[0], self.sid)
        # Byte-for-byte json.dumps' default layout (", " / ": "), which consumers pattern-match on.
        parts = ['{"type": "transcription", "is_final": ', "true" if is_final else "false",
                 ', "session_id": ', dumps(self.sid), ', "processed_s": ', repr(processed_s),
                 ', "num_speakers": ', str(self.turns.num_speakers()),
                 ', "segments": ', self.turns.segments_json(is_final)]
        if self.partials_on:
            parts += [', "partial": ', "[]" if is_final else dumps(self.turns.partial())]
        if self.speech_events:   # who the diarizer heard in the latest step, before any word is decoded
            parts += [', "active": ', dumps([_spk(k) for k in self.last_active])]
        parts.append("}")
        return "".join(parts)

    def _message(self, is_final: bool):
        states = getattr(self.streamer.instance_manager, "batch_asr_states", None)
        processed_s = round(float(self.streamer._offset_chunk_start_time), 2)
        if self.turns is not None:
            if states:
                self.turns.pull(states[0], processed_s, is_final)
                if not is_final:
                    self.turns.close_inactive(self.tail_inactive, processed_s)
                if is_final and self.m.turns_flags.verify:
                    self.turns.verify(states[0], self.sid)
            msg = {"type": "transcription", "is_final": is_final, "session_id": self.sid,
                   "processed_s": processed_s, "num_speakers": self.turns.num_speakers(),
                   "segments": self.turns.segments(is_final)}
            if self.partials_on:
                msg["partial"] = [] if is_final else self.turns.partial()
            return msg
        if states:
            states[0].__dict__.pop("_mt_turn_events", None)   # recorded for other sessions' sake; unused here
        if is_final:
            segl = self.streamer.generate_seglst_dicts_from_parallel_streaming(
                samples=[{"audio_filepath": "streaming_session.wav"}])
            segs = sorted([{"speaker": _spk(s["speaker"]), "start": round(float(s["start_time"]), 2),
                            "end": round(float(s["end_time"]), 2), "text": s["words"]} for s in segl
                           if str(s.get("words", "")).strip()], key=lambda s: s["start"])
        else:
            segs = self._segments()
        return {"type": "transcription", "is_final": is_final, "session_id": self.sid,
                "processed_s": processed_s,
                "num_speakers": len({s["speaker"] for s in segs}), "segments": segs}
