"""Control group: identical to faster-whisper-v3 EXCEPT transcribe params
are passed explicitly (the old way) instead of relying on faster-whisper defaults.

Deploy both v3 and v3-control, run the same test with/without --per-vad-chunk,
and compare to isolate which change affects segmentation:
  - If v3 == v3-control (same params): param simplification doesn't matter
  - If --per-vad-chunk makes the difference on both: per_vad_chunk is the fix
"""
import base64
import logging
import os
from tempfile import NamedTemporaryFile
from typing import Dict

import numpy as np
import requests
import ctranslate2
import faster_whisper
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

logger = logging.getLogger(__name__)

_NO_SPEECH_COEFF = -3.7054642362321193
_AVG_LOGPROB_COEFF = 0.04797854548053526
_BIAS = 1.0332079730313122
_COMPRESSION_RATIO_THRESHOLD = 2.4

MODEL_CACHE_PATH = "/app/model_cache/faster-whisper-large-v3"
SAMPLE_RATE = 16000
TEMPERATURE_FALLBACK = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _is_segment_valid(
    no_speech_prob: float, avg_logprob: float, compression_ratio: float
) -> bool:
    speech_score = (
        _NO_SPEECH_COEFF * no_speech_prob
        + _AVG_LOGPROB_COEFF * avg_logprob
        + _BIAS
    )
    return speech_score > 0 and compression_ratio < _COMPRESSION_RATIO_THRESHOLD


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._lazy_data_resolver = kwargs["lazy_data_resolver"]
        self._model = None

    def load(self):
        self._lazy_data_resolver.block_until_download_complete()

        model_repo = os.getenv("WHISPER_MODEL_REPO") or None
        if model_repo:
            from huggingface_hub import snapshot_download

            try:
                hf_token = self._secrets.get("hf_access_token")
            except Exception:
                hf_token = None
            logger.info(f"Downloading fine-tuned model from HF: {model_repo}")
            model_path = snapshot_download(model_repo, token=hf_token)
            logger.info(f"Model downloaded to: {model_path}")
        else:
            model_path = MODEL_CACHE_PATH
            logger.info(f"Using bundled model: {model_path}")

        self._model = WhisperModel(model_path, device="cuda", compute_type="float16")
        logger.info("WhisperModel loaded")

    def preprocess(self, request: Dict) -> Dict:
        whisper_input = request.get("whisper_input") or {}
        audio_obj = whisper_input.get("audio") or {}

        audio_base64 = request.get("audio")
        url = audio_obj.get("url") or request.get("url")

        if audio_base64 and url:
            return {
                "error": "Only a base64 audio file OR a URL can be passed to the API, not both."
            }
        if not audio_base64 and not url:
            return {
                "error": "Please provide either an audio file in base64 string format or a URL."
            }

        if audio_base64:
            binary_data = base64.b64decode(audio_base64)
        else:
            resp = requests.get(url)
            binary_data = resp.content

        passthrough = {
            "whisper_params": request.get("whisper_params"),
            "vad_parameters": request.get("vad_parameters"),
            "debug": request.get("debug"),
            "disable_segment_filter": request.get("disable_segment_filter"),
            "per_vad_chunk": request.get("per_vad_chunk"),
        }
        passthrough = {k: v for k, v in passthrough.items() if v is not None}
        return {"data": binary_data, **passthrough}

    def predict(self, request: Dict) -> Dict:
        if request.get("error"):
            return request

        audio_data = request.get("data")

        with NamedTemporaryFile(suffix=".wav") as fp:
            fp.write(audio_data)
            fp.flush()
            audio = decode_audio(fp.name, sampling_rate=SAMPLE_RATE)

        duration_s = len(audio) / SAMPLE_RATE

        req_params = request.get("whisper_params") or {}
        debug_mode = _as_bool(request.get("debug"), default=False)
        disable_segment_filter = _as_bool(
            request.get("disable_segment_filter"), default=False
        )
        per_vad_chunk = _as_bool(request.get("per_vad_chunk"), default=False)

        # ── CONTROL: always pass ALL params explicitly (the old way) ─────
        transcribe_kwargs = {
            "audio": audio,
            "beam_size": 1,
            "vad_filter": True,
            "word_timestamps": True,
            "temperature": TEMPERATURE_FALLBACK,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": True,
            "language": None,
        }

        # ── Run VAD upfront for diagnostics and per-chunk mode ───────────
        vad_opts = VadOptions()
        vad_chunks = get_speech_timestamps(audio, vad_options=vad_opts)
        total_speech_s = sum(
            (c["end"] - c["start"]) / SAMPLE_RATE for c in vad_chunks
        )
        logger.info(
            f"VAD: {len(vad_chunks)} speech chunks, "
            f"{total_speech_s:.1f}s speech / {duration_s:.1f}s total "
            f"({total_speech_s / duration_s * 100:.0f}%)"
        )

        # ── Transcribe ───────────────────────────────────────────────────
        if per_vad_chunk:
            all_segments, info = self._transcribe_per_vad_chunk(
                audio, vad_chunks, transcribe_kwargs
            )
        else:
            segments, info = self._model.transcribe(**transcribe_kwargs)
            all_segments = list(segments)

        # ── Filter and build response segments ───────────────────────────
        result_segments = []
        dropped_by_speech_score = 0
        dropped_by_compression = 0
        raw_segments_debug = []
        dropped_segments_debug = []

        for seg in all_segments:
            speech_score = (
                _NO_SPEECH_COEFF * seg.no_speech_prob
                + _AVG_LOGPROB_COEFF * seg.avg_logprob
                + _BIAS
            )
            seg_debug = {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "no_speech_prob": seg.no_speech_prob,
                "avg_logprob": seg.avg_logprob,
                "compression_ratio": seg.compression_ratio,
                "speech_score": speech_score,
            }
            raw_segments_debug.append(seg_debug)

            valid = _is_segment_valid(
                seg.no_speech_prob, seg.avg_logprob, seg.compression_ratio
            )
            if not disable_segment_filter and not valid:
                if speech_score <= 0:
                    dropped_by_speech_score += 1
                if seg.compression_ratio >= _COMPRESSION_RATIO_THRESHOLD:
                    dropped_by_compression += 1
                dropped_segments_debug.append(seg_debug)
                continue

            result_segments.append(
                {
                    "text": seg.text,
                    "start_time": seg.start,
                    "end_time": seg.end,
                    "no_speech_prob": seg.no_speech_prob,
                    "avg_logprob": seg.avg_logprob,
                    "compression_ratio": seg.compression_ratio,
                    "words": [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "probability": w.probability,
                        }
                        for w in (seg.words or [])
                    ],
                }
            )

        logger.info(
            f"Transcribed {duration_s:.1f}s audio, "
            f"produced {len(result_segments)} segments "
            f"(raw={len(all_segments)}, filter_disabled={disable_segment_filter}, "
            f"per_vad_chunk={per_vad_chunk})"
        )

        response = {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": duration_s,
            "segments": result_segments,
        }
        if debug_mode:
            vad_windows = [
                {
                    "start_s": c["start"] / SAMPLE_RATE,
                    "end_s": c["end"] / SAMPLE_RATE,
                    "duration_s": (c["end"] - c["start"]) / SAMPLE_RATE,
                }
                for c in vad_chunks
            ]

            response["debug"] = {
                "variant": "CONTROL (explicit params)",
                "faster_whisper_version": faster_whisper.__version__,
                "ctranslate2_version": ctranslate2.__version__,
                "effective_params": {
                    k: v for k, v in transcribe_kwargs.items() if k != "audio"
                },
                "disable_segment_filter": disable_segment_filter,
                "per_vad_chunk": per_vad_chunk,
                "segment_counts": {
                    "raw_count": len(all_segments),
                    "kept_count": len(result_segments),
                    "dropped_count": len(all_segments) - len(result_segments),
                    "dropped_by_speech_score": dropped_by_speech_score,
                    "dropped_by_compression": dropped_by_compression,
                },
                "vad_summary": {
                    "window_count": len(vad_chunks),
                    "total_speech_s": round(total_speech_s, 2),
                    "audio_duration_s": round(duration_s, 2),
                    "speech_ratio": round(total_speech_s / duration_s, 3)
                    if duration_s > 0
                    else 0,
                    "avg_chunk_s": round(total_speech_s / len(vad_chunks), 2)
                    if vad_chunks
                    else 0,
                    "max_chunk_s": round(
                        max(
                            (c["end"] - c["start"]) / SAMPLE_RATE
                            for c in vad_chunks
                        ),
                        2,
                    )
                    if vad_chunks
                    else 0,
                },
                "vad_windows": vad_windows,
                "raw_segments": raw_segments_debug,
                "dropped_segments": dropped_segments_debug,
            }

        return response

    def _transcribe_per_vad_chunk(
        self, audio: np.ndarray, vad_chunks: list, transcribe_kwargs: dict
    ) -> tuple:
        """Transcribe each VAD chunk independently, then stitch results."""
        from collections import namedtuple

        Seg = namedtuple(
            "Seg",
            ["start", "end", "text", "no_speech_prob", "avg_logprob",
             "compression_ratio", "words"],
        )

        chunk_kwargs = {
            k: v for k, v in transcribe_kwargs.items()
            if k not in ("audio", "vad_filter", "vad_parameters")
        }
        chunk_kwargs["vad_filter"] = False

        all_segs = []
        last_info = None

        for chunk in vad_chunks:
            chunk_start_s = chunk["start"] / SAMPLE_RATE
            chunk_audio = audio[chunk["start"]:chunk["end"]]

            if len(chunk_audio) < SAMPLE_RATE * 0.1:
                continue

            segments, info = self._model.transcribe(
                audio=chunk_audio, **chunk_kwargs
            )
            last_info = info

            for seg in segments:
                all_segs.append(Seg(
                    start=seg.start + chunk_start_s,
                    end=seg.end + chunk_start_s,
                    text=seg.text,
                    no_speech_prob=seg.no_speech_prob,
                    avg_logprob=seg.avg_logprob,
                    compression_ratio=seg.compression_ratio,
                    words=[
                        type(w)(
                            start=w.start + chunk_start_s,
                            end=w.end + chunk_start_s,
                            word=w.word,
                            probability=w.probability,
                        )
                        for w in (seg.words or [])
                    ] if seg.words else [],
                ))

        logger.info(
            f"Per-VAD-chunk transcription: {len(vad_chunks)} chunks -> "
            f"{len(all_segs)} segments"
        )
        return all_segs, last_info
