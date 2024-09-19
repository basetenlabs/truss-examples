import base64
import functools
import io
import time
from typing import AsyncIterator, List, NamedTuple, Optional

import torch
import utilities.data_types as data_types
from silero_vad import load_cpu_model
from silero_vad.utils_vad import BatchSettings, get_speech_timestamps
from utilities.helpers import AudioStreamer


@functools.lru_cache
def get_vad_model():
    # TODO: check gpu availability, if gpu is not aviailable just use the normal load model
    """Returns the VAD model instance."""
    return load_cpu_model()


class VadOptions(NamedTuple):
    """VAD options.

    Attributes:
      threshold: Speech threshold. Silero VAD outputs speech probabilities for each audio chunk,
        probabilities ABOVE this value are considered as SPEECH. It is better to tune this
        parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
      min_speech_duration_ms: Final speech chunks shorter min_speech_duration_ms are thrown out.
      max_speech_duration_s: Maximum duration of speech chunks in seconds. Chunks longer
        than max_speech_duration_s will be split at the timestamp of the last silence that
        lasts more than 100ms (if any), to prevent aggressive cutting. Otherwise, they will be
        split aggressively just before max_speech_duration_s.
      min_silence_duration_ms: In the end of each speech chunk wait for min_silence_duration_ms
        before separating it
      speech_pad_ms: Final speech chunks are padded by speech_pad_ms each side
    """

    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400


def get_vad_segments(
    audio: torch.Tensor,
    sampling_rate: int = 16_000,
    vad_options: Optional[VadOptions] = None,
    batch_settings: Optional[BatchSettings] = None,
    **kwargs,
) -> List[dict]:
    """This method is used for splitting long audios into speech chunks using silero VAD.

    Args:
      audio: One dimensional float array.
      vad_options: Options for VAD processing.
      kwargs: VAD options passed as keyword arguments for backward compatibility.

    Returns:
      List of dicts containing begin and end samples of each speech chunk.
    """
    if vad_options is None:
        vad_options = VadOptions(**kwargs)

    if batch_settings is None:
        # TODO: make this nicer
        batch_settings = BatchSettings(lookback_window_secs=4)  # **kwargs)

    model = get_vad_model()
    return get_speech_timestamps(
        audio,
        model,
        batch_settings=batch_settings,
        threshold=vad_options.threshold,
        min_speech_duration_ms=vad_options.min_speech_duration_ms,
        max_speech_duration_s=vad_options.max_speech_duration_s,
        min_silence_duration_ms=vad_options.min_silence_duration_ms,
        speech_pad_ms=vad_options.speech_pad_ms,
        window_size_samples=512,
        sampling_rate=sampling_rate,
    )


async def wav_chunker_vad(
    audio_streamer: AudioStreamer,
    vad_options: Optional["VadOptions"] = None,
    timing_dict: Optional[dict] = None,
) -> AsyncIterator[tuple[data_types.MicroChunkInfo, str]]:
    """Consumes the download stream and yields small chunks of b64-encoded wav."""
    import torchaudio
    from chunker.vad import get_vad_segments
    from silero_vad.utils_vad import collect_chunks

    start_time = time.time()
    audio_data = await audio_streamer.wav_stream.read()
    wav_read_time = time.time() - start_time
    if timing_dict is not None:
        timing_dict["wav_read_time"] = wav_read_time

    start_chunk_time = time.time()
    buffer = io.BytesIO(audio_data)
    waveform, sampling_rate = torchaudio.load(buffer)
    speech_timestamps = get_vad_segments(waveform, sampling_rate, vad_options)
    vad_time = time.time() - start_chunk_time

    if timing_dict is not None:
        timing_dict["vad_time"] = vad_time
    audio_tensor = waveform.squeeze(0)  # Removes the channel dimension
    for i, sts in enumerate(speech_timestamps):
        audio_chunk_tensor = collect_chunks([sts], audio_tensor)

        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            audio_chunk_tensor.unsqueeze(0),
            sample_rate=sampling_rate,
            format="wav",
        )
        buffer.seek(0)

        # Encode the binary data to Base64
        audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        seg_info = data_types.MicroChunkInfo(
            start_time_sec=round(sts["start"] / sampling_rate, 6),
            end_time_sec=round(sts["end"] / sampling_rate, 6),
            duration_sec=round(sts["end"] / sampling_rate, 6)
            - round(sts["start"] / sampling_rate, 6),
            index=i,
        )
        yield seg_info, audio_b64
