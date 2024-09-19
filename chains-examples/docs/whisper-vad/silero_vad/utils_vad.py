import math
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class BatchSettings:
    lookback_window_secs: float = 4
    max_batch_size: int = 400
    min_section_length: int = 32


def init_jit_model(model_path: str, device=torch.device("cpu")):
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


@torch.no_grad()
def get_speech_timestamps(
    audio: torch.Tensor,
    model,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float("inf"),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
    visualize_probs: bool = False,
    progress_tracking_callback: Callable[[float], None] = None,
    window_size_samples: int = 512,
    batch_settings: Optional[BatchSettings] = None,
):
    """
    This method is used for splitting long audios into speech chunks using silero VAD

    Parameters
    ----------
    audio: torch.Tensor, one dimensional
        One dimensional float torch.Tensor, other types are casted to torch if possible

    model: preloaded .jit/.onnx silero VAD model

    threshold: float (default - 0.5)
        Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
        It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

    sampling_rate: int (default - 16000)
        Currently silero VAD models support 8000 and 16000 (or multiply of 16000) sample rates

    min_speech_duration_ms: int (default - 250 milliseconds)
        Final speech chunks shorter min_speech_duration_ms are thrown out

    max_speech_duration_s: int (default -  inf)
        Maximum duration of speech chunks in seconds
        Chunks longer than max_speech_duration_s will be split at the timestamp of the last silence that lasts more than 100ms (if any), to prevent agressive cutting.
        Otherwise, they will be split aggressively just before max_speech_duration_s.

    min_silence_duration_ms: int (default - 100 milliseconds)
        In the end of each speech chunk wait for min_silence_duration_ms before separating it

    speech_pad_ms: int (default - 30 milliseconds)
        Final speech chunks are padded by speech_pad_ms each side

    return_seconds: bool (default - False)
        whether return timestamps in seconds (default - samples)

    visualize_probs: bool (default - False)
        whether draw prob hist or not

    progress_tracking_callback: Callable[[float], None] (default - None)
        callback function taking progress in percents as an argument

    window_size_samples: int (default - 512 samples)
        !!! DEPRECATED, DOES NOTHING !!!

    Returns
    ----------
    speeches: list of dicts
        list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
    """

    if not torch.is_tensor(audio):
        try:
            audio = torch.Tensor(audio)
        except:
            raise TypeError("Audio cannot be casted to tensor. Cast it manually")

    if len(audio.shape) > 1:
        for i in range(len(audio.shape)):  # trying to squeeze empty dimensions
            audio = audio.squeeze(0)
        if len(audio.shape) > 1:
            raise ValueError(
                "More than one dimension in audio. Are you trying to process audio with 2 channels?"
            )

    if sampling_rate > 16000 and (sampling_rate % 16000 == 0):
        step = sampling_rate // 16000
        sampling_rate = 16000
        audio = audio[::step]
        warnings.warn(
            "Sampling rate is a multiply of 16000, casting to 16000 manually!"
        )
    else:
        step = 1

    if sampling_rate not in [8000, 16000]:
        raise ValueError(
            "Currently silero VAD models support 8000 and 16000 (or multiply of 16000) sample rates"
        )

    window_size_samples = 512 if sampling_rate == 16000 else 256

    model.reset_states()
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s
        - window_size_samples
        - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    speech_probs = []
    if batch_settings is None:
        for current_start_sample in range(0, audio_length_samples, window_size_samples):
            chunk = audio[
                current_start_sample : current_start_sample + window_size_samples
            ]
            if len(chunk) < window_size_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, int(window_size_samples - len(chunk)))
                )
            speech_prob = model(chunk, sampling_rate).item()
            speech_probs.append(speech_prob)
            # caculate progress and seng it to callback function
            progress = current_start_sample + window_size_samples
            if progress > audio_length_samples:
                progress = audio_length_samples
            progress_percent = (progress / audio_length_samples) * 100
            if progress_tracking_callback:
                progress_tracking_callback(progress_percent)
    else:
        # batch prediction
        speech_probs = _batch_predict(
            model,
            audio,
            window_size_samples,
            batch_settings,
            sampling_rate,
        )

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15
    temp_end = 0  # to save potential segment end (and tolerate some silence)
    prev_end = next_start = (
        0  # to save potential segment limits in case of maximum segment size reached
    )

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = window_size_samples * i

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech["start"] = window_size_samples * i
            continue

        if (
            triggered
            and (window_size_samples * i) - current_speech["start"] > max_speech_samples
        ):
            if prev_end:
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                if (
                    next_start < prev_end
                ):  # previously reached silence (< neg_thres) and is still not speech (< thres)
                    triggered = False
                else:
                    current_speech["start"] = next_start
                prev_end = next_start = temp_end = 0
            else:
                current_speech["end"] = window_size_samples * i
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            if (
                (window_size_samples * i) - temp_end
            ) > min_silence_samples_at_max_speech:  # condition to avoid cutting in very short silence
                prev_end = temp_end
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                if (
                    current_speech["end"] - current_speech["start"]
                ) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    if (
        current_speech
        and (audio_length_samples - current_speech["start"]) > min_speech_samples
    ):
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )

    if return_seconds:
        for speech_dict in speeches:
            speech_dict["start"] = round(speech_dict["start"] / sampling_rate, 1)
            speech_dict["end"] = round(speech_dict["end"] / sampling_rate, 1)
    elif step > 1:
        for speech_dict in speeches:
            speech_dict["start"] *= step
            speech_dict["end"] *= step

    if visualize_probs:
        make_visualization(speech_probs, window_size_samples / sampling_rate)

    return speeches


def collect_chunks(tss: List[dict], wav: torch.Tensor):
    chunks = []
    for i in tss:
        chunks.append(wav[i["start"] : i["end"]])
    return torch.cat(chunks)


def _batch_predict(
    model, audio, window_size_samples, batch_settings: BatchSettings, sampling_rate: int
):
    device = next(model.parameters()).device
    audio = audio.to(device)
    audio_length_samples = len(audio)
    num_chunks = int(math.ceil(audio_length_samples / window_size_samples))
    to_pad_for_chunking = num_chunks * window_size_samples - audio_length_samples
    padded_audio = torch.nn.functional.pad(
        audio, (0, to_pad_for_chunking), "constant", 0
    )
    chunks = padded_audio.view(num_chunks, window_size_samples)

    # In number of chunks
    section_length = _calc_section_length(
        num_chunks,
        batch_settings.max_batch_size,
        batch_settings.min_section_length,
    )
    num_sections = int(math.ceil(num_chunks / section_length))

    # In number of chunks
    lookback_window = int(
        batch_settings.lookback_window_secs * sampling_rate / window_size_samples
    )
    chunks = F.pad(chunks, (0, 0, 0, lookback_window + section_length), "constant", 0)
    chunk_probabilities = torch.zeros(
        num_sections * section_length + lookback_window
    ).to(device)
    for step in range(lookback_window + section_length):
        step_index_range = slice(
            step, num_sections * section_length + step, section_length
        )
        chunk_batch = chunks[step_index_range]

        model_probs = model(chunk_batch, sampling_rate)
        # skip first LOOKBACK_WINDOW steps, because we don't have enough context.
        if step < lookback_window:
            chunk_probabilities[step] = model_probs[0].item()
        else:
            chunk_probabilities[step_index_range] = model_probs.view(-1)
    return chunk_probabilities.to(torch.device("cpu")).tolist()[:num_chunks]


def _calc_section_length(
    num_chunks: int, max_batch_size: int, min_section_length: int = 32
):
    min_section_length = 32
    section_length_based_on_max_batch_size = int(math.ceil(num_chunks / max_batch_size))
    return max(min_section_length, section_length_based_on_max_batch_size)
