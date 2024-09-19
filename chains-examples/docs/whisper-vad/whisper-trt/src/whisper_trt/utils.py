# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import os
import soundfile
import torch
import torch.nn.functional as F
from functools import lru_cache
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import Optional, Union
import logging

Pathlike = Union[str, Path]

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30  # 30-second chunks because of whisper training setup
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk


def _load_audio(file: str, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0", "-i", file, "-f", "s16le", "-ac",
        "1", "-acodec", "pcm_s16le", "-ar",
        str(sr), "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    return (
        torch.frombuffer(out, dtype=torch.int16).flatten().to(torch.float32) / 32768.0
    )


def _load_audio_wav_format(wav_path: str) -> tuple[torch.Tensor, int]:
    # make sure audio in .wav format
    assert wav_path.endswith(".wav"), f"Only support .wav format, but got {wav_path}"
    waveform, sample_rate = soundfile.read(wav_path, dtype=np.float32)
    assert sample_rate == 16000, f"Only support 16k sample rate, but got {sample_rate}"
    return torch.from_numpy(waveform), sample_rate


def pad_or_trim(
    array: torch.Tensor,
    length: int = N_SAMPLES,
    *,
    axis: int = -1,
    raise_when_trimmed: bool = False,
) -> torch.Tensor:
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    input_length = array.shape[axis]
    input_length_sec = input_length / SAMPLE_RATE
    length_secs = length / SAMPLE_RATE

    # if input_length > length:
    #     if raise_when_trimmed:
    #         raise ValueError(
    #             "Audio input is longer than the supported max length: :"
    #             f"`{input_length_sec}`s,  (supported:`{length_secs}`s). "
    #             "This would result in lost audio. Use shorter inputs or deploy a "
    #             "model with longer input length."
    #         )
    #     logging.warn("Trimming audio input to the supported max length.")
    #     array = torch.narrow(array, dim=axis, start=0, length=length)

    # el
    if input_length < length:
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])

    return array


@lru_cache(maxsize=None)
def mel_filters(
    device, n_mels: int, mel_filters_dir: Optional[str] = None
) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"
    if mel_filters_dir is None:
        mel_filters_path = os.path.join(
            os.path.dirname(__file__), "assets", "mel_filters.npz"
        )
    else:
        mel_filters_path = os.path.join(mel_filters_dir, "mel_filters.npz")
    with np.load(mel_filters_path) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio_input: Union[str, np.ndarray, torch.Tensor],
    n_mels: int,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
    mel_filters_dir: Optional[str] = None,
    raise_when_trimmed: bool = False,
) -> torch.Tensor:
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio_input: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT
    mel_filters_dir:
    raise_when_trimmed: raises ValueError if the audio is too long and would be trimmed.

    Returns
    -------
    torch.Tensor, shape = (80 or 128, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if isinstance(audio_input, str):
        if audio_input.endswith(".wav"):
            audio, _ = _load_audio_wav_format(audio_input)
        else:
            audio = _load_audio(audio_input)
    elif isinstance(audio_input, torch.Tensor):
        audio = audio_input.to(torch.float32)
    elif isinstance(audio_input, np.ndarray):
        audio = torch.from_numpy(audio_input.astype(np.float32))
    else:
        raise ValueError(f"Unsupported audio type: {type(audio_input)}")

    assert audio.dtype == torch.float32
    audio = pad_or_trim(audio, N_SAMPLES, raise_when_trimmed=raise_when_trimmed)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels, mel_filters_dir)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
