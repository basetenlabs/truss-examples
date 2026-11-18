"""WeSpeaker ResNet34-LM speaker embeddings for cross-chunk speaker linking.

Imported only when linking is on (``VV_LINK=embed`` and ``VV_CHUNK_S>0``), so the
default single-pass proxy never imports torch.

Features follow WeSpeaker's own extraction (``wespeaker/cli/speaker.py``): 16 kHz audio
scaled to int16 range, Kaldi fbank (80 mel bins, 25 ms / 10 ms, hamming window, no
dither), cepstral mean normalisation over the utterance, then ResNet34 + TSTP pooling
and the 256-d ``seg_1`` embedding. The fbank is re-implemented in torch (the vLLM image
has no torchaudio) and checked against ``torchaudio.compliance.kaldi.fbank`` in the
tests when torchaudio is installed.
"""

from __future__ import annotations

import math
import os
import threading

import numpy as np
import torch
import torch.nn.functional as F

from .wespeaker.resnet import ResNet34

SAMPLE_RATE = 16_000
FRAME_LEN = 400  # 25 ms
FRAME_SHIFT = 160  # 10 ms
N_FFT = 512
N_MELS = 80
LOW_FREQ = 20.0
PREEMPH = 0.97


def _mel(f):
    return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def kaldi_mel_banks(n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SAMPLE_RATE,
                    low: float = LOW_FREQ, high: float = 0.0) -> np.ndarray:
    """Kaldi's triangular mel filterbank, shape ``(n_mels, n_fft // 2 + 1)`` (last column 0)."""
    nyq = 0.5 * sr
    if high <= 0.0:
        high += nyq
    n_bins = n_fft // 2
    mel_lo, mel_hi = _mel(low), _mel(high)
    delta = (mel_hi - mel_lo) / (n_mels + 1)
    b = np.arange(n_mels)[:, None]
    left, center, right = mel_lo + b * delta, mel_lo + (b + 1) * delta, mel_lo + (b + 2) * delta
    m = _mel(np.arange(n_bins) * (sr / n_fft))[None, :]
    up = (m - left) / (center - left)
    down = (right - m) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down))
    return np.pad(banks, ((0, 0), (0, 1))).astype(np.float32)


def fbank(wav: torch.Tensor, banks: torch.Tensor) -> torch.Tensor:
    """Kaldi fbank of a 1-D waveform already scaled to int16 range: ``(frames, n_mels)``.

    Matches ``torchaudio.compliance.kaldi.fbank(x, num_mel_bins=80, frame_length=25,
    frame_shift=10, dither=0, window_type='hamming', sample_frequency=16000)``
    (snip_edges, remove_dc_offset, preemphasis 0.97, power spectrum, log floor = float eps).
    """
    if wav.numel() < FRAME_LEN:
        wav = F.pad(wav, (0, FRAME_LEN - wav.numel()))
    frames = wav.unfold(0, FRAME_LEN, FRAME_SHIFT)  # (T, 400), snip_edges=True
    frames = frames - frames.mean(dim=1, keepdim=True)
    prev = torch.cat([frames[:, :1], frames[:, :-1]], dim=1)
    frames = frames - PREEMPH * prev
    window = torch.hamming_window(FRAME_LEN, periodic=False, alpha=0.54, beta=0.46,
                                  device=wav.device, dtype=wav.dtype)
    frames = F.pad(frames * window, (0, N_FFT - FRAME_LEN))
    power = torch.fft.rfft(frames).abs().pow(2.0)
    mel = power @ banks.T
    return mel.clamp_min(torch.finfo(mel.dtype).eps).log()


def resample_24k_to_16k(pcm: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly

    return resample_poly(pcm.astype(np.float32), 2, 3).astype(np.float32)


def resolve_device(want: str) -> torch.device:
    """``auto``/``cuda``/``cuda:N``/``cpu`` -> a concrete device; CUDA always carries an index.

    ``torch.cuda.set_per_process_memory_fraction`` rejects an un-indexed ``torch.device("cuda")``
    (exec 5 crash-looped on exactly that), so resolve it to the current device here.
    """
    want = (want or "auto").strip().lower()
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(want)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    return dev


class SpeakerEmbedder:
    """ResNet34-LM on GPU when available (else CPU). Thread-safe; calls serialize on a lock."""

    def __init__(self, model_path: str, device: str = "auto", gpu_mem_fraction: float = 0.02,
                 cpu_threads: int = 2):
        ckpt = os.path.join(model_path, "avg_model")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"speaker-embedding checkpoint not found: {ckpt}")
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith("projection.")}
        model = ResNet34(feat_dim=N_MELS, embed_dim=256, pooling_func="TSTP", two_emb_layer=False)
        model.load_state_dict(sd, strict=True)
        self.model = model.eval()
        self.banks = torch.from_numpy(kaldi_mel_banks())
        self.device = torch.device("cpu")
        self._lock = threading.Lock()
        self._want = device
        self._gpu_mem_fraction = gpu_mem_fraction
        torch.set_num_threads(max(1, cpu_threads))

    def to_device(self) -> str:
        """Move to the configured device (call once vLLM has taken its GPU memory)."""
        dev = resolve_device(self._want)
        if dev.type == "cuda":
            if 0 < self._gpu_mem_fraction < 1:
                # Caps this process's caching allocator at fraction * total device memory
                # (H100 80 GB x 0.02 ~ 1.6 GB); a 30 s clip peaks near 150 MB plus 25 MB weights.
                torch.cuda.set_per_process_memory_fraction(float(self._gpu_mem_fraction), dev.index)
            torch.set_num_threads(1)
        with self._lock:
            self.model = self.model.to(dev)
            self.banks = self.banks.to(dev)
            self.device = dev
        self.embed(np.random.default_rng(0).normal(0, 1000, SAMPLE_RATE * 2).astype(np.float32))  # warm-up
        return str(dev)

    @torch.inference_mode()
    def embed(self, wav16k: np.ndarray) -> np.ndarray:
        """L2-normalised 256-d embedding of mono 16 kHz audio in int16 range (float array)."""
        with self._lock:
            x = torch.from_numpy(np.ascontiguousarray(wav16k, dtype=np.float32)).to(self.device)
            feats = fbank(x, self.banks)
            feats = feats - feats.mean(dim=0, keepdim=True)
            _, emb = self.model(feats.unsqueeze(0))
            emb = F.normalize(emb.float(), dim=-1)[0]
            out = emb.cpu().numpy()
        if not np.all(np.isfinite(out)) or math.isclose(float(np.linalg.norm(out)), 0.0):
            raise ValueError("non-finite speaker embedding")
        return out

    def embed_pcm24k(self, pcm: np.ndarray) -> np.ndarray:
        return self.embed(resample_24k_to_16k(pcm))
