"""Audio-encoder forward without GPU->CPU syncs.

vLLM's Qwen3-Omni audio encoder reads ``feature_lens`` back from the GPU five times per
batch (``chunk_num.sum()``, two ``tolist()``, ``max().item()`` and a boolean-mask gather).
Every one of those values is a function of the per-item feature lengths, which the runner
already holds on the CPU (``audio_feature_lengths`` is a keep-on-CPU field) and copies to
the GPU right before the call. Each readback drains the CUDA queue, so the engine thread
sits idle until the encoder kernels finish and the LLM step cannot be enqueued behind
them: on fresh audio the encoder is ~5 % of GPU time but ~25 % of wall.

This module recomputes the same lengths, chunk splits, gather indices and ``cu_seqlens``
from the CPU copy and replaces the mask gather with ``index_select``. Kernels, weights and
the order of rows are unchanged, so the output is identical; only the host waits go.

``install(encoder_cls, asr_cls)`` swaps the encoder's ``forward``; ``process_audio_input``
replaces ``Qwen3ASRForConditionalGeneration._process_audio_input`` so the CPU lengths reach
the encoder.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cnn_output_length(length: int) -> int:
    for _ in range(3):
        length = (length - 1) // 2 + 1
    return length


def _chunk_lengths(feature_lens: list[int], n2: int) -> list[int]:
    # Mirrors: ceil(len / n2) chunks of n2 frames, tail = len % n2, a zero tail means a full chunk.
    out: list[int] = []
    for length in feature_lens:
        n = -(-length // n2)
        tail = length % n2 or n2
        out.extend([n2] * (n - 1))
        out.append(tail)
    return out


def audio_encoder_forward_nosync(
    self,
    input_features: torch.Tensor,
    feature_lens: torch.Tensor,
    aftercnn_lens: torch.Tensor,
    *,
    feature_lens_cpu: torch.Tensor | None = None,
    aftercnn_lens_cpu: torch.Tensor | None = None,
):
    from vllm.model_executor.layers.attention.mm_encoder_attention import (
        MMEncoderAttention,
    )
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed
    from vllm.utils.torch_utils import async_tensor_h2d

    device = input_features.device
    n2 = self.n_window * 2

    if feature_lens_cpu is None:  # caller did not pass the CPU copy: one readback instead of five
        with gpu_sync_allowed():
            feature_lens_cpu = feature_lens.cpu()
    if aftercnn_lens_cpu is None:
        with gpu_sync_allowed():
            aftercnn_lens_cpu = aftercnn_lens.cpu()
    feature_lens_list = [int(x) for x in feature_lens_cpu.tolist()]
    aftercnn_lens_list = [int(x) for x in aftercnn_lens_cpu.tolist()]

    chunk_lengths = _chunk_lengths(feature_lens_list, n2)
    chunk_list = input_features.T.split(chunk_lengths, dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)

    lens_after_cnn = [_cnn_output_length(length) for length in chunk_lengths]
    max_len_after_cnn = max(lens_after_cnn)

    padded_feature = padded_feature.unsqueeze(1)

    if padded_feature.size(0) <= self.conv_chunksize:
        padded_embed = F.gelu(self.conv2d1(padded_feature))
        padded_embed = F.gelu(self.conv2d2(padded_embed))
        padded_embed = F.gelu(self.conv2d3(padded_embed))
    else:
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)

    b, c, f, t = padded_embed.size()
    assert t == max_len_after_cnn, (t, max_len_after_cnn)
    padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

    positional_embedding = (
        self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
        .unsqueeze(0)
        .to(padded_embed.dtype)
    )
    padded_embed = padded_embed + positional_embedding

    # Same rows, same order as padded_embed[mask]: row-major over (chunk, valid frame).
    gather_idx = np.concatenate(
        [i * max_len_after_cnn + np.arange(length) for i, length in enumerate(lens_after_cnn)]
    ).astype(np.int64)
    hidden_states = padded_embed.reshape(b * t, -1).index_select(
        0, async_tensor_h2d(gather_idx, device)
    )

    cu_chunk_lens = [0]
    window_aftercnn = max_len_after_cnn * (self.n_window_infer // n2)
    for cnn_len in aftercnn_lens_list:
        num_full_chunks = cnn_len // window_aftercnn
        remainder = cnn_len % window_aftercnn
        cu_chunk_lens.extend([window_aftercnn] * num_full_chunks)
        if remainder:
            cu_chunk_lens.append(remainder)
    cu_seqlens_np = np.cumsum(np.asarray(cu_chunk_lens, dtype=np.int32), dtype=np.int32)
    cu_seqlens = async_tensor_h2d(cu_seqlens_np, device)

    # Original: a device tensor for FLASH_ATTN / ROCM_AITER_FA / TRITON_ATTN, None otherwise. The
    # custom op (vit_attn_wrappers.flash_attn_maxseqlen_wrapper) requires a Tensor and calls
    # .item() on it in every encoder layer; on a CPU tensor that is free instead of a sync.
    max_seqlen = None
    if self.attn_backend in _MAX_SEQLEN_BACKENDS():
        max_seqlen = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu_seqlens_np),
            dtype=torch.int32,
        )

    for encoder_layer in self.layers:
        hidden_states = encoder_layer(hidden_states, cu_seqlens, max_seqlen)

    hidden_states = self.ln_post(hidden_states)
    hidden_states = self.proj1(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.proj2(hidden_states)
    return hidden_states


def _MAX_SEQLEN_BACKENDS():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    return {
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.ROCM_AITER_FA,
        AttentionBackendEnum.TRITON_ATTN,
    }


def process_audio_input(self, audio_input) -> tuple[torch.Tensor, ...]:
    """Replacement for Qwen3ASRForConditionalGeneration._process_audio_input."""
    from vllm.model_executor.models.qwen3_asr import _get_feat_extract_output_lengths
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed
    from vllm.utils.torch_utils import async_tensor_h2d

    input_features = audio_input["input_features"]
    feature_lens_cpu = torch.as_tensor(audio_input["audio_feature_lengths"])
    if not feature_lens_cpu.is_cpu:
        with gpu_sync_allowed():
            feature_lens_cpu = feature_lens_cpu.cpu()
    feature_lens_cpu = feature_lens_cpu.reshape(-1)
    aftercnn_lens_cpu = _get_feat_extract_output_lengths(feature_lens_cpu)

    audio_features = self.audio_tower(
        input_features.to(self.audio_tower.dtype),
        feature_lens=async_tensor_h2d(feature_lens_cpu, input_features.device),
        aftercnn_lens=async_tensor_h2d(aftercnn_lens_cpu, input_features.device),
        feature_lens_cpu=feature_lens_cpu,
        aftercnn_lens_cpu=aftercnn_lens_cpu,
    )
    return audio_features.split([int(x) for x in aftercnn_lens_cpu.tolist()])


def install(encoder_cls, asr_cls) -> None:
    encoder_cls._forward_vllm = encoder_cls.forward
    encoder_cls.forward = audio_encoder_forward_nosync
    asr_cls._process_audio_input_vllm = asr_cls._process_audio_input
    asr_cls._process_audio_input = process_audio_input
