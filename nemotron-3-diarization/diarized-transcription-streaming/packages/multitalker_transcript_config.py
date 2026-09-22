# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the NVIDIA Open Model License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MultitalkerTranscriptionConfig:
    """
    Configuration for Multi-talker transcription with an ASR model and a diarization model.
    """
    # Required configs
    diar_model: Optional[str] = None  # Path to a .nemo file
    diar_pretrained_name: Optional[str] = None  # Name of a pretrained model
    max_num_of_spks: Optional[int] = 4  # maximum number of speakers
    parallel_speaker_strategy: bool = True  # whether to use parallel speaker strategy
    masked_asr: bool = True  # whether to use masked ASR
    mask_preencode: bool = False  # whether to mask preencode or mask features
    cache_gating: bool = True  # whether to use cache gating
    cache_gating_buffer_size: int = 2  # buffer size for cache gating
    single_speaker_mode: bool = False  # whether to use single speaker mode

    # General configs
    session_len_sec: float = -1  # End-to-end diarization session length in seconds
    num_workers: int = 8
    random_seed: Optional[int] = None  # seed number going to be used in seed_everything()
    log: bool = True  # If True,log will be printed

    # Streaming diarization configs
    streaming_mode: bool = True  # If True, streaming diarization will be used.
    spkcache_len: int = 188
    spkcache_refresh_rate: int = 0
    fifo_len: int = 188
    chunk_len: int = 0
    chunk_left_context: int = 0
    chunk_right_context: int = 0

    # If `cuda` is a negative number, inference will be on CPU only.
    cuda: Optional[int] = None
    allow_mps: bool = False  # allow to select MPS device (Apple Silicon M-series GPU)
    matmul_precision: str = "highest"  # Literal["highest", "high", "medium"]

    # ASR Configs
    asr_model: Optional[str] = None
    device: str = 'cuda'
    audio_file: Optional[str] = None
    manifest_file: Optional[str] = None
    use_amp: bool = True
    debug_mode: bool = False
    batch_size: int = 32
    chunk_size: int = -1
    shift_size: int = -1
    left_chunks: int = 2
    online_normalization: bool = False
    output_path: Optional[str] = None
    pad_and_drop_preencoded: bool = False
    set_decoder: Optional[str] = None  # ["ctc", "rnnt"]
    att_context_size: Optional[List[int]] = field(default_factory=lambda: [70, 13])
    generate_realtime_scripts: bool = False

    word_window: int = 50
    sent_break_sec: float = 30.0
    fix_prev_words_count: int = 5
    update_prev_words_sentence: int = 5
    left_frame_shift: int = -1
    right_frame_shift: int = 0
    min_sigmoid_val: float = 1e-2
    discarded_frames: int = 8
    print_time: bool = True
    print_sample_indices: List[int] = field(default_factory=lambda: [0])
    colored_text: bool = True
    real_time_mode: bool = False
    print_path: Optional[str] = None

    ignored_initial_frame_steps: int = 5
    verbose: bool = False

    feat_len_sec: float = 0.01
    finetune_realtime_ratio: float = 0.01

    spk_supervision: str = "diar"  # ["diar", "rttm"]
    binary_diar_preds: bool = False

    @staticmethod
    def init_diar_model(cfg, diar_model):
        # Set streaming mode diar_model params (matching the diarization setup from lines 263-271 of reference file)
        diar_model.streaming_mode = cfg.streaming_mode
        diar_model.sortformer_modules.chunk_len = cfg.chunk_len if cfg.chunk_len > 0 else 6
        diar_model.sortformer_modules.spkcache_len = cfg.spkcache_len
        diar_model.sortformer_modules.chunk_left_context = cfg.chunk_left_context
        diar_model.sortformer_modules.chunk_right_context = cfg.chunk_right_context if cfg.chunk_right_context > 0 else 7
        diar_model.sortformer_modules.fifo_len = cfg.fifo_len
        diar_model.sortformer_modules.log = cfg.log
        diar_model.sortformer_modules.spkcache_refresh_rate = cfg.spkcache_refresh_rate
        return diar_model