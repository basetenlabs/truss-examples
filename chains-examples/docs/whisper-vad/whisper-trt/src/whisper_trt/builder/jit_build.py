import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import NamedTuple, Optional

from ..assets import ASSETS_DIR, CACHE_DIR, get_engine_path


class BuildArguments(NamedTuple):
    # TODO: support quanitzation
    beam_width: int = 5
    batch_size: int = 8
    max_input_len: int = 256
    max_seq_len: int = 1000


SOURCE_WEIGHTS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
}


def build_engine(model_name: str, args: Optional[BuildArguments] = None) -> Path:
    local_engine_dir = get_engine_path(model_name)
    # Skip build if there is a viable engine.
    if local_engine_dir.exists():
        return local_engine_dir
    local_engine_dir.mkdir(parents=True, exist_ok=True)

    if not args:
        args = BuildArguments()

    try:
        source_weights_url = SOURCE_WEIGHTS[model_name]
    except KeyError:
        # TODO: support distil models
        raise ValueError(f"{model_name} is not a valid Whisper model name.")
    source_weight_path = ASSETS_DIR / source_weights_url.split("/")[-1]

    urllib.request.urlretrieve(source_weights_url, source_weight_path)

    # TODO: get this form build args and support quantization.
    inference_precision = "float16"
    temp_checkpoint_dir = CACHE_DIR / f"temp_{os.getpid()}_{model_name}"
    temp_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Convert checkpoint
    convert_command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "convert_checkpoint.py"),
        "--model_dir",
        str(ASSETS_DIR),
        "--model_name",
        model_name,
        "--dtype",
        inference_precision,
        "--output_dir",
        str(temp_checkpoint_dir),
    ]
    subprocess.run(convert_command, check=True)

    # Build encoder
    build_encoder_command = [
        "trtllm-build",
        "--checkpoint_dir",
        str(temp_checkpoint_dir / "encoder"),
        "--output_dir",
        str(local_engine_dir / "encoder"),
        "--paged_kv_cache",
        "disable",
        "--moe_plugin",
        "disable",
        "--enable_xqa",
        "disable",
        "--use_custom_all_reduce",
        "disable",
        "--max_batch_size",
        str(args.batch_size),
        "--gemm_plugin",
        "disable",
        "--bert_attention_plugin",
        inference_precision,
        "--remove_input_padding",
        "disable",
        "--max_input_len",
        "1500",
    ]
    subprocess.run(build_encoder_command, check=True)

    # Build decoder
    build_decoder_command = [
        "trtllm-build",
        "--checkpoint_dir",
        str(temp_checkpoint_dir / "decoder"),
        "--output_dir",
        str(local_engine_dir / "decoder"),
        "--paged_kv_cache",
        "disable",
        "--moe_plugin",
        "disable",
        "--enable_xqa",
        "disable",
        "--use_custom_all_reduce",
        "disable",
        "--max_beam_width",
        str(args.beam_width),
        "--max_batch_size",
        str(args.batch_size),
        "--max_seq_len",
        str(args.max_seq_len),
        "--max_input_len",
        str(args.max_input_len),
        "--max_encoder_input_len",
        "1500",
        "--gemm_plugin",
        inference_precision,
        "--bert_attention_plugin",
        inference_precision,
        "--gpt_attention_plugin",
        inference_precision,
        "--remove_input_padding",
        "disable",
    ]
    subprocess.run(build_decoder_command, check=True)

    # Delete source weight files
    source_weight_path.unlink()
    shutil.rmtree(temp_checkpoint_dir)

    return local_engine_dir
