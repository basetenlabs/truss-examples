import re
from pathlib import Path

import urllib.request
from typing import Iterable
import torch
from huggingface_hub import snapshot_download
import tensorrt_llm

CACHE_DIR = Path.home() / ".cache" / "whisper-trt"
ASSETS_DIR = CACHE_DIR / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def _download_files(urls: Iterable[str]) -> Path:
    for url in urls:
        file_name = url.split("/")[-1]
        file_path = ASSETS_DIR / file_name
        if not file_path.exists():
            urllib.request.urlretrieve(url, file_path)
    return ASSETS_DIR


def download_assets() -> Path:
    return _download_files(
        [
            # TODO: download non-multilingual token for `.en` model variants
            "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken",
            "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz",
        ]
    )


# Aloways donwload assets on startup
download_assets()


def get_engine_repo_name(model_name: str) -> str:
    if torch.cuda.is_available():
        return "baseten/" + re.sub(
            r"[^a-zA-Z0-9]",
            "_",
            f"whisper_trt_{model_name}_{torch.cuda.get_device_name(0)}_{tensorrt_llm.__version__}",
        )

    raise ValueError("No CUDA device found.")


def download_engine(model_name: str) -> Path:
    engine_dir = get_engine_path(model_name)
    if not engine_dir.exists():
        engine_repo = get_engine_repo_name(model_name=model_name)
        snapshot_download(
            engine_repo,
            local_dir=engine_dir,
            max_workers=8,
        )
    return engine_dir


def get_engine_path(model_name: str) -> Path:
    engines_dir = ASSETS_DIR / "engines"
    model_dir = engines_dir / model_name
    return model_dir
