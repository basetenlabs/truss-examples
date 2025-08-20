""" Utiliy functions to load pre-trained models more easily """

import os
import pkg_resources
from omegaconf import OmegaConf

import torch
from huggingface_hub import hf_hub_download

from src.mvdream.ldm.util import instantiate_from_config


PRETRAINED_MODELS = {
    "sd-v2.1-base-4view": {
        "config": "sd-v2-base.yaml",
        "repo_id": "MVDream/MVDream",
        "filename": "sd-v2.1-base-4view.pt",
    },
    "sd-v1.5-4view": {
        "config": "sd-v1.yaml",
        "repo_id": "MVDream/MVDream",
        "filename": "sd-v1.5-4view.pt",
    },
}


def get_config_file(config_path):
    cfg_file = pkg_resources.resource_filename(
        "src.mvdream", os.path.join("configs", config_path)
    )
    if not os.path.exists(cfg_file):
        raise RuntimeError(f"Config {config_path} not available!")
    return cfg_file


def build_model(model_name, ckpt_path=None, cache_dir=None):
    if not model_name in PRETRAINED_MODELS:
        raise RuntimeError(
            f"Model name {model_name} is not a pre-trained model. Available models are:\n- "
            + "\n- ".join(PRETRAINED_MODELS.keys())
        )
    model_info = PRETRAINED_MODELS[model_name]

    # Instiantiate the model
    print(f"Loading model from config: {model_info['config']}")
    config_file = get_config_file(model_info["config"])
    config = OmegaConf.load(config_file)
    model = instantiate_from_config(config.model)

    # Load pre-trained checkpoint from huggingface
    if not ckpt_path:
        ckpt_path = hf_hub_download(
            repo_id=model_info["repo_id"],
            filename=model_info["filename"],
            cache_dir=cache_dir,
        )
        print(f"Loading model from cache file: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if "state_dict" in checkpoint.keys():  # if used pytorch lightening
        print(f"Loading model from {ckpt_path}...")
        adjusted_state_dict = {}
        for key, value in checkpoint["state_dict"].items():
            # Keep and adjust the key only if it starts with 'model.'
            if key.startswith("model."):
                new_key = key[len("model.") :]  # Remove 'model.' from the beginning
                adjusted_state_dict[new_key] = value
        model.load_state_dict(adjusted_state_dict)
    elif ckpt_path.endswith(".bin"):
        print(f"Loading model from {ckpt_path}...")
        adjusted_state_dict = {}
        for key, value in checkpoint.items():
            # Keep and adjust the key only if it starts with 'model.'
            if key.startswith("model."):
                new_key = key[len("model.") :]  # Remove 'model.' from the beginning
                adjusted_state_dict[new_key] = value
        model.load_state_dict(adjusted_state_dict)
    else:
        model.load_state_dict(checkpoint)

    return model
