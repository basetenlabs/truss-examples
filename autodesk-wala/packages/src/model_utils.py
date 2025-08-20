import logging
import torch
from src.latent_module import (
    Trainer_Condition_Network,
)
from src.networks.callbacks import EMACallback
import os
import json
import backoff
import urllib.error
from huggingface_hub import hf_hub_download
import re


class DotDict(dict):
    def __getattr__(self, attr_):
        try:
            return self[attr_]
        except KeyError:
            print(f"'DotDict' object has no attribute '{attr_}'")

    def __setattr__(self, attr_, value):
        self[attr_] = value

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __deepcopy__(self, memo):
        return DotDict(copy.deepcopy(dict(self), memo))


@backoff.on_exception(
    backoff.expo, (AttributeError, urllib.error.URLError), max_time=120
)
def load_latent_model(
    json_path,
    checkpoint_path,
    compile_model,
    device=None,
    eval=True,
):
    with open(json_path, "r") as file:
        args = json.load(file, object_hook=DotDict)

    model = Trainer_Condition_Network.load_from_checkpoint(
        checkpoint_path=checkpoint_path, map_location="cpu", args=args
    )

    if hasattr(model, "ema_state_dict") and model.ema_state_dict is not None:
        # load EMA weights
        ema = EMACallback(decay=0.9999)
        ema.reload_weight = model.ema_state_dict
        ema.reload_weight_for_pl_module(model)
        ema.copy_to_pl_module(model)

    if compile_model:
        logging.info("Compiling models...")
        model.network.training_losses = torch.compile(model.network.training_losses)
        model.network.inference = torch.compile(model.network.inference)
        if hasattr(model, "clip_model"):
            model.clip_model.forward = torch.compile(model.clip_model.forward)
        logging.info("Done Compiling models...")

    if device is not None:
        model = model.to(device)
    if eval:
        model.eval()

    return model


class Model:

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        if os.path.isfile(pretrained_model_name_or_path):
            checkpoint_path = pretrained_model_name_or_path
            json_path = os.path.dirname(checkpoint_path) + "/args.json"
        else:
            checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename="checkpoint.ckpt"
            )
            json_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path, filename="args.json"
            )

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        model = load_latent_model(
            json_path,
            checkpoint_path,
            compile_model=False,
            device=device,
        )
        return model
