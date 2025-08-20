import logging
import urllib.error

import backoff
import torch
from src.mvdream_module import MVDreamModule
from src.latent_module import Trainer_Condition_Network
from huggingface_hub import hf_hub_download
import os


def load_mvdream_model(
    pretrained_model_name_or_path,
    device=None,
    eval=True,
):

    if os.path.isfile(pretrained_model_name_or_path):
        checkpoint_path = pretrained_model_name_or_path
        json_path = os.path.dirname(pretrained_model_name_or_path) + "/args.json"
    else:
        checkpoint_path = hf_hub_download(
            repo_id=pretrained_model_name_or_path, filename="checkpoint.ckpt"
        )

    model = MVDreamModule.load_from_checkpoint(checkpoint_path=checkpoint_path)

    if device is not None:
        model = model.to(device)
    if eval:
        model.eval()

    return model


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

def download_args_path(s3_path):
    from boto3.s3.transfer import TransferConfig
    import boto3
    import botocore
    import io
    import json

    max_concurrency = 10000
    multipart_size = 1024 * 1024 * 8
    boto3_config = botocore.config.Config(
        max_pool_connections=max_concurrency,
        s3={'max_queue_size': max_concurrency},
        connect_timeout=180,
        read_timeout=180,
        retries={'max_attempts': 10}
    )

    # Ensure the path starts with 's3://'
    if not s3_path.startswith("s3://"):
        raise ValueError("s3_path must start with 's3://'")

    # Remove 's3://' from the path and split by the first '/'
    s3_path_cleaned = s3_path[5:]
    bucket_name, file_path = s3_path_cleaned.split("/", 1)  # Split into bucket name and file path

    # Initialize boto3 S3 resource
    s3 = boto3.resource('s3', config=boto3_config, region_name='us-east-1')

    # Download the JSON file
    file = io.BytesIO()
    s3.Bucket(bucket_name).download_fileobj(file_path, file)
    
    # Move the file pointer to the beginning
    file.seek(0)
    
    # Load the JSON data
    data = json.load(file, object_hook=DotDict)
    
    return data

@backoff.on_exception(backoff.expo, (AttributeError, urllib.error.URLError), max_time=120)
def load_latent_model(
    json_path, 
    checkpoint_path,
    compile_model,
    device=None,
    eval=True,
):  
    args = download_args_path(json_path)
    #print(args)
    model = Trainer_Condition_Network.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location="cpu",
        args=args
    )

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