import base64
import random
import logging
from io import BytesIO

import numpy as np
import torch
from diffusers import DiffusionPipeline
from PIL import Image

logging.basicConfig(level=logging.INFO)
MAX_SEED = np.iinfo(np.int32).max


class Model:
    def __init__(self, **kwargs):
        self.pipe = None
        self.repo_id = "Qwen/Qwen-Image"

    def load(self):
        # Ensure CUDA is available for H100 deployment
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for this model. Please ensure GPU is available."
            )

        # Configure for H100 GPU with optimal settings
        torch_dtype = torch.bfloat16
        device = "cuda"

        # Log GPU information for debugging
        logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logging.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

        self.pipe = DiffusionPipeline.from_pretrained(
            self.repo_id, torch_dtype=torch_dtype
        ).to(device)

    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64

    def predict(self, model_input):
        seed = model_input.get("seed")
        prompt = model_input.get("prompt")
        negative_prompt = model_input.get("negative_prompt", "")
        width = model_input.get("width", 1024)
        height = model_input.get("height", 1024)
        num_inference_steps = model_input.get("num_inference_steps", 50)
        true_cfg_scale = model_input.get("true_cfg_scale", 4.0)

        # Add positive magic prompt for better quality
        positive_magic = {
            "en": "Ultra HD, 4K, cinematic composition.",
            "zh": "超清，4K，电影级构图",
        }

        # Determine if prompt is Chinese or English and add appropriate magic
        if any("\u4e00" <= char <= "\u9fff" for char in prompt):
            magic_prompt = positive_magic["zh"]
        else:
            magic_prompt = positive_magic["en"]

        full_prompt = prompt + " " + magic_prompt

        if not seed:
            seed = random.randint(0, MAX_SEED)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        image = self.pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator,
        ).images[0]

        b64_results = self.convert_to_b64(image)
        return {"data": b64_results}
