import base64
import random
from io import BytesIO

import numpy as np
import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from PIL import Image

MODEL_NAME = "stabilityai/stable-diffusion-3-medium-diffusers"
MAX_SEED = np.iinfo(np.int32).max


class Model:
    def __init__(self, **kwargs):
        self._secrets = kwargs["secrets"]
        self.hf_access_token = self._secrets["hf_access_token"]
        self.pipe = None

    def load(self):
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, token=self.hf_access_token
        ).to("cuda")

    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64

    def predict(self, model_input):
        seed = model_input.get("seed")
        prompt = model_input.get("prompt")
        negative_prompt = model_input.get("negative_prompt")
        guidance_scale = model_input.get("guidance_scale", 7.0)
        num_inference_steps = model_input.get("num_inference_steps", 30)
        width = model_input.get("width", 1024)
        height = model_input.get("height", 1024)

        if not seed:
            seed = random.randint(0, MAX_SEED)

        generator = torch.Generator().manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        b64_results = self.convert_to_b64(image)

        return {"data": b64_results}
