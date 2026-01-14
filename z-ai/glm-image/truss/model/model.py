import base64
import logging
import random
from io import BytesIO

import numpy as np
import torch
from diffusers.pipelines.glm_image import GlmImagePipeline
from PIL import Image

logging.basicConfig(level=logging.INFO)
MAX_SEED = np.iinfo(np.int32).max


class Model:
    def __init__(self, **kwargs):
        self.pipe = None
        self.repo_id = "zai-org/GLM-Image"
        self.hf_token = None

    def load(self):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for this model. Please ensure GPU is available."
            )

        logging.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logging.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

        from huggingface_hub import login

        if self.hf_token:
            login(token=self.hf_token)

        self.pipe = GlmImagePipeline.from_pretrained(
            self.repo_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64

    def predict(self, model_input):
        seed = model_input.get("seed")
        prompt = model_input.get("prompt")
        width = model_input.get("width", 1024)
        height = model_input.get("height", 1024)

        if not seed:
            seed = random.randint(0, MAX_SEED)

        generator = torch.Generator(device="cuda").manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=50,
            guidance_scale=1.5,
            generator=generator,
        ).images[0]

        b64_results = self.convert_to_b64(image)
        return {"data": b64_results}
