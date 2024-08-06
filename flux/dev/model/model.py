import base64
import logging
import math
import random
import subprocess
import time
from io import BytesIO

import numpy as np
import torch
from diffusers import FluxPipeline
from PIL import Image

logging.basicConfig(level=logging.INFO)
MAX_SEED = np.iinfo(np.int32).max


class Model:
    def __init__(self, **kwargs):
        self._secrets = kwargs["secrets"]
        self.model_name = kwargs["config"]["model_metadata"]["repo_id"]
        self.hf_access_token = self._secrets["hf_access_token"]
        self.pipe = None

    def load(self):
        self.pipe = FluxPipeline.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, token=self.hf_access_token
        ).to("cuda")
        # self.pipe.enable_model_cpu_offload()
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            logging.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed with code {e.returncode}: {e.stderr}")

    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64

    def predict(self, model_input):
        start = time.perf_counter()
        seed = model_input.get("seed")
        prompt = model_input.get("prompt")
        prompt2 = model_input.get("prompt2")
        max_sequence_length = model_input.get("max_sequence_length", 512)
        guidance_scale = model_input.get("guidance_scale", 7.5)
        num_inference_steps = model_input.get(
            "num_inference_steps", 50
        )  # schnell is timestep-distilled
        width = model_input.get("width", 1024)
        height = model_input.get("height", 1024)
        if not seed:
            seed = random.randint(0, MAX_SEED)
        if len(prompt.split()) > max_sequence_length:
            logging.warning(
                f"Input longer than {max_sequence_length} tokens, truncating"
            )
            tokens = prompt.split()
            prompt = " ".join(tokens[: min(len(tokens), max_sequence_length)])
        if prompt2 and len(prompt2.split()) > max_sequence_length:
            logging.warning(
                f"Input prompt2 longer than {max_sequence_length} tokens, truncating"
            )
            tokens = prompt2.split()
            prompt2 = " ".join(tokens[: min(len(tokens), max_sequence_length)])
        generator = torch.Generator().manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            prompt_2=prompt2,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            output_type="pil",
            generator=generator,
        ).images[0]

        b64_results = self.convert_to_b64(image)

        end = time.perf_counter()
        logging.info(f"Total time taken: {end - start} seconds")
        return {"data": b64_results}
