import base64
from io import BytesIO
from typing import Dict

import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image


class Model:
    def __init__(self, **kwargs):
        self.model = None

    def load(self):
        self.model = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        )
        self.model.to("cuda")

    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64

    def predict(self, model_input: Dict) -> Dict:
        prompt = model_input.get("prompt")
        output_image = self.model(
            prompt=prompt, num_inference_steps=1, guidance_scale=0.0
        ).images[0]
        return {"result": self.convert_to_b64(output_image)}
