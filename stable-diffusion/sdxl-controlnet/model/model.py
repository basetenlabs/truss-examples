from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
)
from diffusers.utils import load_image
from PIL import Image
import torch
import numpy as np
import cv2
import base64
from io import BytesIO

BASE64_PREAMBLE = "data:image/png;base64,"


def pil_to_b64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def b64_to_pil(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, ""))))


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0",
            torch_dtype=torch.float16,
        )
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
        )
        pipe.enable_model_cpu_offload()
        self._model = pipe

    def predict(self, model_input):
        prompt = model_input.pop("prompt")
        image = model_input.pop("image")

        image = b64_to_pil(image)
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)

        negative_prompt = "low quality, bad quality, sketches"
        controlnet_conditioning_scale = 0.5  # recommended for good generalization

        images = self._model(
            prompt,
            num_inference_steps=30,
            negative_prompt=negative_prompt,
            image=image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        ).images

        return {"result": pil_to_b64(images[0])}
