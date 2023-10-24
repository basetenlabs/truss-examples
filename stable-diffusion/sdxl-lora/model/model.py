from typing import Any

from diffusers import DiffusionPipeline, AutoencoderKL
import torch
import base64
from io import BytesIO
from PIL import Image
import time
import functools


class Model:
    def __init__(self, **kwargs) -> None:
        self.pipe = None

    def load(self):
        # Load model here and assign to self._model.
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )

        self.pipe.load_lora_weights(
            "stabilityai/stable-diffusion-xl-base-1.0",
            weight_name="sd_xl_offset_example-lora_1.0.safetensors",
        )

        # self.pipe.load_lora_weights(
        #     "minimaxir/sdxl-wrong-lora"
        # )

        self.pipe.to("cuda")

        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.pipe.text_encoder_2,
            vae=self.pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.to("cuda")

    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64

    def predict(self, model_input: Any) -> Any:
        prompt = model_input.pop("prompt")
        target_size = model_input.pop("size", 1024)
        use_refiner = model_input.pop("use_refiner", True)
        high_noise_frac = model_input.pop("high_noise_frac", 0.8)
        num_inference_steps = model_input.pop("num_inference_steps", 30)

        with torch.inference_mode():
            image = self.pipe(
                prompt=prompt,
                negative_prompt="anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
                num_inference_steps=num_inference_steps,
                denoising_end=high_noise_frac if use_refiner else 1.0,
                output_type="latent" if use_refiner else "pil",
                target_size=(target_size, target_size),
            ).images[0]
            if use_refiner:
                image = self.refiner(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    denoising_start=high_noise_frac,
                    image=image[None, :],
                    target_size=(target_size, target_size),
                ).images[0]
        b64_results = self.convert_to_b64(image)

        return {"result": b64_results}
