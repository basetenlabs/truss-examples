from typing import Any

from diffusers import DiffusionPipeline, AutoencoderKL
import torch
import base64
from io import BytesIO
from PIL import Image
import time
import functools


# Good notebook to learn how to use diffusers + LoRA:
# https://colab.research.google.com/gist/sayakpaul/109b7e792c64514fb3c057ecff4145ff/scratchpad.ipynb#scrollTo=GreOMxAcvlm8

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

        self.pipe.to('cuda')
        
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.pipe.text_encoder_2,
            vae=self.pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.to("cuda")
        self.prev_lora = None

    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64

    def predict(self, model_input: Any) -> Any:
        prompt = model_input.pop("prompt")
        negative_prompt = model_input.pop("negative_prompt", None)
        target_size = model_input.pop("size", 1024)
        use_refiner = model_input.pop("use_refiner", True)
        high_noise_frac = model_input.pop("high_noise_frac", 0.8)
        num_inference_steps = model_input.pop("num_inference_steps", 30)

        lora = model_input.pop("lora", None)
        print(f"Loading LoRA weights: {lora}")

        # example schema: 
        # {"repo_id": "nerijs/pixel-art-xl", "weights": "pixel-art-xl.safetensors"}

        # Note: if LoRA is None, the default behavior is to use the last loaded weights (or no weights if none were loaded)
        if lora is not None:
            use_refiner = False
            if lora != self.prev_lora:
                self.prev_lora = lora
                self.pipe.unload_lora_weights()
                self.pipe.load_lora_weights(
                    lora.get("repo_id", None),
                    weight_name=lora.get("weights", None),
                )
                print("Loaded LoRA weights!")
            else:
                print("Using previously loaded LoRA weights.")
        
        with torch.inference_mode():
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                denoising_end=high_noise_frac if use_refiner else 1.0,
                output_type="latent" if use_refiner else "pil",
                target_size=(target_size, target_size)
            ).images[0]
            if use_refiner:
                image = self.refiner(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    denoising_start=high_noise_frac,
                    image=image[None, :],
                    target_size=(target_size, target_size)
                ).images[0]
        b64_results = self.convert_to_b64(image)

        return {"result": b64_results}
