import base64
import time
from io import BytesIO
from typing import Any

import torch
from diffusers import AutoencoderKL, DiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
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

        # DPM++ 2M Karras (for < 30 steps, when speed matters)
        # self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True)

        # DPM++ 2M SDE Karras (for 30+ steps, when speed doesn't matter)
        # self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)

        self.pipe.unet.to(memory_format=torch.channels_last)
        # self.pipe.unet = torch.compile(self.pipe.unet, mode="max-autotune", fullgraph=True)
        self.pipe.to("cuda")
        self.pipe.enable_xformers_memory_efficient_attention()

        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.pipe.text_encoder_2,
            vae=self.pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.to("cuda")
        # self.refiner.unet = torch.compile(self.refiner.unet, mode="max-autotune", fullgraph=True)
        self.refiner.enable_xformers_memory_efficient_attention()

        # image = self.pipe(prompt="a golden retriever", num_inference_steps=30, output_type="pil").images[0]
        # image = self.refiner(prompt="a golden retriever", num_inference_steps=30, output_type="pil").images[0]

    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64

    def predict(self, model_input: Any) -> Any:
        prompt = model_input.pop("prompt")
        negative_prompt = model_input.pop("negative_prompt", None)
        width = model_input.pop("width", 1024)
        height = model_input.pop("height", 1024)
        use_refiner = model_input.pop("use_refiner", True)
        num_inference_steps = model_input.pop("num_inference_steps", 30)
        denoising_frac = model_input.pop("denoising_frac", 0.8)
        end_cfg_frac = model_input.pop("end_cfg_frac", 0.4)
        guidance_scale = model_input.pop("guidance_scale", 7.5)
        seed = model_input.pop("seed", None)

        scheduler = model_input.pop(
            "scheduler", None
        )  # Default: EulerDiscreteScheduler (works pretty well)

        # See schedulers: https://huggingface.co/docs/diffusers/api/schedulers/overview
        if scheduler == "DPM++ 2M":
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
        elif scheduler == "DPM++ 2M Karras":
            # DPM++ 2M Karras (for < 30 steps, when speed matters)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config, use_karras_sigmas=True
            )
        elif scheduler == "DPM++ 2M SDE Karras":
            # DPM++ 2M SDE Karras (for 30+ steps, when speed doesn't matter)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                algorithm_type="sde-dpmsolver++",
                use_karras_sigmas=True,
            )

        generator = None
        if seed is not None:
            torch.manual_seed(seed)
            generator = [torch.Generator(device="cuda").manual_seed(seed)]

        if not use_refiner:
            denoising_frac = 1.0

        start_time = time.time()
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            generator=generator,
            end_cfg=end_cfg_frac,
            num_inference_steps=num_inference_steps,
            denoising_end=denoising_frac,
            guidance_scale=guidance_scale,
            output_type="latent" if use_refiner else "pil",
        ).images[0]
        scheduler = self.pipe.scheduler
        if use_refiner:
            self.refiner.scheduler = scheduler
            image = self.refiner(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=generator,
                end_cfg=end_cfg_frac,
                num_inference_steps=num_inference_steps,
                denoising_start=denoising_frac,
                guidance_scale=guidance_scale,
                image=image[None, :],
            ).images[0]

        b64_results = self.convert_to_b64(image)
        end_time = time.time() - start_time

        print(f"Time: {end_time:.2f} seconds")

        return {"status": "success", "data": b64_results, "time": end_time}
