import base64
import time
from io import BytesIO
from typing import Any

import tensorrt as trt
import torch
from cuda import cudart
from diffusers import AutoencoderKL, DiffusionPipeline, DPMSolverMultistepScheduler
from diffusion.trtclip import TRTClip
from diffusion.trtunet import TRTUnet
from huggingface_hub import snapshot_download
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        snapshot_download(
            repo_id="baseten/sdxl-1.0-trt-8.6.1.post1-engine-H100",
            local_dir="/app/data",
        )
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

        cuda_stream = cudart.cudaStreamCreate()[1]
        self.pipe.unet.to(memory_format=torch.channels_last)
        self.pipe.to("cuda")
        self.pipe.unet = _wrap_in_tunet(self.pipe.unet, "engine", cuda_stream)
        self.pipe.text_encoder = _wrap_in_trtclip(
            self.pipe.text_encoder, "engine", cuda_stream
        )
        self.pipe.text_encoder_2 = _wrap_in_trtclip2(
            self.pipe.text_encoder_2, "engine", cuda_stream
        )

        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            vae=self.pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.to("cuda")
        self.refiner.text_encoder_2 = _wrap_in_trtclip2(
            self.refiner.text_encoder_2, "engine_xl_refiner", cuda_stream
        )
        self.refiner.unet = _wrap_in_tunet(
            self.refiner.unet, "engine_xl_refiner", cuda_stream
        )

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


def _wrap_in_tunet(unet, engine_dir_name, stream):
    return _wrap_in_trt(unet, engine_dir_name, "unetxl", TRTUnet, stream)


def _wrap_in_trtclip(clip, engine_dir_name, stream):
    return _wrap_in_trt(clip, engine_dir_name, "clip", TRTClip, stream, is_clip2=False)


def _wrap_in_trtclip2(clip, engine_dir_name, stream):
    return _wrap_in_trt(clip, engine_dir_name, "clip2", TRTClip, stream, is_clip2=True)


def _wrap_in_trt(model, engine_dir_name, engine_name, trt_class, stream, **kwargs):
    model.to("cpu")
    torch.cuda.empty_cache()
    trt_model = trt_class(
        model,
        stream=stream,
        engine_path=f"/app/data/{engine_dir_name}/{engine_name}.trt{trt.__version__}.plan",
        **kwargs,
    )
    trt_model.load()
    return trt_model
