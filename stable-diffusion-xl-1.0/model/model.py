from diffusers import DiffusionPipeline, AutoencoderKL, DPMSolverMultistepScheduler
import torch
import base64
from PIL import Image
from io import BytesIO
from typing import Any

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
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, use_karras_sigmas=True)
        
        # DPM++ 2M SDE Karras (for 30+ steps, when speed doesn't matter)
        # self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)

        self.pipe.unet.to(memory_format=torch.channels_last)
        self.pipe.to('cuda')

        # if you are using a LoRA, comment out anything related to xformers
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
        self.refiner.enable_xformers_memory_efficient_attention()

    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64
    
    def predict(self, model_input: Any) -> Any:
        prompt = model_input.pop("prompt")
        use_refiner = model_input.pop("use_refiner", True)
        num_inference_steps = model_input.pop("num_inference_steps", 30)
        denoising_frac = model_input.pop("denoising_frac", 0.8)
        end_cfg_frac = model_input.pop("end_cfg_frac", 0.4)
        guidance_scale = model_input.pop("guidance_scale", 5.0)

        image = self.pipe(prompt=prompt,
                          end_cfg = end_cfg_frac,
                          num_inference_steps=num_inference_steps,
                          denoising_end=denoising_frac, 
                          guidance_scale=guidance_scale,
                          output_type="latent" if use_refiner else "pil").images[0]
        if use_refiner:
            self.refiner.scheduler = self.pipe.scheduler
            image = self.refiner(prompt=prompt,
                                 end_cfg = end_cfg_frac, 
                                 num_inference_steps=num_inference_steps, 
                                 denoising_start=denoising_frac,
                                 image=image[None, :]).images[0]
        b64_results = self.convert_to_b64(image)

        return {"status": "success", "data": b64_results, "message": None}
