import torch
import base64
import qrcode
from PIL import Image
from io import BytesIO
from typing import Dict


from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler
)


class Model:
    def __init__(self, **kwargs):
        self.model = None

    def create_code(self, content: str):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=16,
            border=0,
        )
        qr.add_data(content)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")

        offset_min = 8 * 16
        w, h = img.size
        w = (w + 255 + offset_min) // 256 * 256
        h = (h + 255 + offset_min) // 256 * 256
        bg = Image.new('L', (w, h), 128)

        coords = ((w - img.size[0]) // 2 // 16 * 16,
                  (h - img.size[1]) // 2 // 16 * 16)
        bg.paste(img, coords)
        return bg

    def pil_to_b64(self, pil_img):
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def load(self):
        controlnet = ControlNetModel.from_pretrained(
            "monster-labs/control_v1p_sd15_qrcode_monster", torch_dtype=torch.float16
        )

        self.model = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.model.enable_xformers_memory_efficient_attention()

    def predict(self, request: Dict) -> Dict:
        qr_code_content = request.get("qr_code_content")
        prompt = request.get("prompt")
        negative_prompt = request.get("negative_prompt", "")
        guidance_scale = request.get("guidance_scale", 7.5)
        controlnet_conditioning_scale = request.get("condition_scale", 1.2)
        seed = request.get("seed", -1)
        sampler = request.get("sampler", "Euler a")
        num_inference_steps = request.get("inference_steps", 40)

        SAMPLER_MAP = {
            "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
            "DPM++ Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True),
            "Heun": lambda config: HeunDiscreteScheduler.from_config(config),
            "Euler a": lambda config: EulerAncestralDiscreteScheduler.from_config(config),
            "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
            "DDIM": lambda config: DDIMScheduler.from_config(config),
            "DEIS": lambda config: DEISMultistepScheduler.from_config(config),
        }

        self.model.scheduler = SAMPLER_MAP[sampler](self.model.scheduler.config)
        generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()
        qrcode_image = self.create_code(qr_code_content)
        out = self.model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=qrcode_image,
            width=qrcode_image.width,
            height=qrcode_image.height,
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        b64_img = self.pil_to_b64(out.images[0])

        return {"result": b64_img}
