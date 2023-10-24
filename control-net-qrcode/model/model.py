import base64
from io import BytesIO
from typing import Dict

import qrcode
import torch
from diffusers import (ControlNetModel, DDIMScheduler, DEISMultistepScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       HeunDiscreteScheduler,
                       StableDiffusionControlNetPipeline)
from PIL import Image
from PIL.Image import LANCZOS

BASE64_PREAMBLE = "data:image/png;base64,"


class Model:
    def __init__(self, **kwargs):
        self.model = None

    def resize_image(self, pil_image, width=768, height=768):
        input_image = pil_image.convert("RGB")
        image_width, image_height = input_image.size
        k = float(min(width, height)) / min(image_height, image_width)
        image_height *= k
        image_width *= k
        image_height = int(round(image_height / 64.0)) * 64
        image_width = int(round(image_width / 64.0)) * 64
        img = input_image.resize((image_width, image_height), resample=LANCZOS)
        return img

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
        image_width, image_height = img.size
        image_width = (image_width + 255 + offset_min) // 256 * 256
        image_height = (image_height + 255 + offset_min) // 256 * 256
        bg = Image.new("L", (image_width, image_height), 128)

        coords = (
            (image_width - img.size[0]) // 2 // 16 * 16,
            (image_height - img.size[1]) // 2 // 16 * 16,
        )
        bg.paste(img, coords)
        return bg

    def pil_to_b64(self, pil_img):
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def b64_to_pil(self, b64_str):
        return Image.open(
            BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, "")))
        )

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
        qr_code_content = request.get("qr_code_content", None)
        prompt = request.get("prompt", None)
        mask = request.get("mask", None)

        if not prompt:
            raise Exception("prompt is required for this model")

        negative_prompt = request.get("negative_prompt", "")
        guidance_scale = request.get("guidance_scale", 7.5)
        controlnet_conditioning_scale = request.get("condition_scale", 1.2)
        seed = request.get("seed", -1)
        sampler = request.get("sampler", "Euler a")
        num_inference_steps = request.get("inference_steps", 40)

        SAMPLER_MAP = {
            "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(
                config, use_karras=True, algorithm_type="sde-dpmsolver++"
            ),
            "DPM++ Karras": lambda config: DPMSolverMultistepScheduler.from_config(
                config, use_karras=True
            ),
            "Heun": lambda config: HeunDiscreteScheduler.from_config(config),
            "Euler a": lambda config: EulerAncestralDiscreteScheduler.from_config(
                config
            ),
            "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
            "DDIM": lambda config: DDIMScheduler.from_config(config),
            "DEIS": lambda config: DEISMultistepScheduler.from_config(config),
        }

        self.model.scheduler = SAMPLER_MAP[sampler](self.model.scheduler.config)
        generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()
        if qr_code_content:
            image = self.create_code(qr_code_content)
        elif mask:
            pil_image = self.b64_to_pil(mask)
            image = self.resize_image(pil_image)
        else:
            raise Exception("qr_code_content or mask is required")

        out = self.model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            width=image.width,
            height=image.height,
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(controlnet_conditioning_scale),
            generator=generator,
            num_inference_steps=num_inference_steps,
        )

        b64_img = self.pil_to_b64(out.images[0])

        return {"result": b64_img}
