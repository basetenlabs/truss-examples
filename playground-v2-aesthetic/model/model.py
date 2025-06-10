import base64
from io import BytesIO
from typing import Dict

import torch
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
)


MODEL_NAME = "playgroundai/playground-v2-1024px-aesthetic"
DEFAULT_SCHEDULER = "DPMSolverMultistep"
DEFAULT_NEGATIVE_PROMPT = None
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 3.0
DEFAULT_SEED = None
SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}


class Model:
    def __init__(self, **kwargs):
        self.pipe = None

    def load(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
            variant="fp16",
        )
        self.pipe.to("cuda")

    def pil_to_b64(self, pil_img):
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def predict(self, model_input: Dict) -> Dict:
        prompt = model_input.get("prompt")
        negative_prompt = model_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        scheduler = model_input.get("scheduler", "DPMSolverMultistep")
        steps = model_input.get("steps", DEFAULT_STEPS)
        guidance_scale = model_input.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
        seed = model_input.get("seed", DEFAULT_SEED)
        generator = None
        if seed:
            generator = torch.Generator("cuda").manual_seed(seed)
        self.pipe.scheduler = SCHEDULERS[scheduler].from_config(
            self.pipe.scheduler.config
        )

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
        base64_image = self.pil_to_b64(image)

        return {"output": base64_image}
