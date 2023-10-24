import base64
import os
from io import BytesIO
from typing import Dict, List

import torch
from diffusers import (DDIMScheduler, DPMSolverMultistepScheduler,
                       EulerDiscreteScheduler, LMSDiscreteScheduler,
                       PNDMScheduler, StableDiffusionPipeline)
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs.get("secrets")
        self.model_id = "stabilityai/stable-diffusion-2-1-base"
        self.model = None
        self.schedulers = None

    def load(self):
        self.model = StableDiffusionPipeline.from_pretrained(
            str(self._data_dir),
            revision="fp16",
            torch_dtype=torch.float16,
        ).to("cuda")

        schedulers = [
            ("ddim", DDIMScheduler),
            ("dpm", DPMSolverMultistepScheduler),
            ("euler", EulerDiscreteScheduler),
            ("lms", LMSDiscreteScheduler),
            ("pndm", PNDMScheduler),
        ]

        self.schedulers = self.load_schedulers(schedulers)
        self.model.scheduler = self.schedulers["ddim"]

    def load_schedulers(self, schedulers_to_add):
        schedulers = {}
        for name, scheduler in schedulers_to_add:
            schedulers[name] = scheduler.from_config(self.model.scheduler.config)

        print(schedulers)
        return schedulers

    def preprocess(self, request):
        scheduler = request.pop("scheduler", "ddim")
        self.model.scheduler = self.schedulers[scheduler]

        return request

    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image[0].save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64

    @torch.inference_mode()
    def predict(self, request: Dict) -> Dict[str, List]:
        results = []
        random_seed = int.from_bytes(os.urandom(2), "big")
        generator = torch.Generator("cuda").manual_seed(
            request.get("seed", random_seed)
        )
        try:
            output = self.model(
                **request,
                generator=generator,
                return_dict=False,
            )
            b64_results = self.convert_to_b64(output[0])
            results = results + [b64_results]

        except Exception as exc:
            return {"status": "error", "data": None, "message": str(exc)}

        return {"status": "success", "data": results, "message": None}
