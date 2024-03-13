import base64
from io import BytesIO

import torch
from photomaker import EulerDiscreteScheduler, PhotoMakerStableDiffusionXLPipeline
from PIL import Image


class Model:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None

    def load(self):
        self.pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V3.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.device)

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.fuse_lora()

    def predict(self, input_dict):
        base64_img = input_dict.get("img_base64")
        prompt = input_dict.get("prompt")

        img = self._base64_to_img(base64_img)
        processed_image = self.pipe.preprocess(img)

        predictions = self.pipe(prompt=prompt, images=processed_image)
        output_images = predictions.images

        results = self._img_to_base64(output_images[0])

        return {"image": results}

    def _base64_to_img(self, base64_string):
        img_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(img_data)).convert("RGB")
        return img

    def _img_to_base64(self, img):
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
