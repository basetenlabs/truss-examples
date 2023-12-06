from diffusers import DiffusionPipeline
import torch
from PIL import Image
import base64
from io import BytesIO

class Model:
    def __init__(self, **kwargs):
        self.pipe = None

    def load(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            "playgroundai/playground-v2-1024px-aesthetic",
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
            variant="fp16"
        )
        self.pipe.to("cuda")

    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64

    def predict(self, model_input):
        prompt = model_input["prompt"]
        guidance_scale = model_input.get("guidance_scale", 3.0)
        num_inference_steps = model_input.get("num_inference_steps", 25)

        image  = self.pipe(prompt=prompt, guidance_scale=guidance_scale, num_inference_steps=num_inference_steps).images[0]
        return {"result": self.convert_to_b64(image)}
