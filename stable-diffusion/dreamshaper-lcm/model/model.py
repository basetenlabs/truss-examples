from diffusers import DiffusionPipeline
import torch
from PIL import Image
from io import BytesIO
import base64

class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        self._model = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", custom_pipeline="latent_consistency_txt2img", custom_revision="main").to(torch_device="cuda", torch_dtype=torch.float32)

    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64

    def predict(self, model_input):
        prompt = model_input["prompt"]
        num_inference_steps = model_input.get("num_inference_steps", 4)

        image = self._model(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0, lcm_origin_steps=50, output_type="pil").images[0]
        img_b64 = self.convert_to_b64(image)

        return {"image": img_b64}
