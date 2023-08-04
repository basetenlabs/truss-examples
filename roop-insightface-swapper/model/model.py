from typing import Any
import base64
import torch
from io import BytesIO
from diffusers import StableDiffusionPipeline
from PIL import Image

from . import swapper

SD_BASE_MODEL_CHECKPOINT = "SG161222/Realistic_Vision_V4.0_noVAE"

BASE64_PREAMBLE = "data:image/png;base64,"

def pil_to_b64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return BASE64_PREAMBLE + str(img_str)[2:-1]


def b64_to_pil(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, ""))))

class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self.pipe = None
        self.swapper = None

    def load(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            SD_BASE_MODEL_CHECKPOINT,
            width=768,
            height=768,
            torch_dtype=torch.float16
        ).to("cuda")
        
        self.swapper = swapper.getFaceSwapModel(f"{str(self._data_dir)}/models/inswapper.onnx")

    def predict(self, model_input: Any) -> Any:
        prompt = model_input.get("prompt")
        negative_prompt = model_input.get("negative_prompt", "")
        source_img = model_input.get("source_img")

        print(source_img)

        # Convert base64 image to PIL image
        source_img = b64_to_pil(source_img)

        base_image = self.pipe(
            prompt,
            negative_prompt = negative_prompt
        ).images[0]
        
        result_image = swapper.process([source_img], base_image, "-1", "-1", self.swapper)

        return {"result": pil_to_b64(result_image)}