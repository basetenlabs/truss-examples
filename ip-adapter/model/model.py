import base64
from io import BytesIO

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

BASE64_PREAMBLE = "data:image/png;base64,"


def pil_to_b64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def b64_to_pil(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, ""))))


class Model:
    def __init__(self, **kwargs):
        self.pipeline = None

    def load(self):
        # You can also use an SDXL pipeline with an IP adapter for SDXL
        # be sure to see all your optiosn here:
        # https://huggingface.co/h94/IP-Adapter
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        )
        self.pipeline.to("cuda")
        self.pipeline.load_ip_adapter(
            "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin"
        )

    # more examples here: https://github.com/huggingface/diffusers/pull/5713
    def predict(self, model_input):
        input_image = model_input["image"]
        image = b64_to_pil(input_image)

        # generator = torch.Generator(device="cpu").manual_seed(33)
        image = self.pipeline(
            prompt="best quality, high quality",
            ip_adapter_image=image,
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            num_inference_steps=50,
            # generator=generator,
        ).images[0]

        output_image = pil_to_b64(image)

        return {"result": output_image}
