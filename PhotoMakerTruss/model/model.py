import base64
from io import BytesIO

import torch
from huggingface_hub import hf_hub_download
from photomaker import PhotoMakerStableDiffusionXLPipeline
from PIL import Image


class Model:
    def __init__(self):
        self.pipe = None

    def load(self):
        # Download the PhotoMakerStableDiffusionXLPipeline model
        photomaker_path = hf_hub_download(
            repo_id="TencentARC/PhotoMaker",
            filename="photomaker-v1.bin",
            repo_type="model",
        )
        self.pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            photomaker_path, torch_dtype=torch.float16, use_auth_token=True
        ).to("cuda")

    def predict(self, prompt, negative_prompt, num_inference_steps, start_merge_step):
        # Generate images with the pipeline
        images = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            start_merge_step=start_merge_step,
            output_type="pil",
        ).images

        # Convert the PIL image to base64 to return
        buffered = BytesIO()
        images[0].save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return img_str

    def convert_to_pil(self, image_b64):
        # Convert base64 string to PIL Image
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data))
        return image
