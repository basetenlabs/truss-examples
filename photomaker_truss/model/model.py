from base64 import b64decode, b64encode
from io import BytesIO

import torch
from huggingface_hub import hf_hub_download
from photomaker import PhotoMakerStableDiffusionXLPipeline
from PIL import Image


class Model:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None

    def load(self):
        model_path = hf_hub_download(
            repo_id="TencentARC/PhotoMaker",
            filename="photomaker-v1.bin",
            repo_type="model",
        )
        self.pipeline = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

    def predict(self, base64_image: str) -> str:
        image_bytes = b64decode(base64_image)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Preprocessing image for the model
        input_image = self.pipeline.preprocess(image)

        # Ensure input is on the correct device
        input_image = input_image.to(self.device)

        # Generating prediction
        with torch.no_grad():
            output_images = self.pipeline([input_image])

        # Assuming the first image in the batch is our desired output
        output_image = output_images.images[0]

        # Encode the output image to base64
        buffer = BytesIO()
        output_image.save(buffer, format="JPEG")
        output_base64 = b64encode(buffer.getvalue()).decode("utf-8")

        return output_base64
