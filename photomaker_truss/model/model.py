import base64
from io import BytesIO

import torch
from huggingface_hub import hf_hub_download
from photomaker.pipeline import (
    PhotoMakerStableDiffusionXLPipeline,  # This path assumes the 'photomaker' structure
)
from PIL import Image


class Model:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None

    def load(self):
        model_path = hf_hub_download(
            repo_id="TencentARC/PhotoMaker",
            filename="photomaker-v1.bin",
            use_auth_token=True,
        )  # Adding use_auth_token as best practice
        self.pipeline = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

    def predict(self, base64_image: str) -> str:
        image_bytes = b64decode(base64_image)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Assuming a correct handling in the pipeline for raw images
        with torch.no_grad():
            prediction = self.pipeline.predict(image=image)

        # Convert the prediction image to base64
        buffer = BytesIO()
        prediction.save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return encoded_image
