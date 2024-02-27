import base64
from io import BytesIO
from typing import Dict, List

import torch
from photomaker import PhotoMakerStableDiffusionXLPipeline
from PIL import Image


class Model:
    def __init__(self):
        self.pipe = None

    def load(self):
        # Initialize the PhotoMakerStableDiffusionXLPipeline with the appropriate configuration
        self.pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            "TencentARC/PhotoMaker",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.pipe.to("cuda")

    def pil_to_b64(self, pil_img: Image) -> str:
        """Convert PIL images to base64 strings."""
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def preprocess(self, model_input: Dict) -> Dict:
        """Preprocess the input if necessary."""
        # This is a placeholder for any preprocessing steps you might need,
        # such as resizing images or normalizing data.
        return model_input

    def postprocess(self, model_output: List[Image]) -> List[str]:
        """Postprocess the output of the model."""
        # Convert images to base64 strings for easier handling in web applications.
        return [self.pil_to_b64(img) for img in model_output]

    def predict(self, model_input: Dict) -> Dict:
        """Run inference using the pipeline based on the 'photomaker_demo.ipynb' notebook."""
        processed_input = self.preprocess(model_input)
        prompt = processed_input.get("prompt")

        # Running the prediction
        output_images = self.pipe(
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
        ).images

        # Postprocessing the output images
        base64_images = self.postprocess(output_images)

        return {"output": base64_images}
