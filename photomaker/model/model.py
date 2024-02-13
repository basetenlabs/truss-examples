import numpy as np
import torch
from huggingface_hub import hf_hub_download
from photomaker import PhotoMakerStableDiffusionXLPipeline
from PIL import Image


class PhotoMakerModel:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        # Download the model from Hugging Face Hub
        model_path = hf_hub_download(
            repo_id="TencentARC/PhotoMaker",
            filename="photomaker-v1.bin",
            repo_type="model",
        )
        self.model = PhotoMakerStableDiffusionXLPipeline.from_pretrained(model_path).to(
            self.device
        )

    def predict(self, input_image: Image) -> Image:
        """
        Takes an input image and produces an output image using the PhotoMakerStableDiffusionXLPipeline model.
        """
        # Ensure the model is loaded
        if self.model is None:
            self.load_model()

        # Convert input image to tensor
        input_tensor = self._prepare_input(input_image)

        # Generate image
        with torch.no_grad():
            generated_image = self.model(input_tensor)["sample"]

        # Convert tensor to PIL Image
        output_image = self._tensor_to_image(generated_image)
        return output_image

    def _prepare_input(self, image: Image) -> torch.Tensor:
        """
        Prepares the input image for the model.
        """
        image = image.convert("RGB")
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        return image_tensor

    def _tensor_to_image(self, tensor: torch.Tensor) -> Image:
        """
        Converts a tensor to a PIL Image.
        """
        tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        tensor = np.clip(tensor, 0, 255).astype(np.uint8)
        image = Image.fromarray(tensor)
        return image


# Example usage
if __name__ == "__main__":
    photomaker_model = PhotoMakerModel()
    input_image = Image.open("path_to_your_input_image.jpg")
    output_image = photomaker_model.predict(input_image)
    output_image.save("path_to_your_output_image.jpg")
