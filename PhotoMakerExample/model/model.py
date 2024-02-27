import torch
from huggingface_hub import hf_hub_download
from photomaker import PhotoMakerStableDiffusionXLPipeline


class PhotoMakerModel:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.float16
            if torch.cuda.is_available()
            and torch.cuda.get_device_properties(0).total_memory > 4e9
            else torch.float32
        )

    def load_model(self):
        photomaker_path = hf_hub_download(
            repo_id="TencentARC/PhotoMaker",
            filename="photomaker-v1.bin",
            repo_type="model",
        )
        self.model = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            photomaker_path, torch_dtype=self.dtype
        ).to(self.device)

    def predict(self, image):
        if self.model is None:
            self.load_model()
        transformed_image = self.model(image)
        return transformed_image


# Example usage
if __name__ == "__main__":
    photo_maker = PhotoMakerModel()
    photo_maker.load_model()
    # Assuming `input_image` is a PIL Image or a path to an image file
    # output_image = photo_maker.predict(input_image)
    # output_image.save("transformed_image.png")
