import os

import torch
from diffusers import EulerDiscreteScheduler
from photomaker import PhotoMakerStableDiffusionXLPipeline


class PhotoMakerModel:
    def __init__(self, base_model_path, photomaker_checkpoint_path, device="cuda"):
        self.device = device
        self.base_model_path = base_model_path
        self.photomaker_checkpoint_path = photomaker_checkpoint_path
        self.pipe = None
        self.load_model()

    def load_model(self):
        self.pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.device)

        self.pipe.load_photomaker_adapter(
            os.path.dirname(self.photomaker_checkpoint_path),
            subfolder="",
            weight_name=os.path.basename(self.photomaker_checkpoint_path),
            trigger_word="img",
        )

        self.pipe.scheduler = EulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

    def predict(
        self,
        prompt,
        negative_prompt,
        input_id_images,
        num_images_per_prompt=1,
        num_inference_steps=50,
        start_merge_step=10,
    ):
        generator = torch.Generator(device=self.device).manual_seed(42)
        images = self.pipe(
            prompt=prompt,
            input_id_images=input_id_images,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=num_inference_steps,
            start_merge_step=start_merge_step,
            generator=generator,
        ).images[0]
        return images


if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_path = "path/to/base/model"
    photomaker_checkpoint_path = "path/to/photomaker/checkpoint"
    photomaker_model = PhotoMakerModel(
        base_model_path, photomaker_checkpoint_path, device=device
    )

    # Define your prompts, negative prompts, and input ID images
    prompt = "a half-body portrait of a man img wearing sunglasses in an Iron Man suit, best quality"
    negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth, grayscale"
    input_id_images = []  # Load your input ID images here

    # Generate an image
    generated_image = photomaker_model.predict(prompt, negative_prompt, input_id_images)
    # Save or display your generated_image as needed
