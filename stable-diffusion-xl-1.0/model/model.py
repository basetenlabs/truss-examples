from typing import Any

from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO
from PIL import Image




class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self.pipe = None

    def load(self):
        # Load model here and assign to self._model.
        
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        self.pipe.to("cuda")
    
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.pipe.text_encoder_2,
            vae=self.pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.to("cuda")

    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64
    
    def predict(self, model_input: Any) -> Any:
        prompt = model_input.pop("prompt")
        use_refiner = model_input.pop("use_refiner", False)
        batch_size = model_input.pop("batch_size", 4)



        images = self.pipe(prompt=[prompt] * batch_size, num_inference_steps=25).images # note: you can decrease num steps if using refiner

        negative_prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off"

        if use_refiner:
            images = self.refiner(prompt=[prompt] * batch_size, negative_prompt=[negative_prompt] * batch_size, image=images).images # this is a list
                
        
        b64_results = [self.convert_to_b64(image) for image in images]

        return {"status": "success", "data": b64_results, "message": None}
