from typing import Any

from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from threading import Thread
from queue import Queue
import logging


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self.pipe = None

    def load(self):
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
        vae = self.pipe.vae
        
        q = Queue() # `latents_callback`` produces, the `inner_cosumer`` consumes
        job_done = object() # signals the processing is done
    
        def latents_callback(i, t, latents):
            outputRs = latents.permute(0, 2, 3, 1)
            k = outputRs.cpu().detach().numpy()
            size = latents.shape[0]
            res = k[0]
            image = Image.fromarray(np.uint8(res)).convert('RGB')
            b64_img = self.convert_to_b64(image)
            q.put(b64_img + "\n")
        
        def generation_task():
            image = self.pipe(prompt=prompt, output_type="latent" if use_refiner else "pil", callback=latents_callback, callback_steps=15).images[0]
            
            if use_refiner:
                image = self.refiner(prompt=prompt, image=image[None, :]).images[0]
            final_image = self.convert_to_b64(image)
            q.put(final_image + "\n")
            q.put(job_done)
        
        thread = Thread(
            target=generation_task,
        )
        thread.start()
        def inner_cosumer():
            import time
            while True:
                time.sleep(.1)
                next_item = q.get(True, timeout=300) # Blocks until an input is available
                if next_item is job_done:
                    break
                yield next_item
            
        return inner_cosumer()
    
