import base64
import io
import logging
import os
import random
import subprocess
import tempfile
import time

import numpy as np
import torch
from huggingface_hub import login
from PIL import Image
from torchvision.utils import save_image

from Sana.app.sana_pipeline import SanaPipeline

MAX_SEED = np.iinfo(np.int32).max
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Model:
    def __init__(self, **kwargs):
        # Uncomment the following to get access
        # to various parts of the Truss config.

        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None
        self.hf_access_token = self._secrets.get("hf_access_token")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"device found {self.device}")

    def load(self):
        logging.info("logging into huggingface")
        # we need the hf_access_token to download the models(gemma-2b-it)
        if self.hf_access_token is None:
            raise ValueError("need to set hf_access_token as a baseten secret")
        login(token=self.hf_access_token, add_to_git_credential=False)
        logging.info("loading model(s)")
        # Load model here and assign to self._model.
        logging.info("loading sana pipleline")
        self.sana = SanaPipeline(
            "/packages/Sana/configs/sana_config/1024ms/Sana_600M_img1024.yaml"
        )
        logging.info("loading sana model from hf checkpoint")
        self.sana.from_pretrained(
            "hf://Efficient-Large-Model/Sana_600M_1024px/checkpoints/Sana_600M_1024px_MultiLing.pth"
        )
        logging.info("all done loading models")
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            logging.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed with code {e.returncode}: {e.stderr}")

    def predict(self, model_input):
        start = time.perf_counter()
        prompt = model_input.get("prompt", None)
        if prompt is None:
            raise ValueError("No prompt provided")
        height = model_input.get("height", 1024)
        width = model_input.get("width", 1024)
        guidance_scale = model_input.get("guidance_scale", 5.0)
        pag_guidance_scale = model_input.get("pag_guidance_scale", 2.0)
        num_inference_steps = model_input.get("num_inference_steps", 18)
        seed = model_input.get("seed", random.randint(0, MAX_SEED))
        image = self.sana(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            pag_guidance_scale=pag_guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )

        with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as temp_file:
            temp_file_path = temp_file.name
            save_image(
                image, temp_file_path, nrow=1, normalize=True, value_range=(-1, 1)
            )
            with Image.open(temp_file_path) as img:
                # Convert the image to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr = img_byte_arr.getvalue()

            # Encode the image bytes to base64
        base64_str = base64.b64encode(img_byte_arr).decode("utf-8")
        end = time.perf_counter()
        logging.info(f"Total time taken: {end - start} seconds")
        return {"data": base64_str}
