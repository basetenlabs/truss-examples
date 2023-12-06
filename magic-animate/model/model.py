import base64
import os
import os.path
from io import BytesIO
from typing import Dict

import numpy as np
from huggingface_hub import snapshot_download
from model.demo.animate import MagicAnimate
from PIL import Image

DEFAULT_SEED = 1
DEFAULT_STEPS = 25
DEFAULT_GUIDANCE_SCALE = 7.5
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
BASE64_PREAMBLE = "data:image/png;base64,"
GRID = False
INDIVIDUAL_CLIP_PATH = "/app/demo/outputs/individual.mp4"


class Model:
    def __init__(self, **kwargs):
        self.model = None

    def load(self):
        snapshot_download(
            repo_id="runwayml/stable-diffusion-v1-5",
            local_dir="./stable-diffusion-v1-5",
        )
        snapshot_download(
            repo_id="stabilityai/sd-vae-ft-mse", local_dir="./sd-vae-ft-mse"
        )
        snapshot_download(repo_id="zcxu-eric/MagicAnimate", local_dir="./MagicAnimate")
        self.model = MagicAnimate()

    def mp4_to_base64(self, file_path: str):
        with open(file_path, "rb") as mp4_file:
            binary_data = mp4_file.read()
            base64_data = base64.b64encode(binary_data)
            base64_string = base64_data.decode("utf-8")

        return base64_string

    def base64_to_mp4(self, base64_string, output_file_path):
        binary_data = base64.b64decode(base64_string)
        with open(output_file_path, "wb") as output_file:
            output_file.write(binary_data)

        return output_file_path

    def b64_to_pil(self, b64_str):
        return Image.open(
            BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, "")))
        )

    def predict(self, model_input: Dict) -> Dict:
        reference_image = model_input.get("reference_image")
        motion_sequence = model_input.get("motion_sequence")
        seed = int(model_input.get("seed", DEFAULT_SEED))
        steps = int(model_input.get("steps", DEFAULT_STEPS))
        guidance_scale = float(
            model_input.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
        )
        grid = bool(model_input.get("grid", GRID))

        reference_image_pil = self.b64_to_pil(reference_image)
        reference_image = np.array(
            reference_image_pil.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        )
        motion_sequence_mp4 = self.base64_to_mp4(motion_sequence, "motion_sequence.mp4")

        output_animation_path = self.model(
            reference_image, motion_sequence_mp4, seed, steps, guidance_scale
        )
        output_animation_path = os.path.join(os.getcwd(), output_animation_path)
        print("output animation path: ", output_animation_path)

        individual_clip = self.mp4_to_base64(INDIVIDUAL_CLIP_PATH)

        if grid:
            grid_clip = self.mp4_to_base64(output_animation_path)
            output = {"output": individual_clip, "grid_clip": grid_clip}
        else:
            output = {"output": individual_clip}

        os.remove(motion_sequence_mp4)
        os.remove(output_animation_path)
        os.remove(INDIVIDUAL_CLIP_PATH)

        return output
