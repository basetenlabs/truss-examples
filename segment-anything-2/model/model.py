# Prediction interface for Cog ⚙️
# https://cog.run/python

import base64
import gc
import os
import subprocess
import sys
import time
from io import BytesIO
from typing import Callable, Dict, List

import cv2
import httpx
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(1, "data")

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

DEVICE = "cuda"
MODEL_DIR = "checkpoints"
BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
checkpoints = {
    "sam2_hiera_tiny.pt": f"{BASE_URL}sam2_hiera_tiny.pt",
    "sam2_hiera_small.pt": f"{BASE_URL}sam2_hiera_small.pt",
    "sam2_hiera_base_plus.pt": f"{BASE_URL}sam2_hiera_base_plus.pt",
    "sam2_hiera_large.pt": f"{BASE_URL}sam2_hiera_large.pt",
}


class Model:
    def __init__(self, **kwargs):
        """Load the model into memory to make running multiple predictions efficient"""
        self._data_dir = kwargs["data_dir"]
        print(self._data_dir)
        self.model_files = [
            # "sam2_hiera_base_plus.pt",
            # "sam2_hiera_large.pt",
            "sam2_hiera_small.pt",
            # "sam2_hiera_tiny.pt",
        ]
        # models are built into the truss itself

        self.model_configs = {
            "tiny": (
                "sam2_hiera_t.yaml",
                f"{self._data_dir}/checkpoints/sam2_hiera_tiny.pt",
            ),
            "small": (
                "sam2_hiera_s.yaml",
                f"{self._data_dir}/checkpoints/sam2_hiera_small.pt",
            ),
            "base": (
                "sam2_hiera_b+.yaml",
                f"{self._data_dir}/checkpoints/sam2_hiera_base_plus.pt",
            ),
            "large": (
                "sam2_hiera_l.yaml",
                f"{self._data_dir}/checkpoints/sam2_hiera_large.pt",
            ),
        }

        self.model_cfg, self.sam2_checkpoint = self.model_configs["small"]
        functions = [self.load, self.predict]
        kwargs_list = [
            {},
            {
                "model_input": {
                    "image": "https://replicate.delivery/pbxt/LMbGi83qiV3QXR9fqDIzTl0P23ZWU560z1nVDtgl0paCcyYs/cars.jpg"
                }
            },
        ]

    def download_file(self, url, filename):
        try:
            print(f"Downloading {filename} checkpoint...")
            with httpx.stream("GET", url) as response:
                response.raise_for_status()  # Raise an error for unsuccessful status codes
                # make sure its stored in checkpoints directory
                os.makedirs(
                    os.path.dirname(f"{self._data_dir}/checkpoints"), exist_ok=True
                )
                filename = f"{self._data_dir}/checkpoints/{filename}"
                # make sure its stored in checkpoints directory
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "wb") as file:
                    for chunk in response.iter_bytes():
                        file.write(chunk)
            print(f"Successfully downloaded {filename}.")
        except httpx.HTTPStatusError as e:
            print(f"Failed to download checkpoint from {url}: {e}")
            exit(1)

    def load(self):
        # run pip freeze
        os.system("pip freeze")
        os.system(f"pip install --no-build-isolation -e data")
        # Download checkpoint
        for model in self.model_files:
            self.download_file(checkpoints.get(model), model)
        # Load model here and assign to self._model.
        self.sam2 = build_sam2(
            self.model_cfg,
            self.sam2_checkpoint,
            device="cuda",
            apply_postprocessing=False,
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(self.sam2)

        # Enable bfloat16 and TF32 for better performance
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with code {e.returncode}: {e.stderr}")

    def predict(self, model_input):
        # Run model inference here
        image = model_input.get("image")  # assuming image is a url
        points_per_side = model_input.get("points_per_side", 32)
        pred_iou_thresh = model_input.get("pred_iou_thresh", 0.88)
        stability_score_thresh = model_input.get("stability_score_thresh", 0.95)
        use_m2m = model_input.get("use_m2m", True)
        response = httpx.get(image)
        input_image = Image.open(BytesIO(response.content))
        input_image = np.array(input_image.convert("RGB"))

        # Configure the mask generator
        self.mask_generator.points_per_side = points_per_side
        self.mask_generator.pred_iou_thresh = pred_iou_thresh
        self.mask_generator.stability_score_thresh = stability_score_thresh
        self.mask_generator.use_m2m = use_m2m

        # Generate masks
        masks = self.mask_generator.generate(input_image)

        # Generate and save combined colored mask
        b64_results = self.return_combined_mask(input_image, masks)

        # Generate and save individual black and white masks
        individual_mask_paths = self.return_individual_masks(masks)
        # create a list of b64_results and individual_mask_paths
        b64_results = [b64_results] + individual_mask_paths
        del masks
        torch.cuda.empty_cache()
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with code {e.returncode}: {e.stderr}")
        return {"status": "success", "masks": b64_results}

    def return_combined_mask(self, input_image, masks):
        """
        Generates a combined mask image from the given input image and masks and returns a base64 encoded webp image
        """
        buffer = BytesIO()
        plt.figure(figsize=(20, 20))
        plt.imshow(input_image)
        self.show_anns(masks)
        plt.axis("off")
        plt.savefig(buffer, format="webp", bbox_inches="tight", pad_inches=0)
        plt.close()
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        return img_base64

    def return_individual_masks(self, masks):
        individual_mask_strings = []
        for i, mask in enumerate(masks):
            mask_image = mask["segmentation"].astype(np.uint8) * 255

            # Convert the image to WebP and encode it in base64
            buffer = BytesIO()
            Image.fromarray(mask_image).save(buffer, format="WEBP")
            base64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

            individual_mask_strings.append(base64_string)

        return individual_mask_strings

    def show_anns(self, anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones(
            (
                sorted_anns[0]["segmentation"].shape[0],
                sorted_anns[0]["segmentation"].shape[1],
                4,
            )
        )
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann["segmentation"]
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask
            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

        ax.imshow(img)
