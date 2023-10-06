"""
The `Model` class is an interface between the ML model that you're packaging and the model
server that you're running it on.

The main methods to implement here are:
* `load`: runs exactly once when the model server is spun up or patched and loads the
   model onto the model server. Include any logic for initializing your model, such
   as downloading model weights and loading the model into memory.
* `predict`: runs every time the model server is called. Include any logic for model
  inference and return the model output.

See https://truss.baseten.co/quickstart for more.
"""

import imghdr
import io
import random
import time
import base64
from collections import defaultdict
import cv2
import numpy as np
import torch
from PIL import Image
from loguru import logger

from lama_cleaner.model.utils import torch_gc
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config
from lama_cleaner.helper import (
    load_img,
    resize_max_size,
    pil_to_bytes,
)


def get_image_ext(img_bytes):
    w = imghdr.what("", img_bytes)
    if w is None:
        w = "jpeg"
    return w

image_quality: int = 95

DEFAULT_MODEL = "lama"
DEFAULT_DEVICE = "cuda"

class Model:
    def __init__(self, **kwargs):
        self.model_manager = None

    def load(self):
        # Load model here and assign to self._model.
        self.model_manager = ModelManager(DEFAULT_MODEL, DEFAULT_DEVICE)
        print("Model loaded")

    def predict(self, model_input):
        form = defaultdict(lambda: None, model_input.get("config", {}))
        images = model_input.get("images", {})
        model_name = model_input.get("model_name", DEFAULT_MODEL)

        self.model_manager.switch(model_name)

        # RGB
        origin_image_bytes = base64.b64decode(images["image"])
        image, alpha_channel, exif_infos = load_img(origin_image_bytes, return_exif=True)

        mask, _ = load_img(base64.b64decode(images["mask"]), gray=True)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

        if image.shape[:2] != mask.shape[:2]:
            return (
                f"Mask shape{mask.shape[:2]} not queal to Image shape{image.shape[:2]}",
                400,
            )

        original_shape = image.shape
        interpolation = cv2.INTER_CUBIC

        size_limit = max(image.shape)

        if "paintByExampleImage" in images:
            paint_by_example_example_image, _ = load_img(
                base64.b64decode(images["paintByExampleImage"])
            )
            paint_by_example_example_image = Image.fromarray(paint_by_example_example_image)
        else:
            paint_by_example_example_image = None

        config = Config(
            ldm_steps=form["ldmSteps"],
            ldm_sampler=form["ldmSampler"],
            hd_strategy=form["hdStrategy"],
            zits_wireframe=form["zitsWireframe"],
            hd_strategy_crop_margin=form["hdStrategyCropMargin"],
            hd_strategy_crop_trigger_size=form["hdStrategyCropTrigerSize"],
            hd_strategy_resize_limit=form["hdStrategyResizeLimit"],
            prompt=form["prompt"],
            negative_prompt=form["negativePrompt"],
            use_croper=form["useCroper"],
            croper_x=form["croperX"],
            croper_y=form["croperY"],
            croper_height=form["croperHeight"],
            croper_width=form["croperWidth"],
            sd_scale=form["sdScale"],
            sd_mask_blur=form["sdMaskBlur"],
            sd_strength=form["sdStrength"],
            sd_steps=form["sdSteps"],
            sd_guidance_scale=form["sdGuidanceScale"],
            sd_sampler=form["sdSampler"],
            sd_seed=form["sdSeed"],
            sd_match_histograms=form["sdMatchHistograms"],
            cv2_flag=form["cv2Flag"],
            cv2_radius=form["cv2Radius"],
            paint_by_example_steps=form["paintByExampleSteps"],
            paint_by_example_guidance_scale=form["paintByExampleGuidanceScale"],
            paint_by_example_mask_blur=form["paintByExampleMaskBlur"],
            paint_by_example_seed=form["paintByExampleSeed"],
            paint_by_example_match_histograms=form["paintByExampleMatchHistograms"],
            paint_by_example_example_image=paint_by_example_example_image,
            p2p_steps=form["p2pSteps"],
            p2p_image_guidance_scale=form["p2pImageGuidanceScale"],
            p2p_guidance_scale=form["p2pGuidanceScale"],
            controlnet_conditioning_scale=form["controlnet_conditioning_scale"],
            controlnet_method=form["controlnet_method"],
        )

        if config.sd_seed == -1:
            config.sd_seed = random.randint(1, 999999999)
        if config.paint_by_example_seed == -1:
            config.paint_by_example_seed = random.randint(1, 999999999)

        logger.info(f"Origin image shape: {original_shape}")
        image = resize_max_size(image, size_limit=size_limit, interpolation=interpolation)

        mask = resize_max_size(mask, size_limit=size_limit, interpolation=interpolation)

        start = time.time()
        try:
            res_np_img = self.model_manager(image, mask, config)
        except RuntimeError as e:
            if "CUDA out of memory. " in str(e):
                # NOTE: the string may change?
                return "CUDA out of memory", 500
            else:
                logger.exception(e)
                return f"{str(e)}", 500
        finally:
            logger.info(f"process time: {(time.time() - start) * 1000}ms")
            torch_gc()

        res_np_img = cv2.cvtColor(res_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        if alpha_channel is not None:
            if alpha_channel.shape[:2] != res_np_img.shape[:2]:
                alpha_channel = cv2.resize(
                    alpha_channel, dsize=(res_np_img.shape[1], res_np_img.shape[0])
                )
            res_np_img = np.concatenate(
                (res_np_img, alpha_channel[:, :, np.newaxis]), axis=-1
            )

        ext = get_image_ext(origin_image_bytes)

        bytes_io = io.BytesIO(
            pil_to_bytes(
                Image.fromarray(res_np_img),
                ext,
                quality=image_quality,
                exif_infos=exif_infos,
            )
        )

        base64_data = base64.b64encode(bytes_io.getvalue()).decode()

        return {"result": base64_data}
