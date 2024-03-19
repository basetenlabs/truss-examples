import base64
import os
from io import BytesIO

import tensorrt as trt
from helpers.inpaint_pipeline import InpaintPipeline
from helpers.utilities import TRT_LOGGER, add_arguments, download_image
from huggingface_hub import snapshot_download
from PIL import Image

BASE64_PREAMBLE = "data:image/png;base64,"
max_batch_size = 4


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        # self._secrets = kwargs["secrets"]
        # self.hf_token = self._secrets["hf_access_token"]
        self.model = None
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")

    def b64_to_pil(self, b64_str):
        return Image.open(
            BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, "")))
        )

    def pil_to_b64(self, pil_img):
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def load(self):
        snapshot_download(
            "baseten/sdxl-controlnet-inpaint-trt-A10G",
            local_dir=self._data_dir,
            max_workers=4,
        )

        print("data directory files: ", os.listdir(self._data_dir))

        self.model = InpaintPipeline(
            scheduler="PNDM",
            denoising_steps=50,
            output_dir=".",
            version="1.5",
            hf_token=None,
            verbose=False,
            nvtx_profile=False,
            max_batch_size=max_batch_size,
        )

        # Load TensorRT engines and pytorch modules
        self.model.loadEngines(
            os.path.join(self._data_dir, "engine-1.5"),
            os.path.join(self._data_dir, "onnx-1.5"),
            None,
            opt_batch_size=None,
            opt_image_height=None,
            opt_image_width=None,
        )
        self.model.loadResources(512, 512, 1, None)

    def predict(self, request):
        prompt = request.pop("prompt")
        negative_prompt = request.pop("negative_prompt", "blurry, low quality")
        mask = request.pop("mask")
        image = request.pop("image")

        image = self.b64_to_pil(image)
        mask = self.b64_to_pil(mask)

        image_width, image_height = image.size
        mask_width, mask_height = mask.size

        if mask_height != image_height or mask_width != image_width:
            raise ValueError(
                f"Input image height and width {image_height} and {image_width} are not equal to "
                f"the respective dimensions of the mask image {mask_height} and {mask_width}"
            )

        if image_height % 8 != 0 or image_width % 8 != 0:
            raise ValueError(
                f"Image height and width have to be divisible by 8 but specified as: {image_height} and {image_width}."
            )

        images = self.model.infer(
            prompt,
            negative_prompt,
            image,
            mask,
            image_height,
            image_width,
            seed=None,
            strength=0.75,
        )

        print(os.listdir())

        outputs = []
        for image_path in images:
            outputs.append(self.pil_to_b64(Image.open(image_path)))
        return {"outputs": outputs}
