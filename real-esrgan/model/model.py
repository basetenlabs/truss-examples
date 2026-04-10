import base64
import io
import os
from io import BytesIO
from typing import Dict

import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from PIL import Image
from realesrgan import RealESRGANer

original_working_directory = os.getcwd()


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self.model_checkpoint_path = os.path.join(
            original_working_directory,
            self._data_dir,
            "weights",
            "RealESRGAN_x4plus.pth",
        )
        self.model = None

    def pil_to_b64(self, pil_img):
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def load(self):
        rrdb_net_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4

        self.model = RealESRGANer(
            scale=netscale,
            model_path=self.model_checkpoint_path,
            model=rrdb_net_model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )

    def predict(self, request: Dict) -> Dict:
        image = request.get("image")
        scale = 4

        pil_img = Image.open(io.BytesIO(base64.decodebytes(bytes(image, "utf-8"))))
        pil_image_array = np.asarray(pil_img)

        output, _ = self.model.enhance(pil_image_array, outscale=scale)
        output = Image.fromarray(output)
        output_b64 = self.pil_to_b64(output)
        return {"upscaled_image": output_b64}
