import logging
from pathlib import Path

import base64
from io import BytesIO
import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from PIL import Image
from realesrgan import RealESRGANer


logger = logging.getLogger(__name__)

bg_tile = 400  # Tile size for background sampler, 0 for no tile during testing
upscale = 2  # The final upsampling scale of the image
arch = "clean"  # The GFPGAN architecture. Option: clean | original
channel = 2  # Channel multiplier for large networks of StyleGAN2

aligned = False  # Input are aligned faces
only_center_face = False  # Only restore the center face
paste_back = True  # Paste the restored faces back to images

# The model path
GFPGAN_PATH = "GFPGANv1.3.pth"
# The background upsampler
ESRGAN_PATH = (
    "RealESRGAN_x2plus.pth"
)

RESIZE_DEFAULT_MAX = 1400


class RestorationModel:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs.get("data_dir")
        self._config = kwargs.get("config")

    def load(self):
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        self.bg_upsampler = RealESRGANer(
            scale=2,
            model_path=str(Path(self._data_dir) / ESRGAN_PATH),
            model=self.model,
            tile=bg_tile,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )
        self.restorer = GFPGANer(
            model_path=str(Path(self._data_dir) / GFPGAN_PATH),
            upscale=upscale,
            arch=arch,
            channel_multiplier=channel,
            bg_upsampler=self.bg_upsampler,
        )

    def predict(self, request):
        image_b4 = request["image"]
        try:
            image = load_img_from_b64(image_b4)
            (
                input_img,
                cropped_faces,
                restored_faces,
                restored_img,
            ) = self.restore_image(image)
            if restored_img is not None:
                rgb_image = rotate_axis(restored_img)

                img = Image.fromarray(rgb_image)

                b64_image_str = convert_to_b64(img)
                return {"status": "success", "data": b64_image_str, "message": None}
            else:
                return {"status": "error", "data": None, "message": str(exc)}
        except Exception as exc:
            return {"status": "error", "data": None, "message": str(exc)}



    def restore_image(self, input_img):
        cropped_faces, restored_faces, restored_img = self.restorer.enhance(
            input_img,
            has_aligned=aligned,
            only_center_face=only_center_face,
            paste_back=paste_back,
        )
        return input_img, cropped_faces, restored_faces, restored_img


def convert_to_b64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def load_img_from_b64(img_b64: str):
    base64_decoded = base64.b64decode(img_b64)
    pil_image = Image.open(BytesIO(base64_decoded))
    image_content = np.array(pil_image)

    # We need to convert from RGB to BGR because PIL interprets colors differently than OpenCV.
    image = cv2.cvtColor(image_content, cv2.COLOR_RGB2BGR)
    scale = min(RESIZE_DEFAULT_MAX / image.shape[1], RESIZE_DEFAULT_MAX / image.shape[0])
    if scale < 1:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    return image


def rotate_axis(bgr_image):
    """The images from GFPGAN are BGR, need to convert to RGB"""
    return bgr_image[:, :, ::-1]
