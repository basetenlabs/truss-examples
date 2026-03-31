import base64
import os
import tempfile
from io import BytesIO

from model.clone_repo_helper import clone_repo
from PIL import Image

# Need to clone repo before importing the things below
clone_repo()

from model.helpers import (
    create_hyper_parameters,
    download_models,
    load_image,
    predict,
    setup_model,
)

BASE64_PREAMBLE = "data:image/png;base64,"


class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.hypar = None
        self.device = "cuda"
        self._data_dir = kwargs["data_dir"]

    def load(self):
        download_models()
        self.hypar = create_hyper_parameters()
        self.model = setup_model(self.hypar)

    def pil_to_b64(self, pil_img):
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def b64_to_pil(self, b64_str):
        return Image.open(
            BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, "")))
        )

    def predict(self, request: dict) -> dict:
        input_image = request.pop("input_image")
        input_image = self.b64_to_pil(input_image)

        with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as temp_file:
            image_path = temp_file.name
            input_image.save(image_path)

        with open(image_path, "rb") as image_file:
            image_bytes = BytesIO(image_file.read())

        image_bytes.seek(0)

        image_tensor, orig_size = load_image(image_path, self.hypar)
        mask = predict(self.model, image_tensor, orig_size, self.hypar, self.device)

        pil_mask = Image.fromarray(mask).convert("L")
        im_rgb = Image.open(image_bytes).convert("RGB")

        im_rgba = im_rgb.copy()
        im_rgba.putalpha(pil_mask)

        background_less_image = self.pil_to_b64(im_rgba)
        image_mask = self.pil_to_b64(pil_mask)

        os.remove(image_path)

        return {"img_without_bg": background_less_image, "image_mask": image_mask}
