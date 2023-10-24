import urllib
from typing import Any

import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

BASE64_PREAMBLE = "data:image/png;base64,"
import logging

class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        sam = sam_model_registry["vit_h"](
            checkpoint=str(self._data_dir / "sam_vit_h_4b8939.pth")
        )
        sam.to("cuda")
        
        self._model = SamAutomaticMaskGenerator(sam, output_mode="coco_rle")

    def predict(self, model_input: Any) -> Any:
        input_image_url = model_input["image_url"]
        
        req = urllib.request.urlopen(input_image_url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        input_image_cv2 = cv2.imdecode(arr, -1)
        
        masks = self._model.generate(input_image_cv2)
        result = {"masks": masks}

        return result
