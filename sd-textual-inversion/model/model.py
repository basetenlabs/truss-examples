import base64
from typing import Dict, List
import os
import torch
from io import BytesIO
from diffusers import StableDiffusionPipeline

SD_BASE_MODEL_CHECKPOINT = "CompVis/stable-diffusion-v1-4"
REPO_ID_EMBEDS = "sd-concepts-library/cat-toy"
from PIL import Image

class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None

    def load(self):
        self.pipe = StableDiffusionPipeline.from_pretrained(
            SD_BASE_MODEL_CHECKPOINT,
            torch_dtype=torch.float16
        ).to("cuda")
        
        self.pipe.load_textual_inversion(REPO_ID_EMBEDS)
        self.pipe.load_textual_inversion(str(self._data_dir / "charturnerv2.pt"), token="charturnerv2")
        
    def convert_to_b64(self, image: Image) -> str:
        buffered = BytesIO()
        image[0].save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_b64


    @torch.inference_mode()
    def predict(self, request: Dict) -> Dict[str, List]:
        results = []
        random_seed = int.from_bytes(os.urandom(2), "big")
        generator = torch.Generator("cuda").manual_seed(request.get("seed", random_seed))
        try:
            outputs = self.pipe(
                **request,
                generator=generator,
                return_dict=False,
            )
            # for output in outputs:
            b64_results = self.convert_to_b64(outputs[0])
            results = results + [b64_results]

        except Exception as exc:
            return {"status": "error", "data": None, "message": str(exc)}

        return {"status": "success", "data": results, "message": None}
