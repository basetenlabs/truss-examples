from typing import Dict

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor
import requests
from PIL import Image


CHECKPOINT = "Salesforce/blip-image-captioning-large"

class Model:
    def __init__(self, data_dir: str, config: Dict, **kwargs) -> None:
        self._data_dir = data_dir
        self._config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = None
        self.model = None

    def load(self):
        self.processor = AutoProcessor.from_pretrained(CHECKPOINT)
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)
    
    def preprocess(self, request: Dict) -> Dict:
        try:
            img_url = request["image_url"]
            raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
            request['raw_image'] = raw_image
            return request
        except Exception as exc:
                return {"status": "error", "data": None, "message": str(exc)}

    def predict(self, request: Dict) -> Dict:
        try:
            raw_image = request["raw_image"]
            with torch.no_grad():
                inputs = None
                if "text" in request:
                    #Conditional generation
                    text = request["text"]
                    inputs = self.processor(raw_image, text, return_tensors="pt").to(self.device)

                else:
                    #Unconditional generation
                    inputs = self.processor(raw_image, return_tensors="pt").to(self.device)

                outputs = self.model.generate(**inputs)

                caption = self.processor.decode(outputs[0], skip_special_tokens=True)

                return {"data" : caption}

        except Exception as exc:
                return {"status": "error", "data": None, "message": str(exc)}