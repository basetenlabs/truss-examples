import base64
import os
import tempfile
from io import BytesIO

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE64_PREAMBLE = "data:image/png;base64,"


class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.tokenizer = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-VL", trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True
        ).eval()

    def b64_to_pil(self, b64_str):
        return Image.open(
            BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, "")))
        )

    def predict(self, request: dict):
        image = request.pop("image")
        prompt = request.pop("prompt")

        created_temp_file = False
        if not image.startswith("http") or not image.startswith("https"):
            image = self.b64_to_pil(image)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".png")
            image.save(temp_file.name)
            temp_file.close()
            image = temp_file.name
            created_temp_file = True

        query = self.tokenizer.from_list_format(
            [
                {"image": image},
                {"text": prompt},
            ]
        )

        inputs = self.tokenizer(query, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        pred = self.model.generate(**inputs)
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
        if created_temp_file:
            os.remove(image)

        return {"output": response}
