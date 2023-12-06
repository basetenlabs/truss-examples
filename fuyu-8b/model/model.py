import base64
from io import BytesIO
from typing import Dict

from PIL import Image
from transformers import FuyuForCausalLM, FuyuProcessor


class Model:
    MODEL_ID = "adept/fuyu-8b"

    def __init__(self, **kwargs):
        self.processor = None
        self.model = None

    def b64_to_pil(self, b64_str):
        BASE64_PREAMBLE = "data:image/png;base64,"
        return Image.open(
            BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, "")))
        )

    def load(self):
        self.processor = FuyuProcessor.from_pretrained(self.MODEL_ID)
        self.model = FuyuForCausalLM.from_pretrained(self.MODEL_ID, device_map="cuda:0")

    def predict(self, model_input: Dict) -> Dict:
        prompt = model_input.pop("prompt")
        max_new_tokens = model_input.pop("max_new_tokens")
        image = model_input.pop("image")
        pil_image = self.b64_to_pil(image)

        model_inputs = self.processor(text=prompt, images=[pil_image], device="cuda")
        for k, v in model_inputs.items():
            if isinstance(v, list):
                v = v[0]
            model_inputs[k] = v.to("cuda")

        last_tokens = -1 * max_new_tokens
        generation_output = self.model.generate(
            **model_inputs, max_new_tokens=max_new_tokens
        )
        generation_text = self.processor.batch_decode(
            generation_output[:, last_tokens:], skip_special_tokens=True
        )
        model_output = generation_text[0]
        index = model_output.find("\x04")
        if index != -1:
            model_output = model_output[index:].strip()
        return {"model_output": model_output}
