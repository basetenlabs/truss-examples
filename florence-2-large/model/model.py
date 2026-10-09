from typing import Dict, List

import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.processor = None
        self.model_name = "microsoft/Florence-2-large"

    def load(self):
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )

    def preprocess(self, prompt: str, image_url: str) -> Dict:
        image = Image.open(requests.get(image_url, stream=True).raw)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        return inputs

    def postprocess(
        self, generated_ids: List[int], original_image: Image.Image
    ) -> Dict:
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=self.task,
            image_size=(original_image.width, original_image.height),
        )
        return parsed_answer

    def predict(self, model_input: Dict) -> Dict[str, List]:
        self.load()
        prompt = model_input["prompt"]
        image_url = model_input["image_url"]
        self.task = model_input.get("task", "<OD>")

        inputs = self.preprocess(prompt, image_url)
        original_image = inputs.pop("images")

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )

        parsed_answer = self.postprocess(generated_ids, original_image)

        return {"result": parsed_answer}
