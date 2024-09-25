import base64
import io
import subprocess
from typing import Dict, List

import requests
import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self.model = None
        self._tokenizer = None

    def load(self):
        self.model_id = self._config["model_metadata"]["repo_id"]
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=self._secrets["hf_access_token"],
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, token=self._secrets["hf_access_token"]
        )

        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with code {e.returncode}: {e.stderr}")
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": "Can you please describe this image in just one sentence?",
                    },
                ],
            }
        ]

        input_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = self.processor(image, input_text, return_tensors="pt").to(
            self.model.device
        )

        output = self.model.generate(**inputs, max_new_tokens=70)

        print(self.processor.decode(output[0][inputs["input_ids"].shape[-1] :]))

    def _process_image(self, image_input):
        if isinstance(image_input, str):
            if image_input.startswith("http://") or image_input.startswith("https://"):
                # It's a URL
                response = requests.get(image_input)
                image = Image.open(io.BytesIO(response.content))
            else:
                # Assume it's base64
                try:
                    image_data = base64.b64decode(image_input)
                    image = Image.open(io.BytesIO(image_data))
                except:
                    raise ValueError(
                        "Invalid image input. Must be a valid URL or base64 encoded image."
                    )
        elif isinstance(image_input, Image.Image):
            # It's already a PIL Image
            image = image_input
        else:
            raise ValueError(
                "Unsupported image input type. Must be a URL, base64 string, or PIL Image."
            )

        return image

    def predict(self, request: Dict) -> Dict[str, List]:
        messages = request.get("messages", [])
        stream = request.get("stream", False)
        max_new_tokens = request.get("max_new_tokens", 70)
        image_input = request.get("image")
        if image_input:
            image = self._process_image(image_input)
        else:
            image = None
        input_text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = self.processor(image, input_text, return_tensors="pt").to(
            self.model.device
        )

        if stream:

            def generate_stream():
                for token in self.model.generate(
                    **inputs,
                    max_new_tokens=70,
                    num_beams=1,
                    do_sample=True,
                    temperature=0.7,
                    streaming=True,
                ):
                    yield self.processor.decode(token, skip_special_tokens=True)

            return {"generated_text": generate_stream()}
        else:
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_text = self.processor.decode(
                output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
            )
            return {"generated_text": generated_text}
