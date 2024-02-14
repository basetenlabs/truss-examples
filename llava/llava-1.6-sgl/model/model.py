import base64
import json
import tempfile
from io import BytesIO

import aiohttp
import sglang as sgl
from PIL import Image

BASE64_PREAMBLE = "data:image/png;base64,"


class Model:
    def __init__(self, **kwargs):
        self.tokenizer = None

    def b64_to_pil(self, b64_str):
        return Image.open(
            BytesIO(base64.b64decode(b64_str.replace(BASE64_PREAMBLE, "")))
        )

    def load(self):
        self.runtime = sgl.Runtime(
            model_path="dillonlaird/hf-llava-v1.6-34b",
            tokenizer_path="dillonlaird/hf-llava-v1.6-34b",
        )
        sgl.set_default_backend(self.runtime)

    async def add_request(
        self,
        prompt: str,
        image_data: str,
        sampling_params,
    ) -> None:
        json_data = {
            "text": prompt,
            "image_data": image_data,
            "sampling_params": sampling_params,
            "stream": True,
        }

        pos = 0

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(
                "http://127.0.0.1:30000/generate", json=json_data
            ) as response:
                async for chunk, _ in response.content.iter_chunks():
                    chunk = chunk.decode("utf-8")
                    if chunk and chunk.startswith("data:"):
                        if chunk == "data: [DONE]\n\n":
                            break
                        data = json.loads(chunk[5:].strip("\n"))
                        cur = data["text"][pos:]
                        if cur:
                            yield cur
                        pos += len(cur)

    async def generate(
        self,
        engine,
        prompt,
        image_path,
        sampling_params,
    ):
        tokenizer = engine.get_tokenizer()

        messages = [
            {
                "role": "system",
                "content": "Answer the question.",
            },
            {"role": "user", "content": "<image>\n" + prompt},
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt += "<|img_start|>assistant\n"

        stream = self.add_request(prompt, image_path, sampling_params)

        async for output in stream:
            yield output

    def predict(self, request):
        image = request.pop("image")
        prompt = request.pop("prompt")
        max_new_tokens = request.pop("max_new_tokens", 256)
        temperature = request.pop("temperature", 0.2)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".png")
        image = self.b64_to_pil(image)
        image.save(temp_file.name)

        return self.generate(
            self.runtime,
            prompt,
            temp_file.name,
            {"temperature": temperature, "max_new_tokens": max_new_tokens},
        )
