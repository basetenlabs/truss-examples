from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest coding assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


class Model:
    MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"

    def __init__(self, data_dir: str, config: Dict, **kwargs):
        self._data_dir = data_dir
        self._config = config
        self.model = None
        self.tokenizer = None

    def preprocess(self, request: dict):
        generate_args = {
            "max_new_tokens": 128,
            "temperature": 0.5,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "use_cache": True,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if "max_tokens" in request.keys():
            generate_args["max_new_tokens"] = request["max_tokens"]
        if "temperature" in request.keys():
            generate_args["temperature"] = request["temperature"]
        if "top_p" in request.keys():
            generate_args["top_p"] = request["top_p"]
        if "top_k" in request.keys():
            generate_args["top_k"] = request["top_k"]
        request["generate_args"] = generate_args
        return request

    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )

    def predict(self, request: Dict) -> Dict:
        with torch.no_grad():
            try:
                prompt = request.pop("prompt")
                # Code-llama needs the prompt to be formatted in this manner
                formatted_prompt = f"<s><<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n{prompt}"
                input_ids = self.tokenizer(
                    formatted_prompt, return_tensors='pt').input_ids.cuda()
                output = self.model.generate(
                    inputs=input_ids,
                    **request["generate_args"]
                )

                return self.tokenizer.decode(output[0])
            except Exception as exc:
                return {"status": "error", "data": None, "message": str(exc)}
