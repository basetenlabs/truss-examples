from typing import Dict

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

CHECKPOINT = "meta-llama/Llama-2-13b-hf"
DEFAULT_MAX_LENGTH = 128
DEFAULT_TOP_P = 0.95


class Model:
    def __init__(self, data_dir: str, config: Dict, **kwargs) -> None:
        self._data_dir = data_dir
        self._config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.pipeline = None
        self.hf_access_token = kwargs["secrets"]["hf_access_token"]

    def preprocess(self, request: dict):
        generate_args = {
            "max_new_tokens": 128,
            "temperature": 1.0,
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
        self.model = LlamaForCausalLM.from_pretrained(
            CHECKPOINT, use_auth_token=self.hf_access_token, device_map="auto"
        )

        self.tokenizer = LlamaTokenizer.from_pretrained(
            CHECKPOINT, device_map="auto", use_auth_token=self.hf_access_token
        )

    def predict(self, request: Dict) -> Dict:
        with torch.no_grad():
            try:
                prompt = request.pop("prompt")
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
                output = self.model.generate(
                    inputs=input_ids, **request["generate_args"]
                )

                return self.tokenizer.decode(output[0])
            except Exception as exc:
                return {"status": "error", "data": None, "message": str(exc)}
