import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Dict, List
from .patch import patch

CHECKPOINT = "bigcode/starcoder"
DEFAULT_MAX_LENGTH = 128
DEFAULT_TOP_P = 0.95

class Model:
    def __init__(self, data_dir: str, config: Dict, secrets: Dict, **kwargs) -> None:
        self._data_dir = data_dir
        self._config = config
        self._secrets = secrets
        self._hf_api_key = secrets["hf_api_key"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = None
        self._model = None

    def load(self):
        patch()
        self._tokenizer = AutoTokenizer.from_pretrained(
            CHECKPOINT,
            use_auth_token=self._hf_api_key
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT,
            use_auth_token=self._hf_api_key,
            device_map="auto",
            offload_folder="offload"
        )


    def predict(self, request: Dict) -> Dict:
        with torch.no_grad():
            try:
                prompt = request.pop("prompt")
                max_length = request.pop("max_length", DEFAULT_MAX_LENGTH)
                top_p = request.pop("top_p", DEFAULT_TOP_P)
                encoded_prompt = self._tokenizer(prompt, return_tensors="pt").input_ids

                encoded_output = self._model.generate(
                    encoded_prompt,
                    max_length=max_length,
                    top_p=top_p,
                    **request
                )[0]
                decoded_output = self._tokenizer.decode(
                    encoded_output, skip_special_tokens=True
                )
                instance_response = {
                    "completion": decoded_output,
                    "prompt": prompt,
                }

                return {"status": "success", "data": instance_response, "message": None}
            except Exception as exc:
                return {"status": "error", "data": None, "message": str(exc)}