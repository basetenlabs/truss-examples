import logging
from pathlib import Path
import traceback
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM
from transformers import CodeGenTokenizer
from transformers import pipeline


DEFAULT_PAD = 50256
DEFAULT_MAX_LENGTH = 128
DEFAULT_TOP_P = 0.95
DEFAULT_TEMPERATURE = 0.2
DEFAULT_BATCH_SIZE = 1


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_id = 0 if self.device == "cuda:0" else -1
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        self._model = pipeline(
            "text-generation",
            tokenizer=CodeGenTokenizer.from_pretrained(Path(self._data_dir) / "tokenizer"),
            model=AutoModelForCausalLM.from_pretrained(Path(self._data_dir) / "saved_model"),
            device_map="auto",
        )

    def preprocess(self, request: Dict) -> Dict:
        """
        Incorporate pre-processing required by the model if desired here.

        These might be feature transformations that are tightly coupled to the model.
        """
        return request

    def postprocess(self, request: Dict) -> Dict:
        """
        Incorporate post-processing required by the model if desired here.
        """
        return request

    def truncate(self, model_output: str, prompt: str):
        if prompt == model_output[: len(prompt)]:
            return model_output[len(prompt) :]
        return model_output

    def predict(self, request: Dict) -> Dict[str, List]:
        with torch.no_grad():
            try:
                prompt = request["prompt"]
                temp = request.get("temperature", DEFAULT_TEMPERATURE)
                max_length = request.get("max_length", DEFAULT_MAX_LENGTH)
                top_p = request.get("top_p", DEFAULT_TOP_P)
                encoded_prompt = self._model.tokenizer(prompt, return_tensors="pt").input_ids
                encoded_output = self._model.model.generate(
                    encoded_prompt,
                    max_length=max_length,
                    top_p=top_p,
                )[0]
                decoded_output = self._model.tokenizer.decode(
                    encoded_output, skip_special_tokens=True
                )
                truncated_output = self.truncate(decoded_output, prompt)
                instance_response = {
                    "completion": decoded_output,
                    "prompt": prompt,
                    "truncation": truncated_output,
                }

                return {"status": "success", "data": instance_response, "message": None}
            except Exception as exc:
                return {"status": "error", "data": None, "message": str(exc)}