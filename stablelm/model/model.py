from typing import Dict, List
from peft import PeftModel
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None
        self._tokenizer = None

    def load(self):
        model_name = "stabilityai/stablelm-tuned-alpha-7b"  # @param ["stabilityai/stablelm-base-alpha-7b", "stabilityai/stablelm-tuned-alpha-7b", "stabilityai/stablelm-base-alpha-3b", "stabilityai/stablelm-tuned-alpha-3b"]
        # Select "big model inference" parameters
        torch_dtype = "float16"  # @param ["float16", "bfloat16", "float"]
        load_in_8bit = False  # @param {type:"boolean"}
        device_map = "auto"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, torch_dtype),
            load_in_8bit=load_in_8bit,
            device_map=device_map,
            offload_folder="./offload",
        )
        self._model.eval()

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

    def forward(
        self,
        instruction,
        max_new_tokens=64,
        temperature=0.5,
        top_p=0.9,
        top_k=0,
        num_beams=4,
        do_sample=True,
        **kwargs
    ):
        stop = StopOnTokens()
        inputs = self._tokenizer(instruction, return_tensors="pt")
        inputs.to(self._model.device)
        tokens = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self._tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([stop])
        )

        completion_tokens = tokens[0][inputs["input_ids"].size(1) :]
        completion = self._tokenizer.decode(completion_tokens, skip_special_tokens=True)
        return completion

    def predict(self, request: Dict) -> Dict[str, List]:
        prompt = request.pop("prompt")
        completion = self.forward(prompt, **request)
        return {"completion": completion}
