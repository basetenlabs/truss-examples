from typing import Dict, List

import torch
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None
        self._tokenizer = None

    def load(self):
        self._tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        self._model = LlamaForCausalLM.from_pretrained(
            str(self._data_dir),
            torch_dtype=torch.float16,
            device_map="auto",
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

    def forward(self, prompt, temperature=0.1, top_p=0.75, top_k=40, num_beams=2, **kwargs):
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, padding=False, max_length=1056
        )
        input_ids = inputs["input_ids"].to("cuda")
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=1.2,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = self._model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=1152,
                early_stopping=True,
            )

        decoded_output = []
        for beam in generation_output.sequences:
            decoded_output.append(self._tokenizer.decode(beam, skip_special_tokens=True))

        return decoded_output


    def predict(self, request: Dict) -> Dict[str, List]:
        prompt = request.pop("prompt")
        try:
            completions = self.forward(prompt, **request)
        except Exception as exc:
            return {"status": "error", "data": None, "message": str(exc)}

        return {"status": "success", "data": completions, "message": None}
