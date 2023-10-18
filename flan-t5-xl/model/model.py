import random
from typing import Dict, List

from transformers import T5ForConditionalGeneration, T5Tokenizer, set_seed


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
        self._model = T5ForConditionalGeneration.from_pretrained(
            str(self._data_dir), device_map="auto"
        )

    def preprocess(self, request: dict):
        if "bad_words" in request:
            bad_words = request.pop("bad_words")
            bad_word_ids = self._tokenizer(
                bad_words, add_prefix_space=True, add_special_tokens=False
            ).input_ids

            request["bad_words_ids"] = bad_word_ids
        if "seed" in request:
            set_seed(request.pop("seed"))
        else:
            set_seed(random.randint(0, 4294967294))
        return request

    def predict(self, request: Dict) -> Dict[str, List]:
        try:
            decoded_output = []
            prompt = request.pop("prompt")
            input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            outputs = self._model.generate(input_ids, **request)
            for beam in outputs:
                decoded_output.append(
                    self._tokenizer.decode(beam, skip_special_tokens=True)
                )
        except Exception as exc:
            return {"status": "error", "data": None, "message": str(exc)}

        return {"status": "success", "data": decoded_output, "message": None}
