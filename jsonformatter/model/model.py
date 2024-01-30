from transformers import AutoModelForCausalLM, AutoTokenizer
from jsonformer.main import Jsonformer
import json
from pathlib import Path

class Model:
    def __init__(self, **kwargs):

        self._data_dir: Path = kwargs["data_dir"]
        self._config = kwargs["config"]
        # self._secrets = kwargs["secrets"]
        self._model_name = self._config["model_metadata"]["llm_model"]
        self._model = None
        self._tokenizer = None
        self._default_schema = None

    def load(self):
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name, use_cache=True, device_map="auto")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, use_fast=True, use_cache=True)
        self._default_schema = json.loads((self._data_dir / "schema.json").read_text())
        

    def predict(self, model_input: dict):
        schema = self._default_schema
        if "schema" in model_input:
            schema = json.loads(model_input["schema"])
        prompt = model_input["prompt"]
        builder = Jsonformer(
            model=self._model,
            tokenizer=self._tokenizer,
            json_schema=schema,
            prompt=prompt,
        )

        output = builder()

        return output
