import os
from typing import Dict, Any
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs.get("data_dir")
        self._config = kwargs.get("config")
        self._secrets = kwargs.get("secrets")
        self._model = None
        self._tokenizer = None

        self.model_metadata = self._config.get("model_metadata", {})
        self.model_name = self.model_metadata.get("model_name", "llama")
        self.model_repo = self.model_metadata.get("repo_id", "NousResearch/Meta-Llama-3.1-8B-Instruct") 
        
    def load(self):
        """
        Initialize and load the vLLM model
        """
        if self._secrets and "hf_access_token" in self._secrets:
            os.environ["HF_TOKEN"] = self._secrets["hf_access_token"]

        self._model = LLM(
            model=self.model_repo,
            # TODO: configure based on model metadata
            tensor_parallel_size=1,
            speculative_model="[ngram]",
            num_speculative_tokens=64,
            ngram_prompt_lookup_max=64,
            ngram_prompt_lookup_min=2
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_repo)
        
    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:            
        messages = request.get("messages", [])
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False)

        sampling_params = SamplingParams(
            temperature=request.get("temperature", 0.0),
            max_tokens=request.get("max_tokens", 512),
        )

        outputs = self._model.generate([prompt], sampling_params) 
        return outputs[0].outputs[0].text