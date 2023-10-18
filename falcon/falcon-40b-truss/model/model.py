import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import transformers
from typing import Dict, List

CHECKPOINT = "tiiuae/falcon-40b-instruct"
DEFAULT_MAX_LENGTH = 128
DEFAULT_TOP_P = 0.95

class Model:
    def __init__(self, data_dir: str, config: Dict, secrets: Dict, **kwargs) -> None:
        self._data_dir = data_dir
        self._config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.pipeline = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=CHECKPOINT,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )


    def predict(self, request: Dict) -> Dict:
        with torch.no_grad():
            try:
                prompt = request.pop("prompt")
                data = self.pipeline(
                    prompt,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **request
                )[0]
                return {"data": data}

            except Exception as exc:
                return {"status": "error", "data": None, "message": str(exc)}