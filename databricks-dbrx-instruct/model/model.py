import logging

from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, model_name="databricks/dbrx-instruct") -> None:
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def preprocess(self, request: dict) -> dict:
        prompt = request.get("prompt", "")
        return {"input_ids": self.tokenizer.encode(prompt, return_tensors="pt")}

    def postprocess(self, output) -> dict:
        return {
            "generated_text": self.tokenizer.decode(output[0], skip_special_tokens=True)
        }

    def predict(self, request: dict) -> dict:
        try:
            processed_input = self.preprocess(request)
            output = self.model.generate(**processed_input)
            return self.postprocess(output)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


# Empty file
