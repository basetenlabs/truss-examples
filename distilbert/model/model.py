import torch
from transformers import AutoTokenizer, AutoModel


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):
        # Load model here and assign to self._model.
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # the device to load the model onto

        self._tokenizer = AutoTokenizer.from_pretrained(
            "distilbert/distilbert-base-uncased", device=self.device
        )
        self._model = AutoModel.from_pretrained(
            "distilbert/distilbert-base-uncased",
            torch_dtype=torch.float16,
        ).to(self.device)

    def predict(self, model_input):
        # Run model inference here
        
        text = model_input.get("text")

        encoded_input = self._tokenizer(text, return_tensors='pt').to(self.device)
        
        return self._model(**encoded_input).last_hidden_state.tolist()
