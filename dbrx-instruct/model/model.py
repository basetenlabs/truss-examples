from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        # Initialize the model and tokenizer paths for DBRX-Instruct
        self.model_path = "databricks/dbrx-instruct"
        self.tokenizer_path = "databricks/dbrx-instruct"
        # Load the tokenizer for DBRX-Instruct
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        # Load the model with trust_remote_code=True to allow custom code execution
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True
        ).to(self.device)

    def preprocess(self, prompt: str) -> Dict:
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def postprocess(self, output: Dict) -> str:
        return self.tokenizer.decode(output["generated_token_ids"][0])

    def moderate(self, text: str) -> str:
        # TODO: Implement content moderation logic here
        return text

    def predict(self, model_input: Dict) -> Dict:
        # Extract the input text from the model_input dictionary
        input_text = model_input["prompt"]
        # Preprocess the input by encoding it with the tokenizer
        encoded_input = self.tokenizer.encode(input_text, return_tensors="pt").to(
            self.device
        )
        # Generate text from the model using the encoded input
        generated_tokens = self.model.generate(
            encoded_input, max_length=512, num_return_sequences=1
        )
        # Decode the generated tokens into text
        generated_text = self.tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True
        )
        # Return the generated text in a dictionary
        return {"generated_text": generated_text}
