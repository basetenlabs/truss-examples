import torch

from typing import Any
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._model = None
        self._tokenizer = None

    def load(self):
        self._tokenizer = AutoTokenizer.from_pretrained("Aeala/VicUnlocked-alpaca-30b")
        self._model = AutoModelForCausalLM.from_pretrained(
            "Aeala/VicUnlocked-alpaca-30b",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.eval()

    def generate_prompt(self, instruction):
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:
    """

    def forward(self, instruction, temperature=0.1, top_p=0.75, top_k=40, num_beams=2, **kwargs):
        prompt = self.generate_prompt(instruction)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = self._model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=1024,
            )
        
        s = generation_output.sequences[0]
        output = self._tokenizer.decode(s)
        return output.split("### Response:")[1].strip()


    def predict(self, model_input: Any) -> Any:
        prompt = model_input.pop("prompt")
        return self.forward(prompt)
