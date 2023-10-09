from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Model:
    def __init__(self, **kwargs) -> None:
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self.model = None
        self.tokenizer = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Writer/camel-5b-hf",
            use_auth_token=self._secrets["hf_access_token"],
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "Writer/camel-5b-hf",
            use_auth_token=self._secrets["hf_access_token"],
            torch_dtype=torch.float32,
            device_map="auto",
        )

    def generate_prompt(self, instruction: str, user_input: str = None):
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }

        text = (
            PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
            if not user_input
            else PROMPT_DICT["prompt_input"].format(
                instruction=instruction, user_input=user_input
            )
        )

        return text

    def predict(self, request: Any) -> Any:
        instruction = request.pop("instruction")
        user_input = request.pop("input", None)
        text = self.generate_prompt(instruction=instruction, user_input=user_input)
        model_inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        output_ids = self.model.generate(
            **model_inputs,
            **request,
            max_length=2000,
        )
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ]
        clean_output = output_text.split("### Response:")[1].strip()
        return {"completion": clean_output}
