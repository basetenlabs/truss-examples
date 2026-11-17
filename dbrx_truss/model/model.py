from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    def __init__(self, data_dir: str, config: Dict, **kwargs):
        self.data_dir = data_dir
        self.config = config
        self.cuda_available = torch.cuda.is_available()

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "databricks/dbrx-instruct", trust_remote_code=True, token=True
        )

        if self.cuda_available:
            self.model = AutoModelForCausalLM.from_pretrained(
                "databricks/dbrx-instruct",
                trust_remote_code=True,
                token=True,
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                ),
                device_map="auto",
                attn_implementation=(
                    "flash_attention_2" if "flash_attn" in locals() else "eager"
                ),
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                "databricks/dbrx-instruct", trust_remote_code=True, token=True
            )

    def predict(self, request: Dict) -> Dict:
        self.load()  # Reload model for each request

        prompt = request["prompt"]
        messages = [{"role": "user", "content": prompt}]

        tokenized_input = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        tokenized_input = tokenized_input.to(self.model.device)

        generated = self.model.generate(
            input_ids=tokenized_input,
            max_new_tokens=self.config.get("max_new_tokens", 100),
            temperature=self.config.get("temperature", 0.7),
            top_p=self.config.get("top_p", 0.95),
            top_k=self.config.get("top_k", 50),
            repetition_penalty=self.config.get("repetition_penalty", 1.01),
            pad_token_id=self.tokenizer.pad_token_id,
        )

        decoded_output = self.tokenizer.batch_decode(generated)[0]
        response_text = decoded_output.split("<|im_start|> assistant\n")[-1]

        return {"result": response_text}
