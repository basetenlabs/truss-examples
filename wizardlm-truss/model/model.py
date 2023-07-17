from typing import Any

import sys
import os
import torch
import transformers
import json

from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

class Model:
    def __init__(self, **kwargs) -> None:
        self.model = None
        self.tokenizer = None

    def load(self):
        # Load model here and assign to self._model.
        base_model = "TheBloke/wizardLM-7B-HF"
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.half()
        model.eval()
        
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, request) -> Any:
        prompt = request.pop("prompt")
        _output = evaluate(self.model, self.tokenizer, prompt, **request)
        final_output = _output[0].split("### Response:")[1].strip()
        return final_output

    
def evaluate(
        model,
        tokenizer,
        model_input,
        input=None,
        temperature=1,
        top_p=0.9,
        top_k=40,
        num_beams=1,
        max_new_tokens=2048,
        **kwargs,
):
    prompts = generate_prompt(model_input, input)
    inputs = tokenizer(prompts, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    input_ids = inputs["input_ids"].to("cuda")
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    return output


def generate_prompt(instruction, input=None):
    return f"""{instruction}

### Response:
"""
