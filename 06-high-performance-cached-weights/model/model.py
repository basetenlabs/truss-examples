# In this example, we go through a Truss that serves an LLM, and _caches_ the weights
# at build time. Loading model weights for any model can often be the most time-consuming
# part of starting a model. Caching the weights at build time means that the weights
# will be baked into the Truss image, and will be available _immediately_ when your model
# replica starts. This means that **cold starts** will be _significantly faster_ with this approach.
#
# # Implementing the `Model` class
#
# With weight caching, you don't have to change anything about how the `Model` class
# is implemented to take advantage of the weight caching.
from typing import Dict, List

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
CHECKPOINT = "/app/model_cache/llama-2-7b-chat-hf"


def format_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"{B_INST} {B_SYS} {system_prompt} {E_SYS} {prompt} {E_INST}"


class Model:
    def __init__(self, **kwargs) -> None:
        self.model = None
        self.tokenizer = None

    def load(self):
        self.model = LlamaForCausalLM.from_pretrained(
            CHECKPOINT, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(CHECKPOINT)

    def predict(self, request: Dict) -> Dict[str, List]:
        prompt = request.pop("prompt")
        input_ids = self.tokenizer(
            format_prompt(prompt), return_tensors="pt"
        ).input_ids.cuda()

        outputs = self.model.generate(
            inputs=input_ids, do_sample=True, num_beams=1, max_new_tokens=100
        )
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        return {"response": response}
