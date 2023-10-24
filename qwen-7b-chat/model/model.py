from threading import Thread
from typing import Dict

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer)
from transformers.generation import GenerationConfig


class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.tokenizer = None

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-7B-Chat", trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True
        ).eval()

    def preprocess(self, request: dict):
        generate_args = {
            "max_new_tokens": request.get("max_new_tokens", 512),
            "temperature": request.get("temperature", 0.5),
            "top_p": request.get("top_p", 0.95),
            "top_k": request.get("top_k", 40),
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "use_cache": True,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        request["generate_args"] = generate_args
        return request

    def stream(self, input_ids: list, generation_args: dict):
        streamer = TextIteratorStreamer(self.tokenizer)
        generation_config = GenerationConfig(**generation_args)
        generation_kwargs = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": generation_args["max_new_tokens"],
            "streamer": streamer,
        }

        with torch.no_grad():
            # Begin generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Yield generated text as it becomes available
            def inner():
                for text in streamer:
                    yield text
                thread.join()

        return inner()

    def predict(self, request: Dict):
        stream = request.pop("stream", False)
        prompt = request.pop("prompt")
        generation_args = request.pop("generate_args")
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        if stream:
            return self.stream(input_ids, generation_args)

        with torch.no_grad():
            output = self.model.generate(inputs=input_ids, **generation_args)
            return self.tokenizer.decode(output[0])
