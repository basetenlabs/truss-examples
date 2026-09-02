from threading import Thread
from typing import Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 50
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_NO_REPEAT_NGRAM_SIZE = 0
DEFAULT_STREAM = False
MODEL_NAME = "Deci/DeciLM-7B-instruct"


class Model:
    def __init__(self, **kwargs):
        self.tokenizer = None
        self.model = None

    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(self, request: dict):
        generate_args = {
            "max_new_tokens": request.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
            "temperature": request.get("temperature", DEFAULT_TEMPERATURE),
            "top_p": request.get("top_p", DEFAULT_TOP_P),
            "top_k": request.get("top_k", DEFAULT_TOP_K),
            "repetition_penalty": request.get(
                "repetition_penalty", DEFAULT_REPETITION_PENALTY
            ),
            "no_repeat_ngram_size": request.get(
                "no_repeat_ngram_size", DEFAULT_NO_REPEAT_NGRAM_SIZE
            ),
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

    def predict(self, model_input: Dict):
        stream = model_input.pop("stream", DEFAULT_STREAM)
        prompt = model_input.pop("prompt")
        generation_args = model_input.pop("generate_args")

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        if stream:
            return self.stream(input_ids, generation_args)

        with torch.no_grad():
            output = self.model.generate(inputs=input_ids, **generation_args)
            return self.tokenizer.decode(output[0])
