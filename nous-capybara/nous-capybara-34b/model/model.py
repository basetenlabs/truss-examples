from threading import Thread
from typing import Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)

MODEL_NAME = "NousResearch/Nous-Capybara-34B"
MAX_LENGTH = 256
DO_SAMPLE = True
REPETITION_PENALTY = 1.3
NO_REPEAT_NGRAM_SIZE = 5
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.8
DEFAULT_STREAM = True


class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.tokenizer = None

    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )

    def preprocess(self, request: dict):
        generate_args = {
            "max_length": request.get("max_tokens", MAX_LENGTH),
            "temperature": request.get("temperature", TEMPERATURE),
            "top_p": request.get("top_p", TOP_P),
            "top_k": request.get("top_k", TOP_K),
            "repetition_penalty": request.get("repetition_penalty", REPETITION_PENALTY),
            "no_repeat_ngram_size": request.get(
                "no_repeat_ngram_size", NO_REPEAT_NGRAM_SIZE
            ),
            "do_sample": request.get("do_sample", DO_SAMPLE),
            "use_cache": True,
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
            "max_new_tokens": generation_args["max_length"],
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
        prompt = model_input.get("prompt")
        stream = model_input.get("stream", DEFAULT_STREAM)
        generation_args = model_input.pop("generate_args")

        formatted_prompt = f"USER: {prompt}\n ASSISTANT:"
        input_ids = self.tokenizer(
            formatted_prompt, return_tensors="pt"
        ).input_ids.cuda()

        if stream:
            return self.stream(input_ids, generation_args)

        with torch.no_grad():
            outputs = self.model.generate(inputs=input_ids, **generation_args)
            model_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"output": model_output}
