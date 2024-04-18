from threading import Thread
from typing import Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)

MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"
MAX_LENGTH = 512
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 40
REPETITION_PENALTY = 1.0
NO_REPEAT_NGRAM_SIZE = 0
DO_SAMPLE = True
DEFAULT_STREAM = True


class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.tokenizer = None
        self._secrets = kwargs["secrets"]
        self.hf_access_token = self._secrets["hf_access_token"]

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, use_auth_token=self.hf_access_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.float16,
            use_auth_token=self.hf_access_token,
        )

    def preprocess(self, request: dict):
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
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
            "eos_token_id": terminators,
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

    def predict(self, request: Dict):
        messages = request.pop("messages")
        stream = request.pop("stream", DEFAULT_STREAM)
        generation_args = request.pop("generate_args")

        model_inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(model_inputs, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")

        if stream:
            return self.stream(input_ids, generation_args)

        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, **generation_args)
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return {"output": output_text}
