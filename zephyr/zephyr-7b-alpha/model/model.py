from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)

CHECKPOINT = "HuggingFaceH4/zephyr-7b-alpha"


class Model:
    def __init__(self, **kwargs):
        self.tokenizer = None
        self.model = None

    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            CHECKPOINT,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    def preprocess(self, request: dict):
        generate_args = {
            "max_new_tokens": 512,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "use_cache": True,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        request["generate_args"] = {
            request[k] if k in request else generate_args[k]
            for k in generate_args.keys()
        }

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

    def predict(self, request: dict):
        stream = request.pop("stream", False)
        messages = request.pop("messages")

        model_inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        ).cuda()
        generation_args = request.pop("generate_args")

        if stream:
            return self.stream(model_inputs, generation_args)

        with torch.no_grad():
            try:
                output = self.model.generate(inputs=model_inputs, **generation_args)
                return self.tokenizer.decode(output[0])
            except Exception as exc:
                return {"status": "error", "data": None, "message": str(exc)}
