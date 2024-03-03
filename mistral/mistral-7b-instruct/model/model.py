from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)


class Model:
    def __init__(self, **kwargs):
        self.tokenizer = None
        self.model = None

    def load(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
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
        if "max_tokens" in request.keys():
            generate_args["max_new_tokens"] = request["max_tokens"]
        if "temperature" in request.keys():
            generate_args["temperature"] = request["temperature"]
        if "top_p" in request.keys():
            generate_args["top_p"] = request["top_p"]
        if "top_k" in request.keys():
            generate_args["top_k"] = request["top_k"]
        if "repetition_penalty" in request.keys():
            generate_args["repetition_penalty"] = request["repetition_penalty"]
        if "no_repeat_ngram_size" in request.keys():
            generate_args["no_repeat_ngram_size"] = request["no_repeat_ngram_size"]
        if "use_cache" in request.keys():
            generate_args["use_cache"] = request["use_cache"]
        if "do_sample" in request.keys():
            generate_args["do_sample"] = request["do_sample"]
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

    def predict(self, request: dict):
        stream = request.pop("stream", False)
        prompt = request.pop("prompt")
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        generation_args = request.pop("generate_args")
        input_ids = self.tokenizer(
            formatted_prompt, return_tensors="pt"
        ).input_ids.cuda()

        if stream:
            return self.stream(input_ids, generation_args)

        with torch.no_grad():
            try:
                output = self.model.generate(inputs=input_ids, **generation_args)
                return self.tokenizer.decode(output[0])
            except Exception as exc:
                return {"status": "error", "data": None, "message": str(exc)}
