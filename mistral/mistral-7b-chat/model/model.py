from threading import Thread

import torch
from transformers import GenerationConfig, TextIteratorStreamer, pipeline

CHECKPOINT = "mistralai/Mistral-7B-Instruct-v0.1"


class Model:
    def __init__(self, **kwargs):
        self._model = None

    def load(self):

        self._model = pipeline(
            "text-generation",
            model=CHECKPOINT,
            torch_dtype=torch.bfloat16,
            device_map="auto",
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
            "eos_token_id": self._model.tokenizer.eos_token_id,
            "pad_token_id": self._model.tokenizer.pad_token_id,
            "return_full_text": False,
        }

        request["generate_args"] = {
            k: request[k] if k in request else generate_args[k]
            for k in generate_args.keys()
        }

        return request

    def stream(self, text_inputs: list, generation_args: dict):
        streamer = TextIteratorStreamer(self._model.tokenizer)
        generation_config = GenerationConfig(**generation_args)
        generation_kwargs = {
            "text_inputs": text_inputs,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": generation_args["max_new_tokens"],
            "streamer": streamer,
        }

        with torch.no_grad():
            # Begin generation in a separate thread
            thread = Thread(target=self._model, kwargs=generation_kwargs)
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

        model_inputs = self._model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        generation_args = request.pop("generate_args")

        if stream:
            return self.stream(model_inputs, generation_args)

        with torch.no_grad():
            try:
                return self._model(text_inputs=model_inputs, **generation_args)
            except Exception as exc:
                return {"status": "error", "data": None, "message": str(exc)}
