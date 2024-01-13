from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
)

MODEL_NAME = "NousResearch/Nous-Capybara-34B"

DEFAULT_STREAM = False


def _format_prompt(messages: list[dict], add_generation_prompt: bool = True) -> str:
    formatted_prompts = []
    for message in messages:
        if message["role"] == "user":
            formatted_prompts.append(f"USER: {message['content']}")
        elif message["role"] == "assistant":
            formatted_prompts.append(f"ASSISTANT: {message['content']}")
        # Note: Capybara doesn't support system messages. See https://huggingface.co/NousResearch/Nous-Capybara-34B/discussions/8
    if add_generation_prompt:
        formatted_prompts.append("ASSISTANT:")

    return "\n".join(formatted_prompts)


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
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 5,
            "use_cache": True,
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        request["generate_args"] = {
            k: request[k]
            if k in request and request[k] is not None
            else generate_args[k]
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
        stream = request.pop("stream", DEFAULT_STREAM)
        messages = request.pop("messages")
        input_ids = self.tokenizer(
            _format_prompt(messages), return_tensors="pt"
        ).input_ids.cuda()

        generation_args = request.pop("generate_args")

        if stream:
            return self.stream(input_ids, generation_args)

        with torch.no_grad():
            outputs = self.model.generate(inputs=input_ids, **generation_args)
            if len(outputs) > 0:
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            raise Exception("No results returned from model")
