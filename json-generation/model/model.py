import os
import time
from threading import Thread

import torch
from transformers import GenerationConfig, TextIteratorStreamer, pipeline
import jsonformer

class Model:
    def __init__(self, **kwargs):
        self._repo_id = "NousResearch/Hermes-2-Pro-Mistral-7B"
        self._hf_access_token = kwargs["secrets"]["hf_access_token"]
        self._latency_metrics = dict()
        self._model = None
        self._jsonformer = None

    def get_latency_metrics(self):
        return self._latency_metrics

    def load(self):
        self._model = pipeline(
            "text-generation",
            model=self._repo_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=self._hf_access_token,
        )

        self._jsonformer = jsonformer.Jsonformer(model=self._model, tokenizer=self._model.tokenizer, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


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
            k: request.get(k, generate_args[k])
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

    def predict(self, schema: str, request: dict):
        start_time = time.time()
        prefill_start = time.time()
        model_inputs = self._model.tokenizer.apply_chat_template(messages, ...)
        prefill_end = time.time()
        prefill_time = prefill_end - prefill_start

        stream = request.pop("stream", False)
        messages = request.pop("messages")

        # Create template for JSON generation
        system_prompt = f"""<|im_start|>system
You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{schema}\n</schema><|im_end|>"""
        
        chat_template = system_prompt + "\n"
        chat_template += "{% for message in messages %}"
        chat_template += "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        chat_template += "{% endfor %}"
        chat_template += "{% if add_generation_prompt is not defined %}{% set add_generation_prompt = false %}{% endif %}"
        chat_template += "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        
        model_inputs = self._model.tokenizer.apply_chat_template(
            messages, chat_template=chat_template, tokenize=False, add_generation_prompt=True
        )
        generation_args = request.pop("generate_args")

        generation_start = time.time() 
        
        if stream:
            return self.stream(model_inputs, generation_args)

        with torch.no_grad():
            results = self._jsonformer(text_inputs=model_inputs, **generation_args)
            
        first_token_time = time.time() - generation_start
        total_tokens = len(results.split())
        total_time = time.time() - start_time
        tpot = (total_time - first_token_time) / total_tokens if total_tokens > 0 else 0
        
        self._latency_metrics = {
            "prefill_time": prefill_time,
            "time_to_first_token": first_token_time,
            "time_per_output_token": tpot,
            "total_generation_time": total_time,
        }   


        if len(results) > 0:
            return results[0].get("generated_text")

        raise Exception("No results returned from model")