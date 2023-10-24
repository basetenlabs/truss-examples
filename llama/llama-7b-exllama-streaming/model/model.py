import sys, os

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

import time
from huggingface_hub import snapshot_download

from threading import Thread, Condition




class Model:
    def __init__(self, **kwargs):
        self.generator = None
        self.tokenizer = None
        self.cache = None

    def load(self):
        # Load model here and assign to self._model.
        # GPTQ, EXL2, and safetensors models with Llama-style architecture (including Mistral) are compatible.
        model_directory =  snapshot_download(repo_id="turboderp/Llama2-7B-chat-exl2", revision="4bpw")

        config = ExLlamaV2Config()
        config.model_dir = model_directory
        config.prepare()

        model = ExLlamaV2(config)
        print("Loading model: " + model_directory)

        # allocate 18 GB to CUDA:0 and 24 GB to CUDA:1.
        # (Call `model.load()` if using a single GPU.)
        model.load()


        self.tokenizer = ExLlamaV2Tokenizer(config)

        # Note: if you want to batch -> https://github.com/turboderp/exllamav2/issues/42
        self.cache = ExLlamaV2Cache(model)

        self.generator = ExLlamaV2StreamingGenerator(model, self.cache, self.tokenizer)
        self.generator.warmup()


    def streamer(self, queue, condition_var):
        generated_tokens = 0
        while True:
            chunk, eos, _ = self.generator.stream()
            generated_tokens += 1
            queue.append(chunk)
            with condition_var:
                condition_var.notify()
            if eos or generated_tokens == self.max_new_tokens: break
        queue.append(None)
        with condition_var:
            condition_var.notify()


    def predict(self, model_input):
        prompt = model_input["prompt"]
        self.max_new_tokens = model_input.get("max_new_tokens", None)
        use_stop_token = model_input.get("use_stop_token", True)
        seed = model_input.get("seed", None)

        input_ids = self.tokenizer.encode(prompt)

        # sampling settings
        temperature = model_input.get("temperature", 0.85)
        top_k = model_input.get("top_k", 50)
        top_p = model_input.get("top_p", 0.8)
        token_repetition_penalty = model_input.get("token_repetition_penalty", 1.15)

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature
        settings.top_k = top_k
        settings.top_p = top_p
        settings.token_repetition_penalty = token_repetition_penalty

        if not use_stop_token:
            settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])

        self.generator.set_stop_conditions([])
        self.generator.begin_stream(input_ids, settings)

        # The variables below manage streaming the output from the model by using a queue
        # and a condition variable. The `streamer` function is run in a separate thread
        # and fills the queue with chunks of text as they are generated. The `predict`
        # function then yields these chunks as they are generated.

        queue = []
        cond = Condition()
        thread = Thread(target=self.streamer, kwargs={"queue": queue, "condition_var": cond})
        thread.start()

        # Wait for the first chunk to be generated.
        def inner():
            while True:
                with cond:
                    while len(queue) == 0:
                        cond.wait()
                    if queue[0] is None:
                        break
                    yield queue.pop(0)
            thread.join()

        return inner()
