import sys, os
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler

import time
from huggingface_hub import snapshot_download


class Model:
    def __init__(self, **kwargs):
        self.generator = None
        self.tokenizer = None
        self.cache = None

    def load(self):
        # Load model here and assign to self._model.
        model_directory = snapshot_download(
            repo_id="turboderp/Llama2-7B-chat-exl2", revision="4.0bpw"
        )

        config = ExLlamaV2Config()
        config.model_dir = model_directory
        config.prepare()

        model = ExLlamaV2(config)
        print("Loading model: " + model_directory)

        # allocate 18 GB to CUDA:0 and 24 GB to CUDA:1.
        # (Call `model.load()` if using a single GPU.)
        model.load()

        self.tokenizer = ExLlamaV2Tokenizer(config)
        self.cache = ExLlamaV2Cache(model)
        self.generator = ExLlamaV2BaseGenerator(model, self.cache, self.tokenizer)
        self.generator.warmup()

    def predict(self, model_input):
        prompt = model_input["prompt"]
        max_new_tokens = model_input.get("max_new_tokens", 150)
        use_stop_token = model_input.get("use_stop_token", True)
        seed = model_input.get("seed", None)

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

        time_begin = time.time()

        output = self.generator.generate_simple(
            prompt, settings, max_new_tokens, seed=seed
        )

        time_end = time.time()
        time_total = time_end - time_begin

        print(output)
        print()
        print(
            f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total:.2f} tokens/second"
        )

        return {"result": output}
