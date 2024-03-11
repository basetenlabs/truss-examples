import os
from itertools import count
from pathlib import Path
from threading import Thread
import time
import numpy as np
from client import TritonClient
from transformers import AutoTokenizer
from utils import download_engine, server_loaded

TRITON_MODEL_REPOSITORY_PATH = Path("/packages/inflight_batcher_llm/")


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._request_id_counter = count(start=1)
        self.triton_client = None
        self.tokenizer = None
        self.uses_openai_api = (
            "openai-compatible" in self._config["model_metadata"]["tags"]
        )

    def load(self):
        tensor_parallel_count = self._config["model_metadata"].get(
            "tensor_parallelism", 1
        )
        pipeline_parallel_count = self._config["model_metadata"].get(
            "pipeline_parallelism", 1
        )
        if "hf_access_token" in self._secrets._base_secrets.keys():
            hf_access_token = self._secrets["hf_access_token"]
        else:
            hf_access_token = None
        is_external_engine_repo = "engine_repository" in self._config["model_metadata"]

        # Instantiate TritonClient
        self.triton_client = TritonClient(
            data_dir=self._data_dir,
            model_repository_dir=TRITON_MODEL_REPOSITORY_PATH,
            parallel_count=tensor_parallel_count * pipeline_parallel_count,
        )

        # Download model from Hugging Face Hub if specified
        if is_external_engine_repo:
            if not server_loaded():
                download_engine(
                    engine_repository=self._config["model_metadata"][
                        "engine_repository"
                    ],
                    fp=self._data_dir,
                    auth_token=hf_access_token,
                )

        # Load Triton Server and model
        tokenizer_repository = self._config["model_metadata"]["tokenizer_repository"]
        env = {"triton_tokenizer_repository": tokenizer_repository}
        if hf_access_token is not None:
            env["HUGGING_FACE_HUB_TOKEN"] = hf_access_token

        self.triton_client.load_server_and_model(env=env)

        # setup eos token
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_repository, token=hf_access_token
        )
        self.eos_token_id = self.tokenizer.eos_token_id

        # Warm-up logic
        warmup_prompts = ["Hello, world!", "How are you?", "Tell me a joke."]
        for prompt in warmup_prompts:
            thread = Thread(target=self._warmup, args=(prompt,))
            thread.start()
            thread.join()

    def _warmup(self, prompt):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        model_input = {
            "prompt": prompt,
            "max_tokens": 100
        }
        result = loop.run_until_complete(self.predict(model_input))
        loop.close()

    async def predict(self, model_input):
        stream_uuid = str(os.getpid()) + str(next(self._request_id_counter))
        messages = model_input.get("messages", [])
        prompt = model_input.get("prompt", None)
        if self.uses_openai_api and not prompt:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

        max_tokens = model_input.get("max_tokens", 50) or 50
        beam_width = model_input.get("beam_width", 1) or 1
        bad_words_list = model_input.get("bad_words_list", [""]) or [""]
        stop_words_list = model_input.get("stop_words_list", [""]) or [""] 
        repetition_penalty = model_input.get("repetition_penalty", 1.0) or 1.0
        ignore_eos = model_input.get("ignore_eos", False) or False
        stream = model_input.get("stream", True) or True
        start_time = time.time()

        async def generate():
            first_token_sent = False
            result_iterator = self.triton_client.infer(
                request_id=stream_uuid,
                prompt=prompt,
                max_tokens=max_tokens,
                beam_width=beam_width,
                bad_words=bad_words_list,
                stop_words=stop_words_list,
                stream=stream,
                repetition_penalty=repetition_penalty,
                ignore_eos=ignore_eos,
                eos_token_id=self.eos_token_id,
            )
            async for i in result_iterator:
                if not first_token_sent:
                    first_token_sent = True
                    print(f"First token sent in {time.time() - start_time} seconds.")
                yield i

        return generate()