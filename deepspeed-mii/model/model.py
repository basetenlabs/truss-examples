import asyncio
from typing import Dict, Any

import queue
import mii
import threading
from huggingface_hub import login
import logging

DEFAULT_RESPONSE_MAX_LENGTH = 512


class Model:
    def __init__(self, **kwargs) -> None:
        self.hf_access_token = kwargs["secrets"]["hf_access_token"]
        model_metadata = kwargs["config"]["model_metadata"]
        self.repo_id = model_metadata["repo_id"]
        self.max_length = int(model_metadata["max_length"])
        self.tensor_parallel = int(model_metadata["tensor_parallel"])
        self.is_live_reload = kwargs["config"]["live_reload"]

    def load(self):
        login(token=self.hf_access_token)
        # need to create a new loop because `mii.serve` creates async client at the end,
        # and `load` function being called from new thread
        asyncio.set_event_loop(asyncio.new_event_loop())
        try:
            mii.serve(
                self.repo_id,
                tensor_parallel=self.tensor_parallel,
                max_length=self.max_length,
            )
        except Exception as e:
            if self.is_live_reload:
                # in live reload `mii.serve` fails after reload since server is running already
                logging.info(
                    "An exception occurred while starting mii server: %s, ignoring the exception due to live reload enabled",
                    e,
                )
            else:
                raise e

    def predict(self, request: Dict):
        prompt = request.pop("prompt")
        generate_args = {
            "max_new_tokens": request.pop("max_length", DEFAULT_RESPONSE_MAX_LENGTH)
        }

        if request.pop("stream", False):
            return self.stream(prompt, generate_args)
        else:
            # we need to create new asyncio loop because each request is being server from new thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            client = mii.client(self.repo_id)
            response = client.generate(prompt, **generate_args)
            new_loop.close()

            return {"result": "\n".join(response.response)}

    def stream(self, prompt: str, generate_args: Dict[str, Any]):
        q = queue.Queue()

        def generate():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            client = mii.client(self.repo_id)
            client.generate(prompt, streaming_fn=q.put, **generate_args)
            q.put(None)

        threading.Thread(target=generate).start()

        while True:
            item = q.get()  # This will block until an item is available
            if item is not None:
                yield "".join(item.response)
            else:
                break
