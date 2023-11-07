import asyncio
from typing import Dict

import mii
from huggingface_hub import login

DEFAULT_RESPONSE_MAX_LENGTH = 512
MAX_LENGTH = 4096


class Model:
    def __init__(self, **kwargs) -> None:
        self.hf_access_token = kwargs["secrets"]["hf_access_token"]
        self.repo = kwargs["config"]["model_metadata"]["repo_id"]

    def load(self):
        login(token=self.hf_access_token)
        # need to create a new loop because `mii.serve` creates async client at the end,
        # and `load` function being called from new thread
        asyncio.set_event_loop(asyncio.new_event_loop())
        #
        mii.serve(
            # huggingface repo or folder with model files
            self.repo,
            # increase `tensor_parallel` to use more than one GPU
            tensor_parallel=1,
            # increase `max_length` if you need to support larger inputs/outputs
            max_length=MAX_LENGTH,
        )

    def predict(self, request: Dict):
        # we need to create new asyncio loop because each request is being server from new thread
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        client = mii.client(self.repo)
        response = client.generate(
            request.pop("prompt"),
            max_new_tokens=request.pop("max_length", DEFAULT_RESPONSE_MAX_LENGTH),
        )
        new_loop.close()

        return "\n".join(response.response)
