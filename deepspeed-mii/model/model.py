import asyncio
from typing import Dict

import mii
from huggingface_hub import login

DEFAULT_RESPONSE_MAX_LENGTH = 512

class Model:
    def __init__(self, **kwargs) -> None:
        self.hf_access_token = kwargs["secrets"]["hf_access_token"]
        self.repo_id = kwargs["config"]["model_metadata"]["repo_id"]
        self.max_length = int(kwargs["config"]["model_metadata"]["max_length"])
        self.tensor_parallel = int(kwargs["config"]["model_metadata"]["tensor_parallel"])

    def load(self):
        login(token=self.hf_access_token)
        # need to create a new loop because `mii.serve` creates async client at the end,
        # and `load` function being called from new thread
        asyncio.set_event_loop(asyncio.new_event_loop())
        mii.serve(self.repo_id, tensor_parallel=self.tensor_parallel, max_length=self.max_length)

    def predict(self, request: Dict):
        # we need to create new asyncio loop because each request is being server from new thread
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        client = mii.client(self.repo_id)
        response = client.generate(
            request.pop("prompt"),
            max_new_tokens=request.pop("max_length", DEFAULT_RESPONSE_MAX_LENGTH),
        )
        new_loop.close()

        return "\n".join(response.response)
