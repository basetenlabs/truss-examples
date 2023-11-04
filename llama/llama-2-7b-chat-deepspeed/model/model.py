from typing import Dict
import mii
import asyncio
from huggingface_hub import login

class Model:
    def __init__(self, **kwargs) -> None:
        self.hf_access_token = kwargs["secrets"]["hf_access_token"]
        self.repo = "meta-llama/Llama-2-7b-chat-hf"

    def load(self):
        login(token=self.hf_access_token)
        asyncio.set_event_loop(asyncio.new_event_loop())
        mii.serve(self.repo)

    def predict(self, request: Dict):
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        client = mii.client(self.repo)
        response = client.generate(request.pop("prompt"), max_new_tokens=request.pop("max_length", 50))
        new_loop.close()

        return '\n'.join(response.response)
