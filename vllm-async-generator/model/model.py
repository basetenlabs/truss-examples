
from typing import Any
import os
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import uuid

class Model:
    def __init__(self, **kwargs) -> None:
        self.model_args = None
        self.llm_engine = None

    def load(self) -> None:
        self.model_args = AsyncEngineArgs(model="Gryphe/MythoLogic-L2-13b")
        self.llm_engine = AsyncLLMEngine.from_engine_args(self.model_args)

    async def predict(self, model_input: Any) -> Any:
        prompt = model_input["prompt"]
        sampling_params = SamplingParams(temperature=0.8, top_p=1, max_tokens=500)
        idx = str(uuid.uuid4().hex)
        generator = self.llm_engine.generate(prompt, sampling_params, idx)
        prev = ""
        async for output in generator:
            curr = output.outputs[0].text
            delta = curr[len(prev):]
            prev = curr
            yield delta