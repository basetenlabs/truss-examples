from typing import Any
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import uuid


class Model:
    def __init__(self, **kwargs) -> None:
        self.model_args = None
        self.llm_engine = None

    def load(self) -> None:
        self.model_args = AsyncEngineArgs(model="TheBloke/wizardLM-7B-HF")
        self.llm_engine = AsyncLLMEngine.from_engine_args(self.model_args)

    async def predict(self, request: dict) -> Any:
        prompt = request.pop("prompt")
        sampling_params = SamplingParams(**request)
        idx = str(uuid.uuid4().hex)
        generator = self.llm_engine.generate(prompt, sampling_params, idx)
        prev = ""
        async for output in generator:
            curr = output.outputs[0].text
            delta = curr[len(prev) :]
            prev = curr
            yield delta
