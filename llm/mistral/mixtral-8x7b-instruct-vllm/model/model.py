import subprocess
import uuid

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from vllm import SamplingParams


class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.llm_engine = None
        self.model_args = None

        command = "ray start --head"
        subprocess.check_output(command, shell=True, text=True)

    def load(self):
        self.model_args = AsyncEngineArgs(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1", tensor_parallel_size=2
        )
        self.llm_engine = AsyncLLMEngine.from_engine_args(self.model_args)

    async def predict(self, model_input):
        prompt = model_input.pop("prompt")
        stream = model_input.pop("stream", True)

        sampling_params = SamplingParams(**model_input)
        idx = str(uuid.uuid4().hex)
        vllm_generator = self.llm_engine.generate(prompt, sampling_params, idx)

        async def generator():
            full_text = ""
            async for output in vllm_generator:
                text = output.outputs[0].text
                delta = text[len(full_text) :]
                full_text = text
                yield delta

        if stream:
            return generator()
        else:
            full_text = ""
            async for delta in generator():
                full_text += delta
            return {"text": full_text}
