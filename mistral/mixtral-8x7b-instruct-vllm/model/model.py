import subprocess
import uuid

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


class Model:
    def __init__(self, **kwargs):
        self.model = None
        self.llm_engine = None

        # Install megablocks and run ray process
        subprocess.run(["pip", "install", "megablocks"])
        command = "ray start --head"
        subprocess.check_output(command, shell=True, text=True)

    def load(self):
        self.model_args = AsyncEngineArgs(model="mistralai/Mixtral-8x7B-Instruct-v0.1")
        self.llm_engine = AsyncLLMEngine.from_engine_args(self.model_args)

    async def predict(self, model_input):
        prompt = model_input.pop("prompt")
        sampling_params = SamplingParams(**model_input)
        idx = str(uuid.uuid4().hex)
        generator = self.llm_engine.generate(prompt, sampling_params, idx)

        # stream output from generator
        prev = ""
        async for output in generator:
            curr = output.outputs[0].text
            delta = curr[len(prev) :]
            prev = curr
            yield delta
