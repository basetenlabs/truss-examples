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
        self.model_args = AsyncEngineArgs(model="mistralai/Mistral-7B-Instruct-v0.1",)
        self.llm_engine = AsyncLLMEngine.from_engine_args(self.model_args)
    
    def preprocess(self, request: dict):
        generate_args = {
            "n": 1,
            "best_of": 1,
            "max_tokens": 512,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
            "frequency_penalty": 1.0,
            "presence_penalty": 1.0,
            "use_beam_search": False,
        }
        if "max_tokens" in request.keys():
            generate_args["max_tokens"] = request["max_tokens"]
        if "temperature" in request.keys():
            generate_args["temperature"] = request["temperature"]
        if "top_p" in request.keys():
            generate_args["top_p"] = request["top_p"]
        if "top_k" in request.keys():
            generate_args["top_k"] = request["top_k"]
        if "frequency_penalty" in request.keys():
            generate_args["frequency_penalty"] = request["frequency_penalty"]
        if "presence_penalty" in request.keys():
            generate_args["presence_penalty"] = request["presence_penalty"]
        if "use_beam_search" in request.keys():
            generate_args["use_beam_search"] = request["use_beam_search"]
        request = {**request, **generate_args}
        return request

    async def predict(self, request: dict) -> Any:
        prompt = request.pop('prompt')
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        sampling_params = SamplingParams(**request)
        idx = str(uuid.uuid4().hex)
        generator = self.llm_engine.generate(formatted_prompt, sampling_params, idx)
        
        full_text = ""
        async for output in generator:
            text = output.outputs[0].text
            delta = text[len(full_text):]
            full_text = text
            yield delta