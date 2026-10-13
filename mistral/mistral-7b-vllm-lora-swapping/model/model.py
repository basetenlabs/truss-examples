import uuid
from typing import Any

from huggingface_hub import snapshot_download

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.lora.request import LoRARequest


class Model:
    def __init__(self, **kwargs) -> None:
        self.engine_args = kwargs["config"]["model_metadata"]["engine_args"]
        self.prompt_format = kwargs["config"]["model_metadata"]["prompt_format"]
        self._data_dir = kwargs["data_dir"]
        self.seen_models = set({})
    

    def lora_model_cache_dir(self, model_id: str) -> str:
        return str(self._data_dir/ "loras"/ model_id)

    def load(self) -> None:
        self.llm_engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(**self.engine_args)
        )

    async def predict(self, request: dict) -> Any:
        prompt = request.pop("prompt")
        stream = request.pop("stream", True)
        lora_request = None
        lora_model = request.pop("lora_model", None)
        if lora_model is not None:
            lora_dir = self.lora_model_cache_dir(lora_model)
            if lora_model not in self.seen_models:
                # TODO: scoped down download to only required files and run in parallel
                snapshot_download(lora_model, local_dir=lora_dir)
                self.seen_models.add(lora_model)
            
            lora_request = LoRARequest(lora_model, abs(hash(lora_model)), lora_dir)
                
        formatted_prompt = self.prompt_format.replace("{prompt}", prompt)

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
        generate_args.update(request)

        sampling_params = SamplingParams(**generate_args)
        idx = str(uuid.uuid4().hex)
        vllm_generator = self.llm_engine.generate(
            formatted_prompt, sampling_params, request_id=idx, lora_request=lora_request
        )

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
