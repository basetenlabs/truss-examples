from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm import TokensPrompt
from vllm.entrypoints.chat_utils import parse_chat_messages

from mistral_common.tokens.tokenizers.mistral import ChatCompletionRequest

import os
import uuid
import logging

logger = logging.getLogger(__name__)

model_name = "mistralai/Pixtral-12B-2409"


class Model:
    def __init__(self, **kwargs):
        self._config = kwargs["config"]
        self.llm_engine = None
        self.sample_params = None
        self.tokenizer = None

        os.environ["HF_TOKEN"] = kwargs["secrets"]["hf_access_token"]

    def load(self):
        self._vllm_config = self._config["model_metadata"].get("vllm_config", {})
        logger.info(f"vllm config: {self._vllm_config}")

        self.async_engine_args = AsyncEngineArgs(
            model=model_name, tokenizer_mode="mistral", **self._vllm_config
        )

        self.llm_engine = AsyncLLMEngine.from_engine_args(self.async_engine_args)

    async def predict(self, request):
        idx = str(uuid.uuid4().hex)

        sample_params = SamplingParams(
            max_tokens=request.get("max_tokens", 512),
            temperature=request.get("temperature", 0.7),
        )

        tokenizer = await self.llm_engine.get_tokenizer()
        tokenizer = tokenizer.mistral
        model_config = await self.llm_engine.get_model_config()

        _, mm_data = parse_chat_messages(request["messages"], model_config, tokenizer)
        chat_request = ChatCompletionRequest(messages=request["messages"])
        prompt = tokenizer.encode_chat_completion(chat_request)
        engine_inputs = TokensPrompt(prompt_token_ids=prompt.tokens)

        if mm_data is not None:
            engine_inputs["multi_modal_data"] = mm_data

        vllm_generator = self.llm_engine.generate(
            engine_inputs, sampling_params=sample_params, request_id=idx
        )

        async def generator():
            full_text = ""
            async for output in vllm_generator:
                text = output.outputs[0].text
                delta = text[len(full_text) :]
                full_text = text

                yield delta

        if request.get("stream", False):
            return generator()
        else:
            full_text = ""
            async for delta in generator():
                full_text += delta
            return {"text": full_text}
