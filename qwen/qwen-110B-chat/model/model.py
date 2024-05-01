import subprocess
import uuid
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


class Model:
    def __init__(self, model_name="Qwen/Qwen1.5-110B-Chat"):
        self.model_name = model_name
        self.tokenizer = None
        self.sampling_params = None

        command = "ray start --head"
        subprocess.check_output(command, shell=True, text=True)

    def load(self):
        self.model_args = AsyncEngineArgs(
            model=self.model_name,
            dtype='auto',
            enforce_eager=True,
            tensor_parallel_size=4

        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.sampling_params = SamplingParams(    # Using default values
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.05,
            max_tokens=512
        )

        self.llm_engine = AsyncLLMEngine.from_engine_args(self.model_args)

    async def predict(self, model_input):
        message = model_input.pop("prompt")

        prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ]

        text = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )

        idx = str(uuid.uuid4().hex)
        vllm_generator = self.llm_engine.generate(text, self.sampling_params, idx)

        async def generator():
            full_text = ""
            async for output in vllm_generator:
                text = output.outputs[0].text
                delta = text[len(full_text) :]
                full_text = text
                yield delta

        return generator()