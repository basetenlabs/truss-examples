import logging
import os
import subprocess

from sglang.srt.server import Runtime
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, **kwargs):
        self._config = kwargs["config"]
        self.llm_engine = None
        self.hf_secret_token = kwargs["secrets"]["hf_access_token"]
        os.environ["HF_TOKEN"] = self.hf_secret_token

    @staticmethod
    def get_gpu_memory():
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.info(f"Command failed with code {e.returncode}: {e.stderr}")

    def load(self):
        self.get_gpu_memory()
        model_metadata = self._config["model_metadata"]
        logger.info(f"main model: {model_metadata['repo_id']}")
        logger.info(f"tensor parallelism: {model_metadata['tensor_parallel']}")

        self.llm_engine = Runtime(
            model_path=model_metadata["repo_id"],
            tp_size=model_metadata["tensor_parallel"],
            dtype="auto",
            port=8000,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_metadata["repo_id"])

        self.get_gpu_memory()

    @staticmethod
    def convert_payload_to_sampling_params(payload: dict) -> dict:
        field_mapping = {
            "max_tokens": "max_new_tokens",
            "min_tokens": "min_new_tokens",
            "stop": "stop",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "frequency_penalty": "frequency_penalty",
            "presence_penalty": "presence_penalty",
            "ignore_eos": "ignore_eos",
        }
        return {field_mapping.get(key, key): value for key, value in payload.items()}

    async def predict(self, model_input):
        prompt = model_input.pop("prompt")
        stream = model_input.pop("stream", True)

        sampling_params = self.convert_payload_to_sampling_params(model_input)
        chat = [
            {"role": "user", "content": prompt},
        ]

        input = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        sglang_generator = self.llm_engine.async_generate(input, sampling_params)

        async def generator():
            full_text = ""
            async for output in sglang_generator:
                text = output["text"]
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
