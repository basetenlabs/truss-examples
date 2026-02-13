import logging
import os
import subprocess
import uuid

from model.sighelper import patch
from transformers import AutoTokenizer

patch()

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from vllm import SamplingParams

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = (
    "spawn"  # for multiprocessing to work with CUDA
)
logger = logging.getLogger(__name__)


class Model:
    def __init__(self, **kwargs):
        self._config = kwargs["config"]
        self.model = None
        self.llm_engine = None
        self.model_args = None
        self.hf_secret_token = kwargs["secrets"]["hf_access_token"]
        os.environ["HF_TOKEN"] = self.hf_secret_token

    def load(self):
        model_metadata = self._config["model_metadata"]
        # logger.info(f"main model: {model_metadata['repo_id']}")
        # logger.info(f"tensor parallelism: {model_metadata['tensor_parallel']}")
        # logger.info(f"max num seqs: {model_metadata['max_num_seqs']}")

        self.model_args = AsyncEngineArgs(
            model="meta-llama/Llama-3.1-70B-Instruct",
            trust_remote_code=True,
            tensor_parallel_size=4,
            max_num_seqs=8,
            max_model_len=50000,
            dtype="auto",
            use_v2_block_manager=True,
            enforce_eager=True,
            enable_chunked_prefill=False,
        )

        self.llm_engine = AsyncLLMEngine.from_engine_args(self.model_args)
        # create tokenizer for gemma 2 to apply chat template to prompts

        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-70B-Instruct"
        )

        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with code {e.returncode}: {e.stderr}")

    async def predict(self, model_input):
        prompt = model_input.pop("prompt")
        stream = model_input.pop("stream", True)

        sampling_params = SamplingParams(**model_input)
        idx = str(uuid.uuid4().hex)
        chat = [
            {"role": "user", "content": prompt},
        ]
        # templatize the input to the model
        input = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        # since we accept any valid vllm sampling parameters, we can just pass it through

        vllm_generator = self.llm_engine.generate(input, sampling_params, idx)

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
