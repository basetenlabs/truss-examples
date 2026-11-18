import logging
import subprocess
import uuid

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, **kwargs):
        self._config = kwargs["config"]
        self.model = None
        self.llm_engine = None
        self.model_args = None

        num_gpus = self._config["model_metadata"]["tensor_parallel"]
        logger.info(f"num GPUs ray: {num_gpus}")
        command = f"ray start --head --num-gpus={num_gpus}"
        subprocess.check_output(command, shell=True, text=True)

    def load(self):
        model_metadata = self._config["model_metadata"]
        logger.info(f"main model: {model_metadata['main_model']}")
        logger.info(f"assistant model: {model_metadata['assistant_model']}")
        logger.info(f"tensor parallelism: {model_metadata['tensor_parallel']}")
        logger.info(f"max num seqs: {model_metadata['max_num_seqs']}")

        self.model_args = AsyncEngineArgs(
            model=model_metadata["main_model"],
            speculative_model=model_metadata["assistant_model"],
            trust_remote_code=True,
            tensor_parallel_size=model_metadata["tensor_parallel"],
            max_num_seqs=model_metadata["max_num_seqs"],
            dtype="half",
            use_v2_block_manager=True,
            enforce_eager=True,
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
