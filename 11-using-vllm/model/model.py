import logging
import os
import subprocess
import uuid
import httpx
import time

from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)

class Model:
    MAX_FAILED_SECONDS = 600  # 10 minutes; the reason this would take this long is mostly if we download a large model
    
    def __init__(self, **kwargs):
        self._config = kwargs["config"]
        self.model = None
        self.llm_engine = None
        self.model_args = None
        self.hf_secret_token = kwargs["secrets"]["hf_access_token"]
        self.openai_compatible = self._config["model_metadata"]["openai_compatible"]
        self.vllm_base_url = None
        os.environ["HF_TOKEN"] = self.hf_secret_token

    def load(self):
        model_metadata = self._config["model_metadata"]
        model_repo_id = model_metadata['repo_id']
        self._vllm_config = model_metadata["vllm_config"]
        logger.info(f"main model: {model_repo_id}")
        logger.info(f"vllm config: {self._vllm_config}")
        if self.openai_compatible:
            self._client = httpx.AsyncClient(timeout=None)
            command = ["python3", "-m", "vllm.entrypoints.openai.api_server"]
            for key, value in self._vllm_config.items():
                if value is True:
                    command.append(f"--{key.replace('_', '-')}")
                elif value is False:
                    continue
                else:
                    command.append(f"--{key.replace('_', '-')}")
                    command.append(str(value)) 
            
            logger.info(f"Starting openai compatible vLLM server with command: {command}")
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=True)
                logger.info(f"Conmmand succeeded with output: {result.stdout}")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Command failed with code {e.returncode}: {e.stderr}")

            if "port" in self._vllm_config:
                self._vllm_port = self._vllm_config["port"]
            else:
                self._vllm_port = 8000

            self.vllm_base_url = f"http://localhost:{self._vllm_port}"

            # Polling to check if the server is up
            server_up = False
            start_time = time.time()
            while time.time() - start_time < self.MAX_FAILED_SECONDS:
                try:
                    response = httpx.get(f"{self.vllm_base_url}/health")
                    if response.status_code == 200:
                        server_up = True
                        break
                except httpx.RequestError:
                    time.sleep(1)  # Wait for 1 second before retrying

            if not server_up:
                raise RuntimeError(
                    "Server failed to start within the maximum allowed time."
                )
        else: 
            try:
                result = subprocess.run(
                    ["nvidia-smi"], capture_output=True, text=True, check=True
                )
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Command failed with code {e.returncode}: {e.stderr}")

            self.model_args = AsyncEngineArgs(model=model_repo_id, **self._vllm_config)
            self.llm_engine = AsyncLLMEngine.from_engine_args(self.model_args)
            self.tokenizer = AutoTokenizer.from_pretrained(model_repo_id)

            try:
                result = subprocess.run(
                    ["nvidia-smi"], capture_output=True, text=True, check=True
                )
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Command failed with code {e.returncode}: {e.stderr}")

    async def predict(self, model_input):
        if self.openai_compatible:
            # if the key metrics: true is present, let's return the vLLM /metrics endpoint
            if model_input.get("metrics", False):
                response = await self._client.get(f"{self.vllm_base_url}/metrics")
                return response.text

            # convenience for Baseten bridge
            if "model" not in model_input and "model" in self._vllm_config:
                print(
                    f"model_input missing model due to Baseten bridge, using {self._vllm_config['model']}"
                )
                model_input["model"] = self._vllm_config["model"]

            stream = model_input.get("stream", False)
            if stream:

                async def generator():
                    async with self._client.stream(
                        "POST",
                        f"{self.vllm_base_url}/v1/chat/completions",
                        json=model_input,
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                yield chunk

                return generator()
            else:
                response = await self._client.post(
                    f"{self.vllm_base_url}/v1/chat/completions",
                    json=model_input,
                )
                return response.json()
        else: 
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
