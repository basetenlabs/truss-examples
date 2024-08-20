import asyncio
import logging
import os
import subprocess
import threading
import time
import uuid

import httpx
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

from vllm import SamplingParams
from vllm.model.helper import run_background_vllm_health_check

os.environ["TOKENIZERS_PARALLELISM"] = "true"

logger = logging.getLogger(__name__)


class Model:
    # TODO: better way to detect model server startup failure than using `MAX_FAILED_SECONDS`
    MAX_FAILED_SECONDS = 1500  # 25 minutes; the reason this would take this long is mostly if we download a large model
    HEALTH_CHECK_INTERVAL = 5  # seconds

    def __init__(self, **kwargs):
        self._config = kwargs["config"]
        self.model_id = None
        self.llm_engine = None
        self.model_args = None
        self.hf_secret_token = kwargs["secrets"].get("hf_access_token", None)
        self.openai_compatible = self._config["model_metadata"].get(
            "openai_compatible", False
        )
        self.vllm_base_url = None
        os.environ["HF_TOKEN"] = self.hf_secret_token

    def load(self):
        self._model_metadata = self._config["model_metadata"]
        self._model_repo_id = self._model_metadata["repo_id"]
        self._vllm_config = self._model_metadata["vllm_config"]
        if self._vllm_config is None:
            self._vllm_config = {}
        logger.info(f"main model: {self._model_repo_id}")
        logger.info(f"vllm config: {self._vllm_config}")
        if self.openai_compatible:
            self._client = httpx.AsyncClient(timeout=None)
            command = ["vllm", "serve", self._model_repo_id]
            for key, value in self._vllm_config.items():
                if value is True:
                    command.append(f"--{key.replace('_', '-')}")
                elif value is False:
                    continue
                else:
                    command.append(f"--{key.replace('_', '-')}")
                    command.append(str(value))

            logger.info(
                f"Starting openai compatible vLLM server with command: {command}"
            )

            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Wait for 10 seconds and check if command fails
            time.sleep(10)

            if process.poll() is None:
                logger.info("Command to start vLLM server ran successfully")
            else:
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    logger.error(f"Command failed with error: {stderr}")
                    raise RuntimeError(
                        f"Command failed with code {process.returncode}: {stderr}"
                    )

            if self._vllm_config and "port" in self._vllm_config:
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
                    logger.info(f"Checking server health: {response.status_code}")
                    if response.status_code == 200:
                        server_up = True
                        break
                except httpx.RequestError as e:
                    seconds_passed = int(time.time() - start_time)
                    if seconds_passed % 10 == 0:
                        logger.info(
                            f"Server is starting for {seconds_passed} seconds: {e}"
                        )
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
                logger.info(result.stdout)
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with code {e.returncode}: {e.stderr}")

            self.model_args = AsyncEngineArgs(
                model=self._model_repo_id, **self._vllm_config
            )
            self.llm_engine = AsyncLLMEngine.from_engine_args(
                engine_args=self.model_args
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self._model_repo_id)

            try:
                result = subprocess.run(
                    ["nvidia-smi"], capture_output=True, text=True, check=True
                )
                logger.info(result.stdout)
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with code {e.returncode}: {e.stderr}")
        try:
            run_background_vllm_health_check(
                self.openai_compatible,
                self.HEALTH_CHECK_INTERVAL,
                self.llm_engine,
                self.vllm_base_url,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start background health check: {e}")

    async def predict(self, model_input):
        if "messages" not in model_input and "prompt" not in model_input:
            raise ValueError("Prompt or messages must be provided")

        stream = model_input.pop("stream", False)
        if self.openai_compatible:
            # if the key metrics: true is present, let's return the vLLM /metrics endpoint
            if model_input.get("metrics", False):
                response = await self._client.get(f"{self.vllm_base_url}/metrics")
                return response.text

            # convenience for Baseten bridge
            if "model" not in model_input and self._model_repo_id:
                logger.info(
                    f"model_input missing model due to Baseten bridge, using {self._model_repo_id}"
                )
                model_input["model"] = self._model_repo_id

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
            # SamplingParams does not take/use argument 'model'
            if "model" in model_input:
                model_input.pop("model")
            if "prompt" in model_input:
                prompt = model_input.pop("prompt")
                sampling_params = SamplingParams(**model_input)
                idx = str(uuid.uuid4().hex)
                messages = [
                    {"role": "user", "content": prompt},
                ]
                # templatize the input to the model
                input = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            elif "messages" in model_input:
                messages = model_input.pop("messages")
                sampling_params = SamplingParams(**model_input)
                idx = str(uuid.uuid4().hex)
                # templatize the input to the model
                input = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                )
            logger.info(f"Using SamplingParams: {sampling_params}")
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
