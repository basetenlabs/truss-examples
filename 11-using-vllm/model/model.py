import json
import subprocess
import time
from typing import Any, Dict, List

import httpx  # Changed from aiohttp to httpx


class Model:
    MAX_FAILED_SECONDS = 600  # 10 minutes; the reason this would take this long is mostly if we download a large model

    def __init__(self, data_dir, config, secrets):
        self._secrets = secrets
        self._config = config
        self.vllm_base_url = None

        # TODO: uncomment for multi-GPU support
        # command = "ray start --head"
        # subprocess.check_output(command, shell=True, text=True)

    def load(self):
        self._client = httpx.AsyncClient(timeout=None)

        self._vllm_config = self._config["model_metadata"]["arguments"]

        command = ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        for key, value in self._vllm_config.items():
            command.append(f"--{key.replace('_', '-')}")
            command.append(str(value))

        print(f"[DEBUG] Starting vLLM with command: {command}")

        subprocess.Popen(command)

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

    async def predict(self, model_input):

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
