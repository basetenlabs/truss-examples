import subprocess
from typing import Any, Dict, List, Optional

class Config:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        required_keys = ["model_id"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")

    def get_command(self) -> List[str]:
        command = [
            "/usr/local/bin/text-embeddings-router",
            "--port", "80",
            "--model-id", self.config['model_id']
        ]

        optional_params = {
            "revision": "--revision",
            "tokenization_workers": "--tokenization-workers",
            "dtype": "--dtype",
            "pooling": "--pooling",
            "max_concurrent_requests": "--max-concurrent-requests",
            "max_batch_tokens": "--max-batch-tokens",
            "max_batch_requests": "--max-batch-requests",
            "max_client_batch_size": "--max-client-batch-size",
            "hf_api_token": "--hf-api-token",
            "uds_path": "--uds-path",
            "huggingface_hub_cache": "--huggingface-hub-cache",
            "payload_limit": "--payload-limit",
            "api_key": "--api-key",
            "json_output": "--json-output",
            "otlp_endpoint": "--otlp-endpoint",
        }

        for key, param in optional_params.items():
            value = self.config.get(key)
            if value is not None:
                command.extend([param, str(value)])

        return command

    def run_router(self):
        command = self.get_command()
        with open('/var/log/text_embeddings_router.log', 'w') as log_file:
            subprocess.Popen(command, stdout=log_file, stderr=log_file)
