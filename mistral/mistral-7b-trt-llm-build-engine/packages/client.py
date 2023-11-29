import json
import os
import subprocess
import time
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread

import tritonclient.grpc.aio as grpcclient
import tritonclient.http as httpclient
from utils import prepare_model_repository


class TritonClient:
    def __init__(self, data_dir: Path, model_repository_dir: Path, parallel_count=1):
        self._data_dir = data_dir
        self._model_repository_dir = model_repository_dir
        self._parallel_count = parallel_count
        self._triton_grpc_client = None

    def _start_server(
        self,
        mpi: int = 1,
        env: dict = {},
    ):
        """Triton Inference Server has different startup commands depending on
        whether it is running in a TP=1 or TP>1 configuration. This function
        starts the server with the appropriate command."""
        if mpi == 1:
            command = [
                "tritonserver",
                "--model-repository",
                str(self._model_repository_dir),
                "--grpc-port",
                "8001",
                "--http-port",
                "8003",
                "--log-verbose",
                "2",
            ]
        command = [
            "mpirun",
            "--allow-run-as-root",
        ]
        for i in range(mpi):
            command += [
                "-n",
                "1",
                "tritonserver",
                "--model-repository",
                str(self._model_repository_dir),
                "--grpc-port",
                "8001",
                "--http-port",
                "8003",
                "--log-verbose",
                "2",
                "--disable-auto-complete-config",
                f"--backend-config=python,shm-region-prefix-name=prefix{str(i)}_",
                ":",
            ]
        return subprocess.Popen(
            command,
            env={**os.environ, **env},
        )

    def load_server_and_model(self, env: dict):
        """Loads the Triton server and the model."""
        prepare_model_repository(self._data_dir)
        self._start_server(mpi=self._parallel_count, env=env)

        http_client = httpclient.InferenceServerClient(
            url="localhost:8003", verbose=False
        )
        is_server_up = False
        while not is_server_up:
            try:
                is_server_up = http_client.is_server_live()
            except ConnectionRefusedError:
                time.sleep(2)
                continue

        while http_client.is_model_ready(model_name="ensemble") == False:
            time.sleep(2)
            continue
        self._triton_grpc_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=False
        )
        return self._triton_grpc_client
