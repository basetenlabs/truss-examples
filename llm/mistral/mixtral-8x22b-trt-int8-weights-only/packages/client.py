import json
import os
import subprocess
import time
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Thread

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from utils import (
    GRPC_SERVICE_PORT,
    HTTP_SERVICE_PORT,
    prepare_model_repository,
    server_loaded,
)


class UserData:
    def __init__(self):
        self._completed_requests = Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class TritonClient:
    def __init__(self, data_dir: Path, model_repository_dir: Path, parallel_count=1):
        self._data_dir = data_dir
        self._model_repository_dir = model_repository_dir
        self._parallel_count = parallel_count
        self._http_client = None
        self._grpc_client_map = {}

    def start_grpc_stream(self, user_data, model_name, inputs, stream_uuid):
        """Starts a GRPC stream and sends a request to the Triton server."""
        grpc_client_instance = grpcclient.InferenceServerClient(
            url=f"localhost:{GRPC_SERVICE_PORT}", verbose=False
        )
        self._grpc_client_map[stream_uuid] = grpc_client_instance
        grpc_client_instance.start_stream(callback=partial(callback, user_data))
        grpc_client_instance.async_stream_infer(
            model_name,
            inputs,
            request_id=stream_uuid,
            enable_empty_final_response=True,
        )

    def stop_grpc_stream(self, stream_uuid, stream_thread: Thread):
        """Closes a GRPC stream and stops the associated thread."""
        triton_grpc_stream = self._grpc_client_map[stream_uuid]
        triton_grpc_stream.stop_stream()
        stream_thread.join()
        del self._grpc_client_map[stream_uuid]

    def start_server(
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
                f"{GRPC_SERVICE_PORT}",
                "--http-port",
                f"{HTTP_SERVICE_PORT}",
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
                f"{GRPC_SERVICE_PORT}",
                "--http-port",
                f"{HTTP_SERVICE_PORT}",
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
        if not server_loaded():
            prepare_model_repository(self._data_dir)
            self.start_server(mpi=self._parallel_count, env=env)

        self._http_client = httpclient.InferenceServerClient(
            url=f"localhost:{HTTP_SERVICE_PORT}", verbose=False
        )
        is_server_up = False
        while not is_server_up:
            try:
                is_server_up = self._http_client.is_server_live()
            except ConnectionRefusedError:
                time.sleep(2)
                continue

        while self._http_client.is_model_ready(model_name="ensemble") == False:
            time.sleep(2)
            continue

    @staticmethod
    def stream_predict(user_data: UserData):
        """Static method to yield predictions or errors based on input and a streaming user_data queue."""

        def _is_final_response(result):
            """Check if the given result is a final response according to Triton's specification."""
            if isinstance(result, InferenceServerException):
                return True

            if result:
                final_response_param = result.get_response().parameters.get(
                    "triton_final_response"
                )
                return (
                    final_response_param.bool_param if final_response_param else False
                )
            return False

        result = None

        while not _is_final_response(result):
            try:
                result = user_data._completed_requests.get()
                if not isinstance(result, InferenceServerException):
                    res = result.as_numpy("text_output")
                    yield res[0].decode("utf-8")
                else:
                    yield json.dumps({"status": "error", "message": result.message()})
            except Exception as e:
                yield json.dumps({"status": "error", "message": str(e)})
                break
