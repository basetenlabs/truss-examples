import json
import os
import subprocess
import time
from functools import partial
from pathlib import Path
import asyncio

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

from utils import (
    prepare_model_repository,
    GRPC_SERVICE_PORT,
    HTTP_SERVICE_PORT,
    server_loaded,
)

class ResponseManager():
    def __init__(self):
        self._reqs_to_queues = {}

    async def register(self, request_id):
        self._reqs_to_queues[request_id] = asyncio.Queue()

    async def process_response(
        self,
        response: grpcclient.InferResult,
        sync_callback_timestamp: float,
        async_callback_timestamp: float,
    ):
        def get_request_id():
            return response.get_response().id

        # Getting the request id is a blocking call, so we need to run it in a thread
        request_id = await asyncio.to_thread(get_request_id)
        await self._reqs_to_queues[request_id].put(
            (
                response, # The GRPC response object
                sync_callback_timestamp, # The time at which the sync callback was invoked
                async_callback_timestamp, # The time at which the async callback was invoked
                time.time() # The time at which the response was put into the queue
            )
        )

class UserData:
    def __init__(self):
        self._completed_requests = Queue()

class CallbackManager():
    def __init__(
        self,
        core_event_loop: asyncio.BaseEventLoop,
        response_manager: ResponseManager,
    ):
        self.response_manager = response_manager
        self._core_event_loop = core_event_loop

    async def callback(
        self,
        sync_callback_timestamp: float,
        result,
        error,
    ):
        async_callback_timestamp = time.time()
        response = result if not error else error
        await self.response_manager.process_response(response, sync_callback_timestamp, async_callback_timestamp)

    def get_sync_callback(self):
        def sync_callback(result, error):
            sync_callback_timestamp = time.time()
            asyncio.run_coroutine_threadsafe(
                self.callback(
                    sync_callback_timestamp,
                    result,
                    error
                ),
                self._core_event_loop
            )
        return partial(sync_callback)

class TritonClient:
    def __init__(self, data_dir: Path, model_repository_dir: Path, parallel_count=1):
        self._data_dir = data_dir
        self._model_repository_dir = model_repository_dir
        self._parallel_count = parallel_count
        self.response_manager = None
        self.callback_manager = None
        self._http_client = None
        self.grpc_client_instance = None

    def start_grpc_stream(self, core_event_loop: asyncio.BaseEventLoop):
        if self.response_manager and self.callback_manager:
            return

        self.response_manager = ResponseManager()
        self.callback_manager = CallbackManager(core_event_loop, self.response_manager)

        self.grpc_client_instance = grpcclient.InferenceServerClient(
            url=f"localhost:{GRPC_SERVICE_PORT}", verbose=False
        )
        self.grpc_client_instance.start_stream(callback=self.callback_manager.get_sync_callback())

    async def infer(self, inputs: list, request_id: str, model_name="ensemble"):
        await self.response_manager.register(request_id)
        self.grpc_client_instance.async_stream_infer(
                model_name,
                inputs,
                request_id=request_id,
                enable_empty_final_response=True
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
                "--log-verbose",
                "1",
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

    @staticmethod
    async def stream_predict(response_manager: ResponseManager, request_id: str):

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
                q = await response_manager.get_queue(request_id)
                result, sync_callback_timestamp, async_callback_timestamp, queue_put_time = await q.get()
                triton_completion_timestamp = result.as_numpy("time")[0]
                print(
                    f"""
                    GRPC sit time: {queue_put_time - triton_completion_timestamp}
                    Async callback time: {async_callback_timestamp - sync_callback_timestamp}
                    Sync callback time: {queue_put_time - sync_callback_timestamp}
                    Streaming time: {time.time() - queue_put_time}
                    Total time: {time.time() - triton_completion_timestamp}
                    """
                )
                if not isinstance(result, InferenceServerException):
                    res = result.as_numpy("text_output")
                    yield res[0].decode("utf-8")
                else:
                    yield json.dumps({"status": "error", "message": result.message()})
            except Exception as e:
                yield json.dumps({"status": "error", "message": str(e)})
                break
