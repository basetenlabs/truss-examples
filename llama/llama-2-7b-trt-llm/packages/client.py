import json
import os
import subprocess
import time
from functools import partial
from pathlib import Path
from threading import Thread

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from utils import prepare_model_repository
import asyncio

class ResponseManager():
    def __init__(self):
        self._reqs_to_queues = {}

    async def register(self, request_id):
        self._reqs_to_queues[request_id] = asyncio.Queue()

    async def get_request_id_from_response(self, response):
        def get_id():
            return response.get_response().id
        return await asyncio.to_thread(get_id)

#    async def process_response(self, response_grpc, init_sync_cb_time, init_async_cb_time):
    async def process_response(self, response_grpc):
        request_id = await self.get_request_id_from_response(response_grpc)

        await self._reqs_to_queues[request_id].put(
            (
                response_grpc, # The GRPC response object
                # init_sync_cb_time, # The time at which the sync callback was invoked
                # init_async_cb_time, 
                time.time()
            )
        )

    async def get_queue(self, request_id):
        return self._reqs_to_queues[request_id]


# async def callback(response_manager, init_sync_cb_time, result, error):
async def callback(response_manager, result, error):  
    """
    The async callback invoked from 'sync_callback'
    """
    if error:
        # await response_manager.process_response(error, init_sync_cb_time, init_async_cb_time)
        await response_manager.process_response(error)
    else:
        await response_manager.process_response(result)

def sync_callback(
    response_manager : ResponseManager,
    loop: asyncio.BaseEventLoop,
    result,
    error,
):
    # init_sync_callback_time = time.time()
    # asyncio.run_coroutine_threadsafe(callback(response_manager, init_sync_callback_time, result, error), loop)
    asyncio.run_coroutine_threadsafe(callback(response_manager, result, error), loop)

class TritonClient:
    def __init__(self, data_dir: Path, model_repository_dir: Path, parallel_count=1):
        self._data_dir = data_dir
        self._model_repository_dir = model_repository_dir
        self._parallel_count = parallel_count
        self._http_client = None
        self._stream_started = False
        self.grpc_client_instance = None
        self.response_manager = ResponseManager()

    def start_grpc_stream(self, loop: asyncio.BaseEventLoop):
        """Starts a GRPC stream and initializes the callback"""
        if self._stream_started:
            return

        self.grpc_client_instance = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=False
        )
        self.grpc_client_instance.start_stream(callback=partial(sync_callback, self.response_manager, loop))
        self._stream_started = True

    async def send_inference(self, inputs, request_id, model_name="ensemble"):
        await self.response_manager.register(request_id)

        self.grpc_client_instance.async_stream_infer(
                model_name,
                inputs,
                request_id=request_id,
                enable_empty_final_response=True
        )

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
                "8001",
                "--http-port",
                "8003",
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
                "8001",
                "--http-port",
                "8003",
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
        self.start_server(mpi=self._parallel_count, env=env)

        self._http_client = httpclient.InferenceServerClient(
            url="localhost:8003", verbose=False
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
    async def stream_predict(manager, request_id):
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
                q = await manager.get_queue(request_id)
                result, put_time = await q.get()
                # response_grpc, response_json, triton_completion_time, init_sync_cb_time, init_async_cb_time, put_time = await q.get()
                # print(f"GRPC stream sit time: {init_sync_cb_time - triton_completion_time}. Sync cb to async cb: {init_async_cb_time - init_sync_cb_time}. Async cb to queue put time: {put_time - init_async_cb_time}. Put time to stream time: {time.time() - put_time}")

                if not isinstance(result, InferenceServerException):
                    res = result.as_numpy("text_output")
                    # time = result.as_numpy("time")
                    # res = result.as_numpy("text_output")
                    yield res[0].decode("utf-8")
                else:
                    yield json.dumps({"status": "error", "message": result.message()})

            except Exception as e:
                yield json.dumps({"status": "error", "message": str(e)})
                break