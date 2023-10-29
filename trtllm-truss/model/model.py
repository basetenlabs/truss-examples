import numpy as np
import subprocess
import shutil
import time
from pathlib import Path
from functools import partial
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from tritonclient.utils import InferenceServerException
from queue import Queue


class UserData:
    def __init__(self):
        self._completed_requests = Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)

class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._triton_http_client = None
        self._triton_grpc_client = None
        self._request_id_counter = 0

    def move_all_files(self, src: Path, dest: Path):
            for item in src.iterdir():
                dest_item = dest / item.name

                if item.is_dir():
                    dest_item.mkdir(parents=True, exist_ok=True)
                    self.move_all_files(item, dest_item)
                else:
                    item.rename(dest_item)

    def load(self):
        # Ensure the destination directory exists
        dest_dir = Path("/packages/inflight_batcher_llm/tensorrt_llm/1")
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure empty version directory for `ensemble` model exists
        ensemble_dir = Path("/packages/inflight_batcher_llm/ensemble/1")
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Move all files and directories from data_dir to dest_dir
        self.move_all_files(self._data_dir, dest_dir)

        # Kick off Triton Inference Server
        process = subprocess.Popen(
            [
                "tritonserver",
                "--model-repository", "/packages/inflight_batcher_llm/",
                "--grpc-port", "8001",
                "--http-port", "8003"
            ]
        )
        
        # Create Triton HTTP Client and GRPC Client, retrying every 10 seconds until successful
        self._triton_http_client = httpclient.InferenceServerClient(url="localhost:8003", verbose=False)
        self._triton_grpc_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)
        
        # Before checking if model is ready, wait for the server to come up
        is_server_up = False
        while not is_server_up:
            try:
                is_server_up = self._triton_http_client.is_server_live()
            except ConnectionRefusedError:
                time.sleep(2)
                continue

        # Wait for model to load into Triton Inference Server
        while self._triton_http_client.is_model_ready(model_name="ensemble") == False:
            time.sleep(2)
            continue

    def prepare_tensor(self, name, input):
        t = grpcclient.InferInput(name, input.shape,
                                np_to_triton_dtype(input.dtype))
        t.set_data_from_numpy(input)
        return t
    
    async def inner(self, user_data):
        while True:
            try:
                result = user_data._completed_requests.get()
                if not isinstance(result, InferenceServerException):
                    res = result.as_numpy('text_output')
                    yield res[0].decode("utf-8")
                else:
                    yield {"status": "error", "message": result.message()}
                if result.get_response().parameters["triton_final_response"].bool_param:
                    break
            except Exception:
                break

    async def predict(self, model_input):
        model_name = "ensemble"
        user_data = UserData()
        self._request_id_counter += 1

        prompt = model_input.get("text_input")
        output_len = model_input.get("output_len", 256)
        beam_width = model_input.get("beam_width", 1)
        bad_words_list = model_input.get("bad_words_list", [""])
        stop_words_list = model_input.get("stop_words_list", [""])

        input0 = [[prompt]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.uint32) * output_len
        bad_words_list = np.array([bad_words_list], dtype=object)
        stop_words_list = np.array([stop_words_list], dtype=object)
        streaming = [[True]]
        streaming_data = np.array(streaming, dtype=bool)
        beam_width = [[beam_width]]
        beam_width_data = np.array(beam_width, dtype=np.uint32)

        inputs = [
            self.prepare_tensor("text_input", input0_data),
            self.prepare_tensor("max_tokens", output0_len),
            self.prepare_tensor("bad_words", bad_words_list),
            self.prepare_tensor("stop_words", stop_words_list),
            self.prepare_tensor("stream", streaming_data),
            self.prepare_tensor("beam_width", beam_width_data),
        ]

        self._triton_grpc_client.start_stream(callback=partial(callback, user_data))
        self._triton_grpc_client.async_stream_infer(
            model_name,
            inputs,
            request_id=str(self._request_id_counter),
            enable_empty_final_response=True,
        )

        return self.inner(user_data)