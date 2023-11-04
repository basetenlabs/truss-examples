import os
import requests
import numpy as np
import subprocess
import time
from pathlib import Path
from functools import partial
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from tritonclient.utils import InferenceServerException
from queue import Queue
from threading import Thread


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
        self._triton_grpc_client_map = {}
        self._request_id_counter = 0
        self._secrets = kwargs["secrets"]
        self._hf_token = self._secrets["hf_access_token"]

    def move_all_files(self, src: Path, dest: Path):
            for item in src.iterdir():
                dest_item = dest / item.name

                if item.is_dir():
                    dest_item.mkdir(parents=True, exist_ok=True)
                    self.move_all_files(item, dest_item)
                else:
                    item.rename(dest_item)
    
    def start_grpc_stream(self, user_data, model_name, inputs, stream_uuid):
        grpc_client_instance = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)
        self._triton_grpc_client_map[stream_uuid] = grpc_client_instance
        grpc_client_instance.start_stream(callback=partial(callback, user_data))
        grpc_client_instance.async_stream_infer(
            model_name,
            inputs,
            request_id=stream_uuid,
            enable_empty_final_response=True,
        )

    def load(self):
        engine_files = {
            "gpt_float16_tp2_rank0.engine": "https://huggingface.co/baseten/starcoder7b-fp16-engine/resolve/main/gpt_float16_tp2_rank0.engine",
            "gpt_float16_tp2_rank1.engine": "https://huggingface.co/baseten/starcoder7b-fp16-engine/resolve/main/gpt_float16_tp2_rank1.engine",
            "config.json": "https://huggingface.co/baseten/starcoder7b-fp16-engine/resolve/main/config.json",
        }
        for name, engine_url in engine_files.items():
            with open(self._data_dir / name, "wb") as f:
                f.write(requests.get(engine_url).content)

        # Ensure the destination directory exists
        dest_dir = Path("/packages/inflight_batcher_llm/tensorrt_llm/1")
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure empty version directory for `ensemble` model exists
        ensemble_dir = Path("/packages/inflight_batcher_llm/ensemble/1")
        ensemble_dir.mkdir(parents=True, exist_ok=True)

        # Move all files and directories from data_dir to dest_dir
        self.move_all_files(self._data_dir, dest_dir)

        # Kick off Triton Inference Server
        command = [
            'mpirun',
            '--allow-run-as-root',
            '-n',
            '1',
            'tritonserver',
            '--model-repository=/packages/inflight_batcher_llm/',
            '--grpc-port=8001',
            '--http-port=8003',
            '--disable-auto-complete-config',
            '--backend-config=python,shm-region-prefix-name=prefix0_',
            ':',
            '-n',
            '1',
            'tritonserver',
            '--model-repository=/packages/inflight_batcher_llm/',
            '--grpc-port=8001',
            '--http-port=8003',
            '--disable-auto-complete-config',
            '--backend-config=python,shm-region-prefix-name=prefix1_',
            ':'
        ]
        process = subprocess.Popen(
            command, 
            env={ **os.environ, "HUGGING_FACE_HUB_TOKEN": self._hf_token},
        )
        
        # Create Triton HTTP Client and GRPC Client, retrying every 10 seconds until successful
        self._triton_http_client = httpclient.InferenceServerClient(url="localhost:8003", verbose=False)
        
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

    def predict(self, model_input):
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

        # Start GRPC stream in a separate thread
        stream_uuid = str(self._request_id_counter)
        stream_thread = Thread(target=self.start_grpc_stream, args=(user_data, model_name, inputs, stream_uuid))
        stream_thread.start()

        while True:
            try:
                result = user_data._completed_requests.get()
                if not isinstance(result, InferenceServerException):
                    res = result.as_numpy('text_output')
                    yield res[0].decode("utf-8")
                else:
                    yield {"status": "error", "message": result.message()}
                
                if result.get_response().parameters.get("triton_final_response") and \
                        result.get_response().parameters["triton_final_response"].bool_param:
                    triton_grpc_stream = self._triton_grpc_client_map[stream_uuid]
                    triton_grpc_stream.stop_stream()
                    break
            except Exception as e:
                triton_grpc_stream = self._triton_grpc_client_map[stream_uuid]
                triton_grpc_stream.stop_stream()
                yield {"status": "error", "message": str(e)}
                break

        # Join the streaming thread to ensure all resources are released
        stream_thread.join()
        
        # Delete the GRPC client instance
        del self._triton_grpc_client_map[stream_uuid]
