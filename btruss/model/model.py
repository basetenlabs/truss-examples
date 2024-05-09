from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import subprocess
import time
import grpc
import random
import socket

import briton_pb2
import briton_pb2_grpc
import asyncio


INVALID_UNICODE_CHAR = "�"
BRITON_PORT = 50051

MODEL_INPUT_TO_BRITON_FIELD = {
    "max_tokens": "request_output_len",
    "beam_width": "beam_width",
    "repetition_penalty": "repetition_penalty",
    "presence_penalty": "presence_penalty",
    "temperature": "temperature",
    "length_penalty": "len_penalty",
    "end_id": "end_id",
    "pad_id": "pad_id",
    "runtime_top_k": "runtime_top_k",
    "runtime_top_p": "runtime_top_p",
}

def is_port_available(port, host='localhost'):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Try to connect to the specified host and port
            s.bind((host, port))
            return True
    except OSError:
        # Port is not available
        return False


class Model:
    def __init__(self, **kwargs):
        self._model = None
        self._config = kwargs["config"]
        self._data_dir = kwargs["data_dir"]
        self._stub = None
        self._secrets = kwargs["secrets"]

    def load(self):
        hf_token = self._secrets["hf_access_token"]
        self._tokenizer = AutoTokenizer.from_pretrained(self._config["model_metadata"]["hf_tokenizer"], token=hf_token)
        snapshot_download(
            repo_id=self._config["model_metadata"]["engine_repository"],
            local_dir=str(self._data_dir / "engine"),
        )
        # Start engine
        config_str = f"engine_path: \"{self._data_dir.resolve()}/engine\""
        config_pbtxt_path = (self._data_dir / "config.pbtxt").resolve()
        config_pbtxt_path.write_text(config_str)
        subprocess.Popen(["Briton", "--config", str(config_pbtxt_path)])
        while is_port_available(BRITON_PORT):
            print('Waiting for Briton to start')
            time.sleep(1)

        # create grpc client

    async def predict(self, model_input):
        if self._stub is None:
            channel = grpc.aio.insecure_channel(f"localhost:{BRITON_PORT}")
            self._stub = briton_pb2_grpc.BritonStub(channel)

        # Run model inference here
        prompt = model_input["prompt"]
        if prompt == "":
            yield ""
            return

        input_ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        request = briton_pb2.InferenceRequest(
            request_id=random.randint(1, 100000), 
            input_ids=input_ids,
        )
        set_briton_request_fields_from_model_input(model_input, request)

        print('sending inference request')
        try:
            buffered_tokens, buffered_str_len = last_valid_unicode_tokens(input_ids, self._tokenizer)
            orig_buffered_tokens_len = len(buffered_tokens)
            async for response in self._stub.Infer(request):
                buffered_tokens.extend(response.output_ids)
                decoded = self._tokenizer.decode(buffered_tokens)
                if decoded == "" or decoded[-1] == INVALID_UNICODE_CHAR:
                    continue
                diff_str = decoded[buffered_str_len:]
                buffered_tokens = buffered_tokens[orig_buffered_tokens_len:]
                orig_buffered_tokens_len = len(buffered_tokens)
                buffered_str_len = len(self._tokenizer.decode(buffered_tokens))
                yield diff_str
        except Exception as ex:
            print(f"An error has occurred: {ex}")
            raise ex

def set_briton_request_fields_from_model_input(model_input, briton_request):
    for model_input_key, briton_field in MODEL_INPUT_TO_BRITON_FIELD.items():
        if model_input_key in model_input:
            model_input_value = model_input[model_input_key]
            setattr(briton_request, briton_field, model_input_value)


def last_valid_unicode_tokens(tokens, tokenizer):
    for i in range(len(tokens)-1, -1, -1):
        decoded = tokenizer.decode(tokens[i:])
        if decoded[-1] != INVALID_UNICODE_CHAR:
            return tokens[i:], len(decoded)
