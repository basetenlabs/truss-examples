import os
from itertools import count
from pathlib import Path
from threading import Thread

import numpy as np
from client import TritonClient, UserData
from transformers import AutoTokenizer
from utils import download_engine, prepare_grpc_tensor, server_loaded

TRITON_MODEL_REPOSITORY_PATH = Path("/packages/inflight_batcher_llm/")


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._request_id_counter = count(start=1)
        self.triton_client = None
        self.tokenizer = None
        self.uses_openai_api = (
            "openai-compatible" in self._config["model_metadata"]["tags"]
        )

    def load(self):
        tensor_parallel_count = self._config["model_metadata"].get(
            "tensor_parallelism", 1
        )
        pipeline_parallel_count = self._config["model_metadata"].get(
            "pipeline_parallelism", 1
        )
        if "hf_access_token" in self._secrets._base_secrets.keys():
            hf_access_token = self._secrets["hf_access_token"]
        else:
            hf_access_token = None
        is_external_engine_repo = "engine_repository" in self._config["model_metadata"]

        # Instantiate TritonClient
        self.triton_client = TritonClient(
            data_dir=self._data_dir,
            model_repository_dir=TRITON_MODEL_REPOSITORY_PATH,
            parallel_count=tensor_parallel_count * pipeline_parallel_count,
        )

        # Download model from Hugging Face Hub if specified
        if is_external_engine_repo:
            if not server_loaded():
                download_engine(
                    engine_repository=self._config["model_metadata"][
                        "engine_repository"
                    ],
                    fp=self._data_dir,
                    auth_token=hf_access_token,
                )

        # Load Triton Server and model
        tokenizer_repository = self._config["model_metadata"]["tokenizer_repository"]
        env = {"triton_tokenizer_repository": tokenizer_repository}
        if hf_access_token is not None:
            env["HUGGING_FACE_HUB_TOKEN"] = hf_access_token

        self.triton_client.load_server_and_model(env=env)

        # setup eos token
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_repository, token=hf_access_token
        )
        self.eos_token_id = self.tokenizer.eos_token_id

    def predict(self, model_input):
        user_data = UserData()
        model_name = "ensemble"
        stream_uuid = str(os.getpid()) + str(next(self._request_id_counter))

        if self.uses_openai_api:
            prompt = self.tokenizer.apply_chat_template(
                model_input.get("messages"),
                tokenize=False,
            )
        else:
            prompt = model_input.get("prompt")

        max_tokens = model_input.get("max_tokens", 50)
        beam_width = model_input.get("beam_width", 1)
        bad_words_list = model_input.get("bad_words_list", [""])
        stop_words_list = model_input.get("stop_words_list", [""])
        repetition_penalty = model_input.get("repetition_penalty", 1.0)
        ignore_eos = model_input.get("ignore_eos", False)
        stream = model_input.get("stream", True)

        input0 = [[prompt]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.uint32) * max_tokens
        bad_words_list = np.array([bad_words_list], dtype=object)
        stop_words_list = np.array([stop_words_list], dtype=object)
        stream_data = np.array([[stream]], dtype=bool)
        beam_width_data = np.array([[beam_width]], dtype=np.uint32)
        repetition_penalty_data = np.array([[repetition_penalty]], dtype=np.float32)

        inputs = [
            prepare_grpc_tensor("text_input", input0_data),
            prepare_grpc_tensor("max_tokens", output0_len),
            prepare_grpc_tensor("bad_words", bad_words_list),
            prepare_grpc_tensor("stop_words", stop_words_list),
            prepare_grpc_tensor("stream", stream_data),
            prepare_grpc_tensor("beam_width", beam_width_data),
            prepare_grpc_tensor("repetition_penalty", repetition_penalty_data),
        ]

        if not ignore_eos:
            end_id_data = np.array([[self.eos_token_id]], dtype=np.uint32)
            inputs.append(prepare_grpc_tensor("end_id", end_id_data))
        else:
            # do nothing, trt-llm by default doesn't stop on `eos`
            pass

        # Start GRPC stream in a separate thread
        stream_thread = Thread(
            target=self.triton_client.start_grpc_stream,
            args=(user_data, model_name, inputs, stream_uuid),
        )
        stream_thread.start()

        def generate():
            # Yield results from the queue
            for i in TritonClient.stream_predict(user_data):
                yield i

            # Clean up GRPC stream and thread
            self.triton_client.stop_grpc_stream(stream_uuid, stream_thread)

        if stream:
            return generate()
        else:
            if self.uses_openai_api:
                return "".join(generate())
            else:
                return {"text": "".join(generate())}
