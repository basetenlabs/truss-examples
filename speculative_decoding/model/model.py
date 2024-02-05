import os
from itertools import count

import helpers
import huggingface_hub

TRITON_DIR = os.path.join("/", "triton_model_repo")
DRAFT_MODEL_KEY = "draft_model"


class Model:
    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._request_id_counter = count(start=1)
        self._triton_server = None
        self._triton_client = None
        if "openai-compatible" in self._config["model_metadata"]["tags"]:
            raise NotImplementedError("openai-compatible")

    def load(self):
        tensor_parallel_count = self._config["model_metadata"].get(
            "tensor_parallelism", 1
        )
        pipeline_parallel_count = self._config["model_metadata"].get(
            "pipeline_parallelism", 1
        )

        env = {}
        if "hf_access_token" in self._secrets._base_secrets.keys():
            hf_access_token = self._secrets["hf_access_token"]
            env["HUGGING_FACE_HUB_TOKEN"] = hf_access_token
        else:
            hf_access_token = None

        # Target model.
        huggingface_hub.snapshot_download(
            self._config["model_metadata"]["engine_repository"],
            local_dir=os.path.join(TRITON_DIR, DRAFT_MODEL_KEY, "1"),
            local_dir_use_symlinks=False,
            max_workers=4,
            **(
                {"use_auth_token": hf_access_token}
                if hf_access_token is not None
                else {}
            )
        )
        # Draft model.
        huggingface_hub.snapshot_download(
            self._config["model_metadata"]["speculative_decoding"]["engine_repository"],
            local_dir=os.path.join(TRITON_DIR, DRAFT_MODEL_KEY, "1"),
            local_dir_use_symlinks=False,
            max_workers=4,
            **(
                {"use_auth_token": hf_access_token}
                if hf_access_token is not None
                else {}
            )
        )

        if not helpers.is_triton_server_alive():
            self._triton_server = helpers.TritonServer(
                "/root/workbench/truss-examples/speculative_decoding/triton_model_repo",
                parallel_count=tensor_parallel_count * pipeline_parallel_count,
            )
            self._triton_server.load_server_and_model(env)

    async def predict(self, model_input):
        stream_uuid = str(os.getpid()) + str(next(self._request_id_counter))
        prompt = model_input.get("prompt")

        max_tokens = model_input.get("max_tokens", 50)
        beam_width = model_input.get("beam_width", 1)
        bad_words_list = model_input.get("bad_words_list", [""])
        stop_words_list = model_input.get("stop_words_list", [""])
        repetition_penalty = model_input.get("repetition_penalty", 1.0)
        ignore_eos = model_input.get("ignore_eos", False)
        stream = model_input.get("stream", True)

        # TODO: make async/stream.
