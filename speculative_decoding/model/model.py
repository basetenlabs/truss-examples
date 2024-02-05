import asyncio
import os
from itertools import count

import huggingface_hub
import transformers
import tritonclient.grpc.aio as triton_grpc

from . import helpers, spec_dec

TRITON_DIR = os.path.join("/", "triton_model_repo")
TARGET_MODEL_KEY = "target_model"
DRAFT_MODEL_KEY = "draft_model"


class Model:
    def __init__(self, **kwargs):
        # self._data_dir = kwargs["data_dir"]
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
        # if "hf_access_token" in self._secrets._base_secrets.keys():
        #     hf_access_token = self._secrets["hf_access_token"]
        #     env["HUGGING_FACE_HUB_TOKEN"] = hf_access_token
        # else:
        #     hf_access_token = None

        hf_access_token = None  # TODO: dbg

        # Target model.
        huggingface_hub.snapshot_download(
            self._config["model_metadata"]["engine_repository"],
            local_dir=os.path.join(TRITON_DIR, TARGET_MODEL_KEY, "1"),
            local_dir_use_symlinks=True,  # TODO: dbg
            max_workers=4,
            **(
                {"use_auth_token": hf_access_token}
                if hf_access_token is not None
                else {}
            )
        )
        # Draft model.
        huggingface_hub.snapshot_download(
            self._config["model_metadata"]["speculative_decoding"][
                "draft_engine_repository"
            ],
            local_dir=os.path.join(TRITON_DIR, DRAFT_MODEL_KEY, "1"),
            local_dir_use_symlinks=True,  # TODO: dbg
            max_workers=4,
            **(
                {"use_auth_token": hf_access_token}
                if hf_access_token is not None
                else {}
            )
        )

        if not helpers.is_triton_server_alive():
            self._triton_server = helpers.TritonServer(
                TRITON_DIR,
                parallel_count=tensor_parallel_count * pipeline_parallel_count,
            )
            self._triton_server.load_server_and_model(env)

        client = triton_grpc.InferenceServerClient("0.0.0.0:8001")
        self._target_model = spec_dec.ModelWrapper(
            client,
            TARGET_MODEL_KEY,
            transformers.AutoTokenizer.from_pretrained(
                self._config["model_metadata"]["tokenizer_repository"]
            ),
        )

        self._draft_model = spec_dec.ModelWrapper(
            client,
            DRAFT_MODEL_KEY,
            transformers.AutoTokenizer.from_pretrained(
                self._config["model_metadata"]["speculative_decoding"][
                    "draft_tokenizer_repository"
                ]
            ),
        )

    async def predict(self, model_input):
        # stream_uuid = str(os.getpid()) + str(next(self._request_id_counter))
        request = helpers.GenerationRequest.parse_obj(model_input)

        max_num_draft_tokens = (
            self._config["model_metadata"]["speculative_decoding"][
                "max_num_draft_tokens"
            ],
        )
        return asyncio.ensure_future(
            spec_dec.run_speculative_inference(
                self._target_model,
                self._draft_model,
                request,
                max_num_draft_tokens=max_num_draft_tokens,
                verbose=False,
            )
        )

        # TODO: make async/stream.

    # def __del__(self) -> None:
    #     self._triton_server.shutdown()
    #     del self
