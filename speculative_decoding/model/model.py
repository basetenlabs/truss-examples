"""
This truss model downloads a target and draft model engine, gets their corresponding
tokenizers from HF and starts a tritonserver.

TODO:
* Optional: Supporting openai compatibility.
* Optional: introduce a pydantic schema for specifying the speculative decoding config
  in the truss config yaml.
* Optional: generate the complete triton repo and config dynamically from the truss
 config.
"""

import asyncio
import itertools
import os
from typing import Any, AsyncGenerator, Iterator, Optional

import helpers  # From packages.
import huggingface_hub
import transformers
import tritonclient.grpc.aio as triton_grpc

import speculative_decoding  # From packages.

TRITON_DIR = os.path.join("/", "packages", "triton_model_repo")
TARGET_MODEL_KEY = "target_model"
DRAFT_MODEL_KEY = "draft_model"


class Model:
    _data_dir: Any
    _secrets: Any  # SecretsResolver
    _request_id_counter: Iterator[int]
    _triton_server: Optional[helpers.TritonServer]
    _target_model: Optional[speculative_decoding.ModelWrapper]
    _draft_model: Optional[speculative_decoding.ModelWrapper]

    def __init__(self, **kwargs):
        self._data_dir = kwargs["data_dir"]
        self._config = kwargs["config"]
        self._secrets = kwargs["secrets"]
        self._request_id_counter = itertools.count(start=1)
        self._triton_server = None
        self._target_model = None
        self._draft_model = None

        if "openai-compatible" in self._config["model_metadata"]["tags"]:
            raise NotImplementedError("openai-compatible")

    def load(self):
        if not helpers.is_triton_server_alive():
            tensor_parallel_count = self._config["model_metadata"].get(
                "tensor_parallelism", 1
            )
            pipeline_parallel_count = self._config["model_metadata"].get(
                "pipeline_parallelism", 1
            )

            env = {}
            try:  # Secrets does not have __contains__.
                hf_access_token = self._secrets["hf_access_token"]
                env["HUGGING_FACE_HUB_TOKEN"] = hf_access_token
            except (
                Exception
            ):  # `SecretNotFound` from truss template cannot be referenced.
                hf_access_token = None

            # Target model.
            huggingface_hub.snapshot_download(
                self._config["model_metadata"]["engine_repository"],
                local_dir=os.path.join(TRITON_DIR, TARGET_MODEL_KEY, "1"),
                local_dir_use_symlinks=False,
                max_workers=4,
                use_auth_token=hf_access_token,
            )
            # Draft model.
            huggingface_hub.snapshot_download(
                self._config["model_metadata"]["speculative_decoding"][
                    "draft_engine_repository"
                ],
                local_dir=os.path.join(TRITON_DIR, DRAFT_MODEL_KEY, "1"),
                local_dir_use_symlinks=False,
                max_workers=4,
                use_auth_token=hf_access_token,
            )

            self._triton_server = helpers.TritonServer(
                TRITON_DIR,
                parallel_count=tensor_parallel_count * pipeline_parallel_count,
            )
            self._triton_server.load_server_and_model(env)

        # When Truss server loads model, it does not have an event loop - need to defer.
        def make_async_models():
            client = triton_grpc.InferenceServerClient("localhost:8001")
            self._target_model = speculative_decoding.ModelWrapper(
                client,
                TARGET_MODEL_KEY,
                transformers.AutoTokenizer.from_pretrained(
                    self._config["model_metadata"]["tokenizer_repository"]
                ),
            )

            self._draft_model = speculative_decoding.ModelWrapper(
                client,
                DRAFT_MODEL_KEY,
                transformers.AutoTokenizer.from_pretrained(
                    self._config["model_metadata"]["speculative_decoding"][
                        "draft_tokenizer_repository"
                    ]
                ),
            )

        self._make_async_models = make_async_models

    async def predict(self, model_input) -> str | AsyncGenerator[str, None]:
        if self._draft_model is None:
            self._make_async_models()

        request_id = str(os.getpid()) + str(next(self._request_id_counter))
        request = helpers.GenerationRequest.parse_obj(model_input)
        model_max_num_draft_tokens: int = self._config["model_metadata"][
            "speculative_decoding"
        ]["max_num_draft_tokens"]

        if request.num_draft_tokens is not None:
            max_num_draft_tokens = min(
                model_max_num_draft_tokens, request.num_draft_tokens
            )
        else:
            max_num_draft_tokens = model_max_num_draft_tokens

        streaming: bool = model_input.get("streaming", True)

        maybe_queue: asyncio.Queue[str | None] | None = (
            asyncio.Queue() if streaming else None
        )
        if max_num_draft_tokens > 0:
            infer_co = speculative_decoding.run_speculative_inference(
                self._target_model,
                self._draft_model,
                request,
                max_num_draft_tokens=max_num_draft_tokens,
                request_id=request_id,
                result_queue=maybe_queue,
                verbose=False,
            )
        else:
            infer_co = speculative_decoding.run_conventional_inference(
                self._target_model,
                request,
                request_id=request_id,
                result_queue=maybe_queue,
            )

        # `ensure_future` makes sure the loop immediately runs until completion and
        # fills up the result queue as fast as possible (only limited by the inference
        # requests latency) and doesn't wait for the consumption of the results.
        inference_gen: asyncio.Task[
            speculative_decoding.SpeculationState
        ] = asyncio.ensure_future(infer_co)

        if maybe_queue is not None:

            async def generate_result() -> AsyncGenerator[str, None]:
                while True:
                    text_or_sentinel = await maybe_queue.get()
                    if text_or_sentinel == speculative_decoding.QUEUE_SENTINEL:
                        break
                    yield text_or_sentinel

            return generate_result()

        else:

            async def get_result() -> str:
                return (await inference_gen).get_verified_text()[len(request.prompt) :]

            return await get_result()

    def shutdown(self) -> None:
        if self._triton_server:
            self._triton_server.shutdown()
