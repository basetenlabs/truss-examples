import functools
import os
import time
from typing import Sequence

import colorama
import helpers  # TODO: this dep is relative and not portable
import numpy as np
import pandas as pd
import transformers
import tritonclient.grpc as triton_grpc

# from tensorrt_llm import runtime
from tqdm.contrib import itertools


class _SpeculationState:
    """This class takes the perspective of the target model in terms of tokenization."""

    _target_tokenizer: transformers.AutoTokenizer
    _current_text: str
    _current_ids: np.ndarray[int]
    _draft_text: str | None
    _draft_ids: np.ndarray[int] | None
    _debugging: bool

    def __init__(
        self,
        prompt: str,
        target_tokenizer: transformers.AutoTokenizer,
        debugging: bool = True,
    ):
        self._target_tokenizer = target_tokenizer

        self._current_text = prompt  # This text is confirmed by target (or prompt).
        self._current_ids = np.squeeze(
            target_tokenizer.encode(prompt, return_tensors="np")
        )
        self._draft_text = None
        self._draft_ids = None

        self._num_updates = 0
        self._sum_accepted_tokens = 0

        self._debugging = debugging

    def get_current_text(self) -> str:
        return self._current_text

    @property
    def num_tokens(self) -> int:
        return len(self._current_ids)

    @property
    def text_len(self) -> int:
        return len(self._current_text)

    def update_draft(self, combined_draft: str) -> None:
        # `combined_draft` == `current_text` + `draft_text`
        # TODO: explore if existing tokenization from previous iteration can
        # be reused safely. This is related to whether below check ever fails.
        with helpers.timeit(f"Tokenize for Target model"):
            new_ids = np.squeeze(
                self._target_tokenizer.encode(combined_draft, return_tensors="np")
            )
        current_ids_reencoded, draft_ids = np.split(new_ids, [self.num_tokens])

        # Verify that adding anything to `current_text` did not change tokenization.
        # This could happen if by adding chars, a "merged" token would
        # represent both old and new chars more compactly.
        # In that case we would need to track back to the last point where tokens
        # still agree and continue from there.
        if not np.alltrue(self._current_ids == current_ids_reencoded):
            raise NotImplementedError(
                f"Dang!!! {self._current_ids} vs.\n {current_ids_reencoded}"
            )

        self._draft_ids = draft_ids
        self._draft_text = combined_draft[self.text_len :]

        if self._debugging:
            print(
                f"Draft  : {self._current_text}"
                f"{colorama.Fore.BLUE + colorama.Style.BRIGHT}{self._draft_text}"
            )
            pass

    def get_verification_inputs(self) -> tuple[np.ndarray[int], np.ndarray[int]] | None:
        if self._draft_ids is None:
            # Nothing to verify.
            return None
        return self._current_ids, self._draft_ids

    def update_verifed_text(
        self, verified_text: str, verified_ids: np.ndarray[int]
    ) -> None:
        # Both inputs include the whole context.
        if self._debugging:
            added_text = verified_text[self.text_len :]
            accepted_text = os.path.commonprefix([added_text, self._draft_text])
            if len(accepted_text) < len(self._draft_text):
                disagreed_text = added_text[len(accepted_text) : len(self._draft_text)]
            else:
                disagreed_text = ""
            new_text = added_text[len(self._draft_text) :]

            accepted_tokens = os.path.commonprefix(
                [list(self._draft_ids), list(verified_ids[self.num_tokens :])]
            )

            self._num_updates += 1
            self._sum_accepted_tokens += len(accepted_tokens)

            style = colorama.Fore.YELLOW + colorama.Style.BRIGHT
            print(
                f"Verfied: {self._current_text}"
                f"{colorama.Fore.GREEN + colorama.Back.BLUE}{accepted_text}"
                f"{colorama.Back.RED + style}{disagreed_text}"
                f"{colorama.Style.RESET_ALL + style}{new_text}"
                f"{colorama.Style.RESET_ALL} -> Accepted `{len(accepted_tokens)}` "
                f"tokens <=> {len(accepted_text)}` chars."
            )

        self._current_text = verified_text
        self._current_ids = verified_ids
        self._draft_ids = None
        self._draft_text = None

    def get_aveage_num_accepted_draft_tokens(self) -> float:
        if not self._debugging and self._num_updates == 0:
            raise ValueError(
                "You must turn on `debugging` and run at "
                "least one update to calculate the rate"
            )

        return self._sum_accepted_tokens / self._num_updates


class _ModelWrapper:
    def __init__(
        self,
        client: triton_grpc.InferenceServerClient,
        model_name: str,
        tokenizer: transformers.AutoTokenizer,
    ):
        self._client = client
        self._model_name = model_name
        self._tokenizer = tokenizer

    @functools.lru_cache(maxsize=128)
    def _tokenize_word_list(
        self, word_list: Sequence[str] | None
    ) -> np.ndarray[int] | None:
        if word_list is None:
            return None
        # return runtime.to_word_list_format(
        #     word_list, self._tokenizer, add_special_tokens=False
        # )

    def generate(
        self,
        input_text: str,
        max_num_gen_tokens: str,
        request_id: str,
        sampling_config: helpers.SamplingConfig | None = None,
        bad_word_list: Sequence[str] | None = None,
        stop_words_list: Sequence[str] | None = None,
    ) -> str:
        with helpers.timeit(f"Generate({self._model_name}) - tokenize"):
            input_ids = np.squeeze(
                self._tokenizer.encode(input_text, return_tensors="np")
            )
        with helpers.timeit(f"Generate({self._model_name}) - input prep"):
            inputs = helpers.make_trtllm_inputs(
                input_ids,
                max_num_gen_tokens,
                sampling_config,
                None,  # draft_tokens
                self._tokenizer.eos_token_id,
                self._tokenizer.pad_token_id,
                self._tokenize_word_list(bad_word_list),
                self._tokenize_word_list(stop_words_list),
            )
        with helpers.timeit(f"Generate({self._model_name})"):
            result = self._client.infer(self._model_name, inputs, request_id=request_id)
            output_ids = helpers.extract_trtllm_outputs(result)
        with helpers.timeit(f"Generate({self._model_name}) - detokenize"):
            output_text = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text

    def verify_and_generate(
        self,
        confirmed_ids: np.ndarray[int],
        draft_ids: np.ndarray[int],
        request_id: str,
        sampling_config: helpers.SamplingConfig | None = None,
        bad_word_list: Sequence[str] | None = None,
        stop_words_list: Sequence[str] | None = None,
    ):
        num_gen_tokens = len(draft_ids) + 1 if draft_ids is not None else 1
        # TODO: check len of draft IDs, warn if throw away.
        with helpers.timeit(f"Generate({self._model_name}) - input prep"):
            inputs = helpers.make_trtllm_inputs(
                confirmed_ids,
                num_gen_tokens,
                sampling_config,
                draft_ids,
                self._tokenizer.eos_token_id,
                self._tokenizer.pad_token_id,
                self._tokenize_word_list(bad_word_list),
                self._tokenize_word_list(stop_words_list),
            )
        with helpers.timeit(f"Verify+Generate({self._model_name})"):
            # for inp in inputs:
            #     print(inp.name(), print(inp._input), inp._raw_content)

            result = self._client.infer(self._model_name, inputs, request_id=request_id)
            output_ids = helpers.extract_trtllm_outputs(result)

        with helpers.timeit(f"Generate({self._model_name}) - detokenize"):
            output_text = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text, output_ids


def run_speculative_inference(
    target_model: _ModelWrapper,
    draft_model: _ModelWrapper,
    request: helpers.GenerationRequest,
    max_num_draft_tokens,
    verbose,
):
    # Warmup models with unrelated string.
    target_model.generate("What is a computer?", 4, "111", request.sampling_config)
    draft_model.generate("What is a computer?", 4, "111", request.sampling_config)

    with helpers.timeit("A - speculative_gen"):
        state = _SpeculationState(
            request.prompt, target_model._tokenizer, debugging=verbose
        )
        while True:
            num_draft_tokens = min(
                max_num_draft_tokens,
                request.max_num_generated_tokens - state.num_tokens,
            )
            state.update_draft(
                draft_model.generate(
                    state.get_current_text(),
                    num_draft_tokens,
                    request.request_id,
                    request.sampling_config,
                    request.bad_word_list,
                    request.stop_words_list,
                )
            )
            confirmed_ids, draft_ids = state.get_verification_inputs()
            if len(draft_ids) > max_num_draft_tokens:
                # print(draft_ids, state._draft_text)
                draft_ids = draft_ids[:max_num_draft_tokens]

            verified_text, verfied_ids = target_model.verify_and_generate(
                confirmed_ids,
                draft_ids,
                request.request_id,
                request.sampling_config,
                request.bad_word_list,
                request.stop_words_list,
            )
            state.update_verifed_text(verified_text, verfied_ids)
            if len(verfied_ids) >= request.max_num_generated_tokens:
                break

    with helpers.timeit("B - direct_gen"):
        print(
            target_model.generate(
                request.prompt,
                request.max_num_generated_tokens,
                request.request_id,
                request.sampling_config,
                request.bad_word_list,
                request.stop_words_list,
            )
        )

    helpers.show_timings()
    print(f"Final text:\n{verified_text}")
    print(
        f"Average num of accepted draft tokens: "
        f"{state.get_aveage_num_accepted_draft_tokens():.2f}"
    )
    return verified_text


# def profile(
#     client,
#     model_name,
#     prompt_lens: Sequence[int],
#     generate_lens: Sequence[int],
#     n_samples: int = 20,
# ):
#     measurements = []
#     for prompt_len, generate_len, i in itertools.product(
#         prompt_lens, generate_lens, range(n_samples)
#     ):
#         prompt_ids = np.random.randint(0, 32000, prompt_len)
#         inputs = helpers.make_trtllm_inputs(prompt_ids, generate_len)

#         t0 = time.time()
#         result = client.infer(model_name, inputs, request_id="123")
#         elapsed = time.time() - t0
#         row = {
#             "model_name": model_name,
#             "prompt_len": prompt_len,
#             "generate_len": generate_len,
#             "time": elapsed,
#         }
#         # print(row)
#         measurements.append(row)

#     df = pd.DataFrame(measurements)
#     print(df.to_json())
#     return df


# def profile_verification(
#     client,
#     model_name,
#     prompt_lens: Sequence[int],
#     draft_lens: Sequence[int],
#     n_samples: int = 20,
# ):
#     measurements = []
#     for prompt_len, draft_len, i in itertools.product(
#         prompt_lens, draft_lens, range(n_samples)
#     ):
#         prompt_ids = np.random.randint(0, 32000, prompt_len)
#         inputs = helpers.make_trtllm_inputs(
#             prompt_ids[:-draft_len], 1, prompt_ids[-draft_len:]
#         )

#         t0 = time.time()
#         result = client.infer(model_name, inputs, request_id="123")
#         elapsed = time.time() - t0
#         row = {
#             "model_name": model_name,
#             "prompt_len": prompt_len,
#             "draft_len": draft_len,
#             "time": elapsed,
#         }
#         # print(row)
#         measurements.append(row)

#     df = pd.DataFrame(measurements)
#     print(df.to_json())
#     return df


def run_dummy_request(client):
    inputs = []
    helpers.fill_inputs("input", [[1, 4]], np.int32, inputs)
    response = client.infer("dummy_model", inputs, request_id="blah")


if __name__ == "__main__":
    import huggingface_hub

    colorama.init(autoreset=True)

    DOWNLOAD_ENGINES = False
    TRITON_DIR = os.path.join("/packages", "triton_model_repo")

    DRAFT_MODEL_ENGINE_HF = "baseten/specdec-draft-gpt2"
    DRAFT_MODEL_TOKENIZER_HF = "gpt2"
    DRAFT_MODEL_KEY = "draft_model"
    TARGET_MODEL_ENGINE_HF = "baseten/specdec-target-mistral-7B"
    TARGET_MODEL_TOKENIZER_HF = "mistralai/Mistral-7B-v0.1"
    TARGET_MODEL_KEY = "target_model"

    if DOWNLOAD_ENGINES:
        huggingface_hub.snapshot_download(
            DRAFT_MODEL_ENGINE_HF,
            local_dir=os.path.join(TRITON_DIR, DRAFT_MODEL_KEY, "1"),
            local_dir_use_symlinks=False,
            max_workers=4,
        )
        huggingface_hub.snapshot_download(
            TARGET_MODEL_ENGINE_HF,
            local_dir=os.path.join(TRITON_DIR, TARGET_MODEL_KEY, "1"),
            local_dir_use_symlinks=False,
            max_workers=4,
        )

    # Start triton server *outside* project's poetry env:
    # tritonserver --model-repository \
    # /root/workbench/truss-examples/speculative_decoding/triton_model_repo \
    # --grpc-port 8001 \
    # --http-port 8003

    client_ = triton_grpc.InferenceServerClient("0.0.0.0:8001")

    target_model_ = _ModelWrapper(
        client_,
        TARGET_MODEL_KEY,
        transformers.AutoTokenizer.from_pretrained(TARGET_MODEL_TOKENIZER_HF),
    )

    draft_model_ = _ModelWrapper(
        client_,
        DRAFT_MODEL_KEY,
        transformers.AutoTokenizer.from_pretrained(DRAFT_MODEL_TOKENIZER_HF),
    )

    request_ = helpers.GenerationRequest(
        # prompt="Once upon a time there was",
        prompt="Once upon",
        max_num_generated_tokens=60,
        request_id="123",
    )
    request_.sampling_config.random_seed = 123412
    request_.sampling_config.temperature = 3.0

    output_text = run_speculative_inference(
        target_model_,
        draft_model_,
        request_,
        max_num_draft_tokens=4,
        verbose=True,
    )

    # target_gen_profile = profile(
    #     client_,
    #     model_name="target_model",
    #     prompt_lens=[1, 20, 50, 200, 400],
    #     generate_lens=[1, 2, 3, 4, 10, 20],
    #     n_samples=30,
    # )

    # target_gen_profile = profile(
    #     client_,
    #     model_name="draft_model",
    #     prompt_lens=[1, 20, 50, 200, 400],
    #     generate_lens=[1, 2, 3, 4, 10, 20],
    #     n_samples=30,
    # )

    # target_gen_profile = profile_verification(
    #     client_,
    #     model_name="target_model",
    #     prompt_lens=[20, 50, 200, 400],
    #     draft_lens=[1, 2, 3, 4],
    #     n_samples=30,
    # )
