import functools
import os
from typing import AsyncGenerator, Sequence

import colorama
import numpy as np
import transformers
import tritonclient.grpc.aio as triton_grpc

from . import helpers


class SpeculationState:
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
                f"Draft  : '{self._current_text}"
                f"{colorama.Fore.BLUE + colorama.Style.BRIGHT}{self._draft_text}'"
            )

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
                f"Verfied: '{self._current_text}"
                f"{colorama.Fore.GREEN + colorama.Back.BLUE}{accepted_text}"
                f"{colorama.Back.RED + style}{disagreed_text}"
                f"{colorama.Style.RESET_ALL + style}{new_text}"
                f"{colorama.Style.RESET_ALL}' -> Accepted `{len(accepted_tokens)}` "
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


class ModelWrapper:
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

    async def generate(
        self,
        input_text: str,
        max_num_gen_tokens: str,
        request_id: str,
        sampling_config: helpers.SamplingConfig | None = None,
        bad_word_list: Sequence[str] | None = None,
        stop_words_list: Sequence[str] | None = None,
    ) -> str:
        with helpers.timeit(f"Generate({self._model_name}) - tokenize", skip=True):
            input_ids = np.squeeze(
                self._tokenizer.encode(input_text, return_tensors="np")
            )
        with helpers.timeit(f"Generate({self._model_name}) - input prep", skip=True):
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
            result = await self._client.infer(
                self._model_name, inputs, request_id=request_id
            )
            output_ids = helpers.extract_trtllm_outputs(result)
        with helpers.timeit(f"Generate({self._model_name}) - detokenize", skip=True):
            output_text = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text

    async def verify_and_generate(
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
        with helpers.timeit(f"Generate({self._model_name}) - input prep", skip=True):
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
            result = await self._client.infer(
                self._model_name, inputs, request_id=request_id
            )
            output_ids = helpers.extract_trtllm_outputs(result)

        # with helpers.timeit(f"Generate({self._model_name}) - detokenize"):
        output_text = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text, output_ids


async def run_speculative_inference(
    target_model: ModelWrapper,
    draft_model: ModelWrapper,
    request: helpers.GenerationRequest,
    max_num_draft_tokens,
    verbose,
) -> AsyncGenerator[str, None]:
    # ) -> SpeculationState:

    state = SpeculationState(request.prompt, target_model._tokenizer, debugging=verbose)
    while True:
        num_draft_tokens = min(
            max_num_draft_tokens,
            request.max_num_generated_tokens - state.num_tokens,
        )
        state.update_draft(
            await draft_model.generate(
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

        verified_text, verfied_ids = await target_model.verify_and_generate(
            confirmed_ids,
            draft_ids,
            request.request_id,
            request.sampling_config,
            request.bad_word_list,
            request.stop_words_list,
        )
        yield verified_text

        state.update_verifed_text(verified_text, verfied_ids)

        if len(verfied_ids) >= request.max_num_generated_tokens:
            break

    # return state
