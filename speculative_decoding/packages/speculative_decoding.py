"""Helpers and control flow for draft-target-model speculatice decoding.

Notes:
* Draft and target model can use different tokenizers, clear text is used as exchange
  between them. In fact this implementaiton does not optimize for same encoding, it
  always goes via clear text encoding/decoding (which could be skipped and simplified
  for same tokenizers). There are checks in place for the situation that a string gets
  differently tokenized when appending more charaters.
* There are some limitations of non-zero temperature sampling that could lead to
  deviating results from direct generation from the target model. This is due to using
  "hard" draft tokens and not draft probabilities and might be improved in the future.


TODOs:
* Add unittests.
* Optional: remove debug/timing branches.
* Optional: refactor such that inference results only contain generation and no context.
"""

import asyncio
import functools
import os
import time

import colorama
import helpers  # From packages.
import numpy as np
import transformers
import tritonclient.grpc.aio as triton_grpc
from numpy.typing import NDArray


class SpeculationState:
    """Keeps track of verified and draft text and provides safte getters/setters.

    This version of speculative decoding supports different tokenizers for draft and
    target model, therefore it is neccessary to convert to clear text and back between
    making inference reuqests to the respective models.

    The draft model is treated like a black-box with a text prompt as input and
    generated text as output.

    For making inference with the target model we need to separate the verfied ouput
    and draft output on a *tokenized representation*. Therefore this helper class
    provides a getter method to aceess these tokens (`get_draft_ids`).

    In some cases using different tokenizers can lead to changes of previously
    verified tokens which requires careful handling (see docstring of `update_draft`).

    This class also contains debug functionality to track the draft token acceptance
    rate and print colored visualizations of the draft, accepted and corrected outputs.
    """

    _target_tokenizer: transformers.AutoTokenizer
    _verified_text: str
    _verified_ids: NDArray[np.int32]
    _draft_text: str | None
    _draft_ids: NDArray[np.int32] | None
    _num_updates: int
    _sum_accepted_tokens: int
    _debugging: bool

    def __init__(
        self,
        prompt: str,
        target_tokenizer: transformers.AutoTokenizer,
        debugging: bool = True,
    ):
        self._target_tokenizer = target_tokenizer

        self._verified_text = prompt
        self._verified_ids = np.squeeze(
            target_tokenizer.encode(prompt, return_tensors="np")
        )
        self._draft_text = None
        self._draft_ids = None

        self._num_updates = 0
        self._sum_accepted_tokens = 0

        self._debugging = debugging

    def get_verified_text(self) -> str:
        return self._verified_text

    @property
    def num_tokens(self) -> int:
        return len(self._verified_ids)

    @property
    def text_len(self) -> int:
        return len(self._verified_text)

    def update_draft(self, combined_draft: str) -> None:
        """
        Integrates the draft model prediction into the state.

        `combined_draft` = `verified_text` + `draft_text`

        The main purpose is to check whether adding the draft changes previously
        confirmed tokens when using the target models tokenizer. This can happen if
        by adding chars, a "merged" token would represent both old and new chars
        more compactly. In that case previously confirmed chars get mixed together
        with draft chars which is problematic.

        There are two options to deal with this case:
        * "Demote" the changed token from confirmed to draft status. The problem
          though is, that this would overwrite a previous decision of the target
          model about what exact token should be sampled here.
        * Discard the draft all together (even though it might end up being good)
          and let the target model generate a new token after the original token.

        In this implementation the second option, discarding the draft, is chosen.
        """
        with helpers.timeit(f"Tokenize for Target model", skip=True):
            new_ids = np.squeeze(
                self._target_tokenizer.encode(combined_draft, return_tensors="np")
            )
        verified_ids_reencoded, draft_ids = np.split(new_ids, [self.num_tokens])

        if not np.alltrue(self._verified_ids == verified_ids_reencoded):
            self._draft_ids = None
            self._draft_text = None
            if self._debugging:
                print(
                    f"Draft  : '{combined_draft}'\nDraft causes token flip "
                    "of confirmed text! Must sample target model directly to recover."
                )
            return

        self._draft_ids = draft_ids
        self._draft_text = combined_draft[self.text_len :]

        if self._debugging:
            debug_str = (
                f"Draft  : '{self._verified_text}"
                f"{colorama.Fore.BLUE + colorama.Style.BRIGHT}{self._draft_text}'"
            ).replace(
                "\n", "\\n"
            )  # Colorama does not mix well with newline.
            print(debug_str)

    def get_verified_ids(self) -> NDArray[np.int32]:
        """Returns verified IDs."""
        return self._verified_ids

    def get_draft_ids(self) -> NDArray[np.int32] | None:
        """Returns draft IDs or `None` if there is no draft.

        No draft tokens might happen when re-tonkenizing `verified_text` + `draft_text`
        would lead to flipped tokens in the previously verified text. In that case
        a direct sample from the target model must be taken (see docstring of
        `update_draft`) .
        """
        return self._draft_ids

    def update_verifed_text(
        self, verified_text: str, verified_ids: NDArray[np.int32]
    ) -> None:
        """Note: both inputs must be the full sequence, not just newly generated."""
        if self._debugging:
            draft_text = self._draft_text or ""
            added_text = verified_text[self.text_len :]
            accepted_text = os.path.commonprefix([added_text, draft_text])
            if len(accepted_text) < len(draft_text):
                disagreed_text = added_text[len(accepted_text) : len(draft_text)]
            else:
                disagreed_text = ""
            new_text = added_text[len(draft_text) :]

            if self._draft_ids is None:
                accepted_tokens: list[int] = []
            else:
                accepted_tokens = os.path.commonprefix(
                    [list(self._draft_ids), list(verified_ids[self.num_tokens :])]
                )

            self._num_updates += 1
            self._sum_accepted_tokens += len(accepted_tokens)

            style = colorama.Fore.YELLOW + colorama.Style.BRIGHT
            debug_str = (
                f"Verfied: '{self._verified_text}"
                f"{colorama.Fore.GREEN + colorama.Back.BLUE}{accepted_text}"
                f"{colorama.Back.RED + style}{disagreed_text}"
                f"{colorama.Style.RESET_ALL + style}{new_text}"
                f"{colorama.Style.RESET_ALL}' -> Accepted `{len(accepted_tokens)}` "
                f"tokens <=> {len(accepted_text)}` chars."
            ).replace(
                "\n", "\\n"
            )  # Colorama does not mix well with newline.
            print(debug_str)

        self._verified_text = verified_text
        self._verified_ids = verified_ids
        self._draft_ids = None
        self._draft_text = None

    def get_aveage_num_accepted_draft_tokens(self) -> float:
        if not self._debugging and self._num_updates == 0:
            raise ValueError(
                "You must set `debugging` to `True` and run at "
                "least one update to calculate the rate."
            )

        return self._sum_accepted_tokens / self._num_updates


class ModelWrapper:
    _client: triton_grpc.InferenceServerClient
    _model_name: str
    _tokenizer: transformers.AutoTokenizer

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
        self, word_list: tuple[str, ...] | None
    ) -> NDArray[np.int32] | None:
        if word_list:
            # Implementation is batched, so add and remove batch dimension.
            return helpers.to_word_list_format(
                [word_list], self._tokenizer, add_special_tokens=False
            )
        return None

    async def generate(
        self,
        input_text: str,
        max_num_gen_tokens: int,
        request_id: str,
        sampling_config: helpers.SamplingConfig | None = None,
        bad_word_list: tuple[str, ...] | None = None,
        stop_words_list: tuple[str, ...] | None = None,
    ) -> tuple[str, NDArray[np.int32]]:
        """Generates and appends text/tokens with classic loop without draft tokens."""
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
        return output_text, output_ids

    async def verify_and_generate(
        self,
        confirmed_ids: NDArray[np.int32],
        draft_ids: NDArray[np.int32],
        request_id: str,
        sampling_config: helpers.SamplingConfig | None = None,
        bad_word_list: tuple[str, ...] | None = None,
        stop_words_list: tuple[str, ...] | None = None,
    ) -> tuple[str, NDArray[np.int32]]:
        """Accepts/rejects draft tokens and generates `1` new token.

        If all draft tokens are accepted, the new token extends the the sequence.
        If some draft tokens are rejected, the new token is generated to replace
        the first rejected draft token.
        """
        num_gen_tokens = len(draft_ids) + 1 if draft_ids is not None else 1
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
            result = await self._client.infer(
                self._model_name, inputs, request_id=request_id
            )
            output_ids = helpers.extract_trtllm_outputs(result)

        with helpers.timeit(f"Generate({self._model_name}) - detokenize", skip=True):
            output_text = self._tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text, output_ids


QUEUE_SENTINEL = None


async def run_speculative_inference(
    target_model: ModelWrapper,
    draft_model: ModelWrapper,
    request: helpers.GenerationRequest,
    max_num_draft_tokens: int,
    request_id: str,
    result_queue: asyncio.Queue[str | None] | None = None,
    verbose: bool = False,
    iteration_delay: float = 0.0,
) -> SpeculationState:
    """Runs the speculative decoding control flow loop betweeen draft and target model

    * Model invocations are async inference requests to the triton server.
    * For streaming, intermediate results are published in an optional `result_queue`,
       note that currently each result includes the context prefix, not just generated
       text.
    * For the loop to advance, it is necessary to await the results of all inferernce
      requests. But if this coroutine function is scheduled as a task, e.g.
      by `asyncio.ensure_future`, one python server can run multiple inferences
      concurrently. The inference requests coming from those can be collated to
      batches by the tritonserver if batching is configured, which can increase total
      throughput.
    """
    state = SpeculationState(request.prompt, target_model._tokenizer, debugging=verbose)

    # Use sampling defaults (i.e. greedy sampling) for draft model.
    draft_sampling_conifg = helpers.SamplingConfig()

    max_total_tokens = state.num_tokens + request.max_num_generated_tokens
    num_chars_generated = len(request.prompt)

    bad_words = tuple(request.bad_word_list) if request.bad_word_list else None
    stop_words = tuple(request.stop_words_list) if request.stop_words_list else None

    while True:
        num_draft_tokens = min(
            max_num_draft_tokens,
            max_total_tokens - state.num_tokens,
        )
        combined_draft, _ = await draft_model.generate(
            state.get_verified_text(),
            num_draft_tokens,
            request_id,
            draft_sampling_conifg,
            bad_words,
            stop_words,
        )
        state.update_draft(combined_draft)
        if iteration_delay:
            time.sleep(iteration_delay)

        previously_verified_ids = state.get_verified_ids()
        draft_ids = state.get_draft_ids()
        if draft_ids is not None:
            if len(draft_ids) > max_num_draft_tokens:
                draft_ids = draft_ids[:max_num_draft_tokens]

            verified_text, verfied_ids = await target_model.verify_and_generate(
                previously_verified_ids,
                draft_ids,
                request_id,
                request.sampling_config,
                bad_words,
                stop_words,
            )
        else:
            # There are no drafts to verify if appending the draft leads to a
            # tokenization inconsistent with the previously confirmed text.
            # See docstring of `SpeculationState.update_draft`.
            verified_text, verfied_ids = await target_model.generate(
                state.get_verified_text(),
                1,
                request_id,
                request.sampling_config,
                bad_words,
                stop_words,
            )

        # Check if EOS or stop words were encountered: in that case the target model
        # does not generate any new tokens and the output is identical to the input.
        if np.array_equal(previously_verified_ids, verfied_ids):
            break

        state.update_verifed_text(verified_text, verfied_ids)
        if result_queue is not None:
            complete_text = state.get_verified_text()
            incremental_text = complete_text[num_chars_generated:]
            result_queue.put_nowait(incremental_text)
            num_chars_generated = state.text_len

        if len(verfied_ids) >= max_total_tokens:
            break

        if iteration_delay:
            time.sleep(iteration_delay)

    if result_queue is not None:
        result_queue.put_nowait(QUEUE_SENTINEL)
    return state


async def run_conventional_inference(
    target_model: ModelWrapper,
    request: helpers.GenerationRequest,
    request_id: str,
    result_queue: asyncio.Queue[str | None] | None = None,
) -> SpeculationState:
    """Fallback implementation of conventional generation."""
    state = SpeculationState(request.prompt, target_model._tokenizer, debugging=False)
    bad_words = tuple(request.bad_word_list) if request.bad_word_list else None
    stop_words = tuple(request.stop_words_list) if request.stop_words_list else None
    verified_text, verfied_ids = await target_model.generate(
        request.prompt,
        request.max_num_generated_tokens,
        request_id,
        request.sampling_config,
        bad_words,
        stop_words,
    )

    state.update_verifed_text(verified_text, verfied_ids)
    if result_queue is not None:
        result_queue.put_nowait(state.get_verified_text()[len(request.prompt) :])
        result_queue.put_nowait(QUEUE_SENTINEL)
    return state
