import time
from typing import Sequence

import helpers  # TODO: this dep is relative and not portable
import numpy as np
import pandas as pd
import transformers
import tritonclient.grpc as triton_grpc
from tqdm.contrib import itertools


def _make_trtllm_inputs(
    input_ids: Sequence[int],
    max_num_generated_tokens: int,
    sampling_config: helpers.SamplingConfig | None,
    draft_tokens: Sequence[int] | None = None,
    end_id: int | None = None,
    pad_id: int | None = None,
    bad_words_ids: np.ndarray | None = None,
    stop_words_ids: np.ndarray | None = None,
):
    input_length = len(input_ids)
    inputs = []
    # Add batch dimension.
    helpers.fill_inputs("input_ids", input_ids, np.int32, inputs)
    helpers.fill_inputs("input_lengths", input_length, np.int32, inputs)
    helpers.fill_inputs(
        "request_output_len", max_num_generated_tokens, np.uint32, inputs
    )
    # All below are optional inputs.
    helpers.fill_inputs("draft_input_ids", draft_tokens, np.int32, inputs)
    # Generation.
    helpers.fill_inputs("end_id", end_id, np.uint32, inputs)
    helpers.fill_inputs("pad_id", pad_id, np.uint32, inputs)
    helpers.fill_inputs("bad_words_list", bad_words_ids, np.int32, inputs)
    helpers.fill_inputs("stop_words_list", stop_words_ids, np.int32, inputs)
    if sampling_config:
        helpers.fill_inputs("beam_width", sampling_config.beam_width, np.uint32, inputs)
        helpers.fill_inputs(
            "temperature", sampling_config.temperature, np.float32, inputs
        )
        helpers.fill_inputs(
            "runtime_top_k", sampling_config.runtime_top_p, np.uint32, inputs
        )
        helpers.fill_inputs(
            "runtime_top_p", sampling_config.runtime_top_p, np.float32, inputs
        )
        helpers.fill_inputs(
            "len_penalty", sampling_config.len_penalty, np.float32, inputs
        )
        helpers.fill_inputs(
            "repetition_penalty", sampling_config.repetition_penalty, np.float32, inputs
        )
        helpers.fill_inputs("min_len", sampling_config.min_len, np.float32, inputs)
        helpers.fill_inputs(
            "presence_penalty", sampling_config.presence_penalty, np.float32, inputs
        )
        helpers.fill_inputs(
            "frequency_penalty", sampling_config.frequency_penalty, np.float32, inputs
        )
        helpers.fill_inputs(
            "random_seed", sampling_config.random_seed, np.uint64, inputs
        )
    # helpers.fill_inputs("return_log_probs", True, bool, inputs)
    # The return_X_logits tensor names were only added in
    # https://github.com/NVIDIA/TensorRT-LLM/pull/846.
    # "return_context_logits", "return_generation_logits"
    return inputs


def _extract_trtllm_outputs(result):
    # TODO: Get context_logits, generation_logits and find out why output_log_probs is
    #  always zero.
    # Get batch 0, beam 0 output_ids
    output_ids = np.squeeze(result.as_numpy("output_ids").astype(np.int32), axis=(0, 1))
    sequence_length = int(
        np.squeeze(result.as_numpy("sequence_length").astype(np.int32), axis=(0, 1))
    )
    assert sequence_length == len(output_ids)
    return output_ids


def run_speculative_inference(
    client,
    request: helpers.GenerationRequest,
    max_num_draft_tokens,
    draft_model_name,
    draft_tokenizer,
    target_model_name,
    target_tokenizer,
    verbose,
):
    # TODO: check representation of word lists as tokens.
    # draft_bad_words_ids = draft_tokenizer.encode(bad_words)
    # draft_stop_words_ids = draft_tokenizer.encode(stop_words)

    def call_draft_model(input_text: str) -> str:
        with helpers.timeit("draft_encode"):
            input_ids = draft_tokenizer.encode(input_text)

        num_draft_tokens = min(
            max_num_draft_tokens, request.max_num_generated_tokens - len(input_ids)
        )

        if not num_draft_tokens:
            print("No more drafts to generate")
            return ""

        if verbose:
            clear_text = draft_tokenizer.decode(input_ids)
            assert clear_text == input_text, f"{clear_text} vs. {input_text}"
            print(
                f"Call DRAFT model to generate `{num_draft_tokens}` for:"
                f"\n\t{input_ids}\n\t`{input_text}`"
            )

        with helpers.timeit("draft_input_prep"):
            inputs = _make_trtllm_inputs(
                input_ids,
                num_draft_tokens,
                request.sampling_config,
                None,  # draft_tokens.
                draft_tokenizer.eos_token_id,
                draft_tokenizer.pad_token_id,
            )
        with helpers.timeit("draft_generate"):
            result = client.infer(
                draft_model_name, inputs, request_id=request.request_id
            )

        output_ids = _extract_trtllm_outputs(result)

        with helpers.timeit("draft_decode"):
            clear_text_full = draft_tokenizer.decode(output_ids)

        if verbose:
            draft_ids = output_ids[len(input_ids) :]
            draft_text = clear_text_full[len(input_text) :]
            print(
                f"  Result DRAFT model:\n\t{draft_ids}\n"
                f"\tDraft: `{draft_text}`\n\tAll  : `{clear_text_full}`"
            )
        return clear_text_full

    def call_target_model(input_text: str, confirmed_ids: str) -> tuple[str, list[int]]:
        with helpers.timeit("target_encode"):
            input_ids = target_tokenizer.encode(input_text)
        draft_ids = input_ids[
            len(confirmed_ids) : len(confirmed_ids) + max_num_draft_tokens
        ]
        confirmed_ids_re_encoded = input_ids[: len(confirmed_ids)]
        if not np.allclose(confirmed_ids, confirmed_ids_re_encoded):
            print(confirmed_ids)
            print(confirmed_ids_re_encoded)
            # import pdb
            # pdb.set_trace()

        if verbose:
            clear_text = target_tokenizer.decode(
                confirmed_ids_re_encoded, skip_special_tokens=True
            )
            clear_draft = target_tokenizer.decode(draft_ids, skip_special_tokens=True)
            print(
                f"Call TARGET model with context:"
                f"\n\t{input_ids}\n\t`{clear_text}`\n\tand draft:"
                f"\n\t{draft_ids}\n\t`{clear_draft}`"
            )

        num_gen_tokens = len(draft_ids) + 1 if draft_ids else 1
        with helpers.timeit("target_input_prep"):
            inputs = _make_trtllm_inputs(
                confirmed_ids_re_encoded,
                num_gen_tokens,
                request.sampling_config,
                draft_ids or None,
                target_tokenizer.eos_token_id,
                target_tokenizer.pad_token_id,
            )
        with helpers.timeit("target_verify"):
            result = client.infer(
                target_model_name, inputs, request_id=request.request_id
            )

        output_ids = _extract_trtllm_outputs(result)
        with helpers.timeit("target_decode"):
            clear_text_full = target_tokenizer.decode(
                output_ids, skip_special_tokens=True
            )
        if verbose:
            print(
                f"  Result TARGET model:\n\t{output_ids}\n"
                f"\tResult: `{clear_text_full}`"
            )
        return clear_text_full, output_ids

    def generate_target_model(input_text: str, num_gen_tokens: str) -> str:
        input_ids = target_tokenizer.encode(input_text)
        inputs = _make_trtllm_inputs(
            input_ids,
            num_gen_tokens - len(input_ids),
            request.sampling_config,
            None,  # draft_tokens
            target_tokenizer.eos_token_id,
            target_tokenizer.pad_token_id,
        )
        with helpers.timeit("target_generate"):
            result = client.infer(
                target_model_name, inputs, request_id=request.request_id
            )

        output_ids = _extract_trtllm_outputs(result)
        clear_text_full = target_tokenizer.decode(output_ids, skip_special_tokens=True)
        if verbose:
            print(
                f"  Result TARGET model:\n\t{output_ids}\n"
                f"\tResult: `{clear_text_full}`"
            )
        return clear_text_full

    ####################################################################################

    # Warmup models with unrelated string.
    generate_target_model("What is a computer?", request.max_num_generated_tokens)
    call_draft_model("What is a computer?")

    # Run `direct_gen` in `0.26770877838134766` seconds.
    # with helpers.timeit("direct_gen"):
    #     generate_target_model(prompt, request_output_len)

    # Run `speculative_gen` in `0.1786513328552246` seconds.
    with helpers.timeit("speculative_gen"):
        confirmed_target_ids = target_tokenizer.encode(request.prompt)
        current_text = request.prompt
        while True:
            with helpers.timeit("draft_total"):
                current_text = call_draft_model(current_text)
            with helpers.timeit("target_total"):
                current_text, confirmed_target_ids = call_target_model(
                    current_text, confirmed_target_ids
                )

            print("=======================")
            print(f"Target token len {len(confirmed_target_ids)}")
            if len(confirmed_target_ids) == request.max_num_generated_tokens:
                break

    with helpers.timeit("direct_gen"):
        print(generate_target_model(request.prompt, request.max_num_generated_tokens))

    helpers.show_timings()
    print("Final text:\n", current_text)
    return current_text


def profile(
    client,
    model_name,
    prompt_lens: Sequence[int],
    generate_lens: Sequence[int],
    n_samples: int = 20,
):
    measurements = []
    for prompt_len, generate_len, i in itertools.product(
        prompt_lens, generate_lens, range(n_samples)
    ):
        prompt_ids = np.random.randint(0, 32000, prompt_len)
        inputs = _make_trtllm_inputs(prompt_ids, generate_len, [])

        t0 = time.time()
        result = client.infer(model_name, inputs, request_id="123")
        elapsed = time.time() - t0
        row = {
            "model_name": model_name,
            "prompt_len": prompt_len,
            "generate_len": generate_len,
            "time": elapsed,
        }
        # print(row)
        measurements.append(row)

    df = pd.DataFrame(measurements)
    print(df.to_json())
    return df


def profile_verification(
    client,
    model_name,
    prompt_lens: Sequence[int],
    draft_lens: Sequence[int],
    n_samples: int = 20,
):
    measurements = []
    for prompt_len, draft_len, i in itertools.product(
        prompt_lens, draft_lens, range(n_samples)
    ):
        prompt_ids = np.random.randint(0, 32000, prompt_len)

        inputs = _make_trtllm_inputs(
            prompt_ids[:-draft_len], 1, prompt_ids[-draft_len:]
        )

        t0 = time.time()
        result = client.infer(model_name, inputs, request_id="123")
        elapsed = time.time() - t0
        row = {
            "model_name": model_name,
            "prompt_len": prompt_len,
            "draft_len": draft_len,
            "time": elapsed,
        }
        # print(row)
        measurements.append(row)

    df = pd.DataFrame(measurements)
    print(df.to_json())
    return df


def run_dummy_request(client):
    inputs = []
    helpers.fill_inputs("input", [[1, 4]], np.int32, inputs)
    response = client.infer("dummy_model", inputs, request_id="blah")


if __name__ == "__main__":
    client_ = triton_grpc.InferenceServerClient("0.0.0.0:8001")

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

    draft_tokenizer_ = transformers.AutoTokenizer.from_pretrained("gpt2")
    target_tokenizer_ = transformers.AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1"
    )

    request = helpers.GenerationRequest(
        prompt="Once upon a time there was",
        max_num_generated_tokens=40,
        request_id="123",
    )

    output_text = run_speculative_inference(
        client_,
        request,
        max_num_draft_tokens=4,
        draft_model_name="draft_model",
        draft_tokenizer=draft_tokenizer_,
        target_model_name="target_model",
        target_tokenizer=target_tokenizer_,
        verbose=True,
    )
