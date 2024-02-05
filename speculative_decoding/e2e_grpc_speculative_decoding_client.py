import time

import numpy as np
import transformers
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype


class _TimerContextManager:
    def __init__(self, section_name: str) -> None:
        self._section_name = section_name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"Run `{self._section_name}` in `{elapsed_time}` seconds.")


def timeit(section_name: str) -> _TimerContextManager:
    return _TimerContextManager(section_name)


def prepare_tensor(name, input):
    t = grpcclient.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def get_trtllm_inputs(
    input_ids,
    input_length,
    num_gen_tokens,
    draft_tokens,
    beam_width,
    temperature,
    repetition_penalty,
    presence_penalty,
    frequency_penalty,
    # bad_words_ids,
    # stop_words_ids,
    end_id,
    pad_id,
):
    # input_ids is expected to have shape [input_length]
    # Add batch dimension of 1
    input_ids_data = np.expand_dims(input_ids, axis=0)
    inputs = [
        prepare_tensor("input_ids", input_ids_data.astype(np.int32)),
        prepare_tensor("input_lengths", np.array([[input_length]], dtype=np.int32)),
        prepare_tensor(
            "request_output_len", np.array([[num_gen_tokens]], dtype=np.uint32)
        ),
        prepare_tensor("return_log_probs", np.array([[True]], dtype=bool)),
        # The return_X_logits tensor names were only added in
        # https://github.com/NVIDIA/TensorRT-LLM/pull/846.
        # prepare_tensor("return_context_logits", np.array([[True]], dtype=bool)),
        # prepare_tensor("return_generation_logits", np.array([[True]], dtype=bool)),
        # prepare_tensor("bad_words_list", bad_words_ids),
        # prepare_tensor("stop_words_list", stop_words_ids),
        prepare_tensor("beam_width", np.array([[beam_width]], dtype=np.uint32)),
        prepare_tensor("temperature", np.array([[temperature]], dtype=np.float32)),
    ]

    if draft_tokens is not None:
        draft_tokens_data = np.array([draft_tokens], dtype=np.int32)
        inputs.append(prepare_tensor("draft_input_ids", draft_tokens_data))

    if repetition_penalty is not None:
        repetition_penalty_data = np.array([[repetition_penalty]], dtype=np.float32)
        inputs.append(prepare_tensor("repetition_penalty", repetition_penalty_data))

    if presence_penalty is not None:
        presence_penalty_data = np.array([[presence_penalty]], dtype=np.float32)
        inputs.append(prepare_tensor("presence_penalty", presence_penalty_data))

    if frequency_penalty is not None:
        frequency_penalty_data = np.array([[frequency_penalty]], dtype=np.float32)
        inputs.append(prepare_tensor("frequency_penalty", frequency_penalty_data))

    if end_id is not None:
        end_id_data = np.array([[end_id]], dtype=np.uint32)
        inputs.append(prepare_tensor("end_id", end_id_data))

    if pad_id is not None:
        pad_id_data = np.array([[pad_id]], dtype=np.uint32)
        inputs.append(prepare_tensor("pad_id", pad_id_data))

    return inputs


def extract_trtllm_outputs(result):
    # Get batch 0, beam 0 output_ids
    output_ids = np.squeeze(result.as_numpy("output_ids").astype(np.int32), axis=(0, 1))
    sequence_length_data = result.as_numpy("sequence_length").astype(np.int32)
    assert sequence_length_data.shape[0] == 1
    assert sequence_length_data.shape[1] == 1
    sequence_length = sequence_length_data[0, 0]
    assert sequence_length == len(output_ids)
    cum_log_probs = result.as_numpy("cum_log_probs").astype(np.float32)
    output_log_probs = result.as_numpy("output_log_probs").astype(np.float32)
    # print(cum_log_probs)
    # Log probs are always 0, even when `return_log_probs=True` in request.
    # print(output_log_probs)
    # pdb.set_trace()
    # context_logits = result.as_numpy("context_logits").astype(np.float32)
    # generation_logits = result.as_numpy("generation_logits").astype(np.float32)
    # print(generation_logits)
    return output_ids


def encountered_stop_words(input_ids, stop_words_ids):
    for stop_word_ids in stop_words_ids:
        if np.array_equal(input_ids[-len(stop_word_ids) :], stop_word_ids):
            return True
    return False


def run_speculative_inference(
    client,
    prompt,
    request_output_len,
    max_num_draft_tokens,
    request_id,
    repetition_penalty,
    presence_penalty,
    frequency_penalty,
    temperature,
    stop_words,
    bad_words,
    end_id,
    pad_id,
    beam_width,
    draft_model_name,
    target_model_name,
    verbose,
):
    draft_tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    # TODO: check representation of word lists as tokens.
    # draft_bad_words_ids = draft_tokenizer.encode(bad_words)
    # draft_stop_words_ids = draft_tokenizer.encode(stop_words)
    target_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1"
    )

    def call_draft_model(input_text: str) -> str:
        input_ids = draft_tokenizer.encode(input_text)
        num_draft_tokens = min(
            max_num_draft_tokens, request_output_len - len(input_ids)
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

        inputs = get_trtllm_inputs(
            input_ids,
            len(input_ids),
            num_draft_tokens,
            None,  # draft_tokens.
            beam_width,
            temperature,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            # draft_bad_words_ids,
            # draft_stop_words_ids,
            draft_tokenizer.eos_token_id,
            draft_tokenizer.pad_token_id,
        )
        result = client.infer(draft_model_name, inputs, request_id=request_id)
        output_ids = extract_trtllm_outputs(result)

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
        inputs = get_trtllm_inputs(
            confirmed_ids_re_encoded,
            len(confirmed_ids_re_encoded),
            num_gen_tokens,
            draft_ids or None,
            beam_width,
            temperature,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            # target_bad_words_ids,
            # target_stop_words_ids,
            target_tokenizer.eos_token_id,
            target_tokenizer.pad_token_id,
        )
        result = client.infer(target_model_name, inputs, request_id=request_id)
        output_ids = extract_trtllm_outputs(result)

        clear_text_full = target_tokenizer.decode(output_ids, skip_special_tokens=True)
        if verbose:
            print(
                f"  Result TARGET model:\n\t{output_ids}\n"
                f"\tResult: `{clear_text_full}`"
            )
        return clear_text_full, output_ids

    def generate_target_model(input_text: str, num_gen_tokens: str) -> str:
        input_ids = target_tokenizer.encode(input_text)
        inputs = get_trtllm_inputs(
            input_ids,
            len(input_ids),
            num_gen_tokens - len(input_ids),
            None,
            beam_width,
            temperature,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            # target_bad_words_ids,
            # target_stop_words_ids,
            target_tokenizer.eos_token_id,
            target_tokenizer.pad_token_id,
        )
        result = client.infer(target_model_name, inputs, request_id=request_id)
        output_ids = extract_trtllm_outputs(result)
        clear_text_full = target_tokenizer.decode(output_ids, skip_special_tokens=True)
        if verbose:
            print(
                f"  Result TARGET model:\n\t{output_ids}\n"
                f"\tResult: `{clear_text_full}`"
            )
        return clear_text_full

    # Warmup model with unrelated string.
    generate_target_model("What is a computer", request_output_len)
    call_draft_model("What is a computer")

    # Run `direct_gen` in `0.26770877838134766` seconds.
    # with timeit("direct_gen"):
    #     generate_target_model(prompt, request_output_len)

    # Run `speculative_gen` in `0.1786513328552246` seconds.
    with timeit("speculative_gen"):
        confirmed_target_ids = target_tokenizer.encode(prompt)
        current_text = prompt
        while True:
            with timeit("draft"):
                current_text = call_draft_model(current_text)
            with timeit("target"):
                current_text, confirmed_target_ids = call_target_model(
                    current_text, confirmed_target_ids
                )

            print("=======================")
            print(f"Target token len {len(confirmed_target_ids)}")
            if len(confirmed_target_ids) == request_output_len:
                break

    with timeit("direct_gen"):
        print(generate_target_model(prompt, request_output_len))

    return current_text


if __name__ == "__main__":
    client_ = grpcclient.InferenceServerClient("0.0.0.0:8001")

    output_text = run_speculative_inference(
        client_,
        prompt="Once upon a time there was",
        request_output_len=25,
        max_num_draft_tokens=4,
        request_id="1",
        repetition_penalty=None,
        presence_penalty=None,
        frequency_penalty=None,
        temperature=1.0,
        stop_words=[],
        bad_words=[],
        end_id=None,
        pad_id=None,
        beam_width=1,
        draft_model_name="draft_model",
        target_model_name="target_model",
        verbose=True,
    )

    # Print the final text
    print("Final text:\n", output_text)
