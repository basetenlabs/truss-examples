import queue
import time

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype


def prepare_tensor(name, input):
    t = grpcclient.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)
        output = result.as_numpy("text_output")
        print(output, flush=True)


def get_preprocessor_inputs(prompt, output_len, bad_words, stop_words):
    input0 = [[prompt]]
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.uint32) * output_len

    preprocessor_inputs = [
        prepare_tensor("QUERY", input0_data),
        prepare_tensor("REQUEST_OUTPUT_LEN", output0_len),
    ]
    bad_words_list = np.array([bad_words], dtype=object)
    preprocessor_inputs += [prepare_tensor("BAD_WORDS_DICT", bad_words_list)]
    stop_words_list = np.array([stop_words], dtype=object)
    preprocessor_inputs += [prepare_tensor("STOP_WORDS_DICT", stop_words_list)]

    return preprocessor_inputs


def extract_preprocessor_outputs(result):

    input_ids = np.squeeze(result.as_numpy("INPUT_ID").astype(np.int32), axis=0)
    bad_words_ids = result.as_numpy("BAD_WORDS_IDS").astype(np.int32)
    stop_words_ids = result.as_numpy("STOP_WORDS_IDS").astype(np.int32)

    return input_ids, bad_words_ids, stop_words_ids


def get_trtllm_inputs(
    input_ids,
    input_length,
    request_output_len,
    draft_tokens,
    beam_width,
    temperature,
    repetition_penalty,
    presence_penalty,
    bad_words_ids,
    stop_words_ids,
    end_id,
    pad_id,
):

    # input_ids is expected to have shape [input_length]
    # Add batch dimension of 1
    input_ids_data = np.expand_dims(input_ids, axis=0)
    inputs = [
        prepare_tensor("input_ids", input_ids_data),
        prepare_tensor("input_lengths", np.array([[input_length]], dtype=np.int32)),
        prepare_tensor(
            "request_output_len", np.array([[request_output_len]], dtype=np.uint32)
        ),
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

    if end_id is not None:
        end_id_data = np.array([[end_id]], dtype=np.int32)
        inputs.append(prepare_tensor("end_id", end_id_data))

    if pad_id is not None:
        pad_id_data = np.array([[pad_id]], dtype=np.int32)
        inputs.append(prepare_tensor("pad_id", pad_id_data))

    return inputs


def check_result(result, model_name):
    if type(result) == InferenceServerException:
        print(f"Received an error from server while calling {model_name}: {result}")


def extract_trtllm_outputs(result):
    # Get batch 0, beam 0 output_ids
    output_ids = np.squeeze(result.as_numpy("output_ids").astype(np.int32), axis=(0, 1))
    sequence_length_data = result.as_numpy("sequence_length").astype(np.int32)
    assert sequence_length_data.shape[0] == 1
    assert sequence_length_data.shape[1] == 1
    sequence_length = sequence_length_data[0, 0]
    cum_log_probs = result.as_numpy("cum_log_probs").astype(np.float32)
    output_log_probs = result.as_numpy("output_log_probs").astype(np.float32)
    return output_ids, sequence_length, cum_log_probs, output_log_probs


def get_postprocessor_inputs(output_ids, cum_log_probs, output_log_probs):
    output_ids_data = np.expand_dims(output_ids, axis=(0, 1))
    inputs = [
        prepare_tensor("TOKENS_BATCH", output_ids_data),
        prepare_tensor(
            "SEQUENCE_LENGTH", np.array([[len(output_ids)]], dtype=np.int32)
        ),
        prepare_tensor("CUM_LOG_PROBS", cum_log_probs),
        prepare_tensor("OUTPUT_LOG_PROBS", output_log_probs),
    ]

    return inputs


def encountered_stop_words(input_ids, stop_words_ids):
    for stop_word_ids in stop_words_ids:
        if np.array_equal(input_ids[-len(stop_word_ids) :], stop_word_ids):
            return True
    return False


def run_speculative_inference(
    client_draft,
    client_target,
    prompt,
    output_len,
    in_num_draft_tokens,
    request_id,
    repetition_penalty,
    presence_penalty,
    temperature,
    stop_words,
    bad_words,
    end_id,
    pad_id,
    beam_width,
    preprocessor_model_name,
    draft_tensorrt_llm_model_name,
    target_tensorrt_llm_model_name,
    postprocessor_model_name,
    verbose,
):

    # Call the preprocessor
    preprocessor_inputs = get_preprocessor_inputs(
        prompt, output_len, bad_words, stop_words
    )
    preprocessor_result = client_draft.infer(
        preprocessor_model_name, preprocessor_inputs, request_id=request_id
    )
    check_result(preprocessor_result, preprocessor_model_name)
    prompt_input_ids, bad_words_ids, stop_words_ids = extract_preprocessor_outputs(
        preprocessor_result
    )

    input_ids = prompt_input_ids
    last_input_ids = None
    draft_output_ids = None

    while True:
        num_draft_tokens = min(
            in_num_draft_tokens, len(prompt_input_ids) + output_len - len(input_ids) - 1
        )

        if num_draft_tokens > 0:

            if verbose:
                print("Draft model input ids:")
                print(input_ids.tolist())

            # Generate up to num_draft_tokens with draft model

            def foo(input_ids):
                draft_inputs = get_trtllm_inputs(
                    input_ids,
                    len(input_ids),
                    num_draft_tokens,
                    None,
                    beam_width,
                    temperature,
                    repetition_penalty,
                    presence_penalty,
                    bad_words_ids,
                    stop_words_ids,
                    end_id,
                    pad_id,
                )
                t0 = time.time()
                draft_result = client_draft.infer(
                    draft_tensorrt_llm_model_name, draft_inputs, request_id=request_id
                )
                t1 = time.time()
                print(f"{t1-t0}")
                check_result(draft_result, draft_tensorrt_llm_model_name)
                (
                    draft_output_ids,
                    draft_seq_len,
                    cum_log_probs,
                    output_log_probs,
                ) = extract_trtllm_outputs(draft_result)

            foo(input_ids)
            foo(input_ids)
            foo(input_ids)
            foo(input_ids + 2)

            if verbose:
                print("Draft model output ids:")
                print(draft_output_ids.tolist())
                print("draft_sequence_length")
                print(draft_seq_len)

            # Set the draft token and call the target model to generate up to num_draft_tokens + 1
            draft_tokens = draft_output_ids[len(input_ids) : draft_seq_len]

            if verbose:
                print("draft_tokens")
                print(draft_tokens.tolist())

        if verbose:
            print("Target model input ids")
            print(input_ids.tolist())

        # Generate up to len(draft_tokens) + 1 with target model
        target_inputs = get_trtllm_inputs(
            input_ids,
            input_length=len(input_ids),
            request_output_len=len(draft_tokens) + 1 if num_draft_tokens > 0 else 1,
            draft_tokens=draft_tokens if num_draft_tokens > 0 else None,
            beam_width=beam_width,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            bad_words_ids=bad_words_ids,
            stop_words_ids=stop_words_ids,
            end_id=end_id,
            pad_id=pad_id,
        )

        target_result = client_target.infer(
            target_tensorrt_llm_model_name, target_inputs, request_id=request_id
        )
        check_result(target_result, target_tensorrt_llm_model_name)
        (
            target_output_ids,
            seq_length,
            cum_log_probs,
            output_log_probs,
        ) = extract_trtllm_outputs(target_result)

        if verbose:
            print("Target model output_ids")
            print(target_output_ids.tolist())
            print("target seq_length")
            print(seq_length)

        # Store the last iteration input_ids to check if EOS was encountered
        last_input_ids = input_ids
        # Update the input ids with new output_ids
        input_ids = target_output_ids

        # Evaluate criteria to stop generation loop.
        # If we've hit or exceeded the max output length, should stop
        length_stop = len(input_ids) >= len(prompt_input_ids) + output_len
        # If draft and target have same outputs, should stop. Normally target should return 1 more token.
        # If they are the same length, they should differ at the last token
        target_draft_equal = draft_output_ids is not None and np.array_equal(
            draft_output_ids, target_output_ids
        )
        # If tokens no longer change, should stop, means we have hit early stopping
        last_current_equal = np.array_equal(last_input_ids, input_ids)
        # Need to check if stop words was encountered
        hit_stop_words = encountered_stop_words(input_ids, stop_words_ids[0])

        if verbose:
            print("length_stop:", length_stop)
            print("target_draft_equal:", target_draft_equal)
            print("last_current_equal:", last_current_equal)
            print("hit_stop_words:", hit_stop_words)

        if length_stop or target_draft_equal or last_current_equal or hit_stop_words:
            break

    # Call the postprocessor
    postprocessor_inputs = get_postprocessor_inputs(
        input_ids, cum_log_probs, output_log_probs
    )
    postprocessor_result = client_target.infer(
        postprocessor_model_name, postprocessor_inputs, request_id=request_id
    )
    check_result(postprocessor_result, postprocessor_model_name)
    output = postprocessor_result.as_numpy("OUTPUT")
    return output[0].decode("utf8")


if __name__ == "__main__":
    try:
        client_target = grpcclient.InferenceServerClient("0.0.0.0:8001")
        client_draft = grpcclient.InferenceServerClient("0.0.0.0:8001")
    except Exception as e:
        print("client creation failed: " + str(e))

    output_text = run_speculative_inference(
        client_draft,
        client_target,
        prompt="How is your day going?",
        output_len=100,
        in_num_draft_tokens=3,
        request_id="1",
        repetition_penalty=None,
        presence_penalty=None,
        temperature=1.0,
        stop_words=["stop"],
        bad_words=["fuck"],
        end_id=None,
        pad_id=None,
        beam_width=1,
        preprocessor_model_name="preprocessing",
        draft_tensorrt_llm_model_name="tensorrt_llm",
        target_tensorrt_llm_model_name="tensorrt_llm",
        postprocessor_model_name="postprocessing",
        verbose=True,
    )

    # Print the final text
    print("Final text:\n", output_text)
