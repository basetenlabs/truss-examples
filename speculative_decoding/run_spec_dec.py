import argparse
import asyncio
import os
from typing import Optional

import colorama
import huggingface_hub
import transformers
import tritonclient.grpc.aio as triton_grpc
from model import helpers, spec_dec

TRITON_DIR = os.path.join("/", "triton_model_repo")

DRAFT_MODEL_ENGINE_HF = "baseten/specdec-draft-gpt2"
DRAFT_MODEL_TOKENIZER_HF = "gpt2"
DRAFT_MODEL_KEY = "draft_model"
TARGET_MODEL_ENGINE_HF = "baseten/specdec-target-mistral-7B"
TARGET_MODEL_TOKENIZER_HF = "mistralai/Mistral-7B-v0.1"
TARGET_MODEL_KEY = "target_model"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_num_generated_tokens", type=int, default=60)

    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--runtime_top_k", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=None)

    parser.add_argument("--bad_word_list", type=str, default=None)
    parser.add_argument("--stop_words_list", type=str, default=None)

    parser.add_argument("--concurrent", type=bool, default=False)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--iteration_delay", type=float, default=0.0)
    args = parser.parse_args()

    if args.bad_word_list:
        args.bad_word_list = args.bad_word_list.split(",")
    if args.stop_words_list:
        args.stop_words_list = args.stop_words_list.split(",")

    return args


if __name__ == "__main__":
    colorama.init(autoreset=True)
    args = parse_arguments()

    request = helpers.GenerationRequest(
        prompt=args.prompt,
        max_num_generated_tokens=args.max_num_generated_tokens,
        request_id="123",
        bad_word_list=args.bad_word_list,
        top_words_list=args.stop_words_list,
    )
    request.sampling_config.temperature = args.temperature
    request.sampling_config.runtime_top_k = args.runtime_top_k
    request.sampling_config.random_seed = args.random_seed

    huggingface_hub.snapshot_download(
        DRAFT_MODEL_ENGINE_HF,
        local_dir=os.path.join(TRITON_DIR, DRAFT_MODEL_KEY, "1"),
        local_dir_use_symlinks=True,  # True for dev, False for prod.
        max_workers=4,
    )
    huggingface_hub.snapshot_download(
        TARGET_MODEL_ENGINE_HF,
        local_dir=os.path.join(TRITON_DIR, TARGET_MODEL_KEY, "1"),
        local_dir_use_symlinks=True,
        max_workers=4,
    )

    if not helpers.is_triton_server_alive():
        triton_server = helpers.TritonServer(
            "/root/workbench/truss-examples/speculative_decoding/triton_model_repo"
        )
        triton_server.load_server_and_model({})

    async def main():
        client = triton_grpc.InferenceServerClient("0.0.0.0:8001")

        target_model = spec_dec.ModelWrapper(
            client,
            TARGET_MODEL_KEY,
            transformers.AutoTokenizer.from_pretrained(TARGET_MODEL_TOKENIZER_HF),
        )

        draft_model = spec_dec.ModelWrapper(
            client,
            DRAFT_MODEL_KEY,
            transformers.AutoTokenizer.from_pretrained(DRAFT_MODEL_TOKENIZER_HF),
        )

        # Warmup models with unrelated string.
        await target_model.generate(
            "What is a computer?", 4, "111", request.sampling_config
        )
        await draft_model.generate(
            "What is a computer?", 4, "111", request.sampling_config
        )

        helpers.enable_timing()

        with helpers.timeit("A - speculative_gen"):
            state_co = spec_dec.run_speculative_inference(
                target_model,
                draft_model,
                request,
                max_num_draft_tokens=4,
                result_queue=asyncio.Queue(),
                verbose=args.verbose,
                iteration_delay=args.iteration_delay,
            )

        with helpers.timeit("B - direct_gen"):
            direct_gen_co = target_model.generate(
                request.prompt,
                request.max_num_generated_tokens,
                request.request_id + "123",
                request.sampling_config,
                request.bad_word_list,
                request.stop_words_list,
            )

        if args.concurrent:
            state = asyncio.ensure_future(state_co)
            direct_gen = asyncio.ensure_future(direct_gen_co)
        else:
            state = state_co
            direct_gen = direct_gen_co

        with helpers.timeit("A - await speculative_gen"):
            state_result = await state
            print(f"Final text:\n{state_result.get_current_text()}")
            print(
                f"Average num of accepted draft tokens: "
                f"{state_result.get_aveage_num_accepted_draft_tokens():.2f}"
            )
        with helpers.timeit("B - await direct_gen"):
            direct_gen_result = await direct_gen
            print(f"Direct Gen text:\n{direct_gen_result}")

        helpers.show_timings()

    state = asyncio.run(main())

"""
Concurrent:
A - speculative_gen            0.000006  0.000006   1  174762.666667  0.000006  0.000006
B - direct_gen                 0.000007  0.000007   1  144631.172414  0.000007  0.000007
A - await speculative_gen      1.046489  1.046489   1       0.955577  1.046489  1.046489
Generate(draft_model)          0.424932  0.020235  21      49.419699  0.006953  0.027036
Generate(target_model)         0.901044  0.901044   1       1.109824  0.901044  0.901044
Tokenize for Target model      0.005091  0.000242  21    4124.772127  0.000167  0.000302
Verify+Generate(target_model)  0.598043  0.028478  21      35.114506  0.015787  0.035304
B - await direct_gen           0.000018  0.000018   1   56679.783784  0.000018  0.000018

A - speculative_gen            0.000007  0.000007   1  149796.571429  0.000007  0.000007
B - direct_gen                 0.000008  0.000008   1  131072.000000  0.000008  0.000008
A - await speculative_gen      0.516541  0.516541   1       1.935953  0.516541  0.516541
Generate(draft_model)          0.152776  0.007275  21     137.456357  0.006834  0.007958
Tokenize for Target model      0.005022  0.000239  21    4181.758724  0.000187  0.000292
Verify+Generate(target_model)  0.339970  0.016189  21      61.770136  0.015723  0.017120
B - await direct_gen           0.846346  0.846346   1       1.181550  0.846346  0.846346
Generate(target_model)         0.845856  0.845856   1       1.182234  0.845856  0.845856
"""
