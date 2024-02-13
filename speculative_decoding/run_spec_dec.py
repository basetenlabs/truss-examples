"""Self contained demo script to run speculative inference.

Must be run on host with deps (see truss yaml) and tritonserver installed.
The used engines require at least A100 40GB GPU.

Examples:

```
python run_spec_dec.py --prompt="Once upon" --iteration_delay=1.5 --max_num_generated_tokens=20
python run_spec_dec.py --prompt="How does a car work?" --temperature=0.2 --runtime_top_k=10 --random_seed=123
```
"""
import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path

import colorama
import huggingface_hub
import transformers
import tritonclient.grpc.aio as triton_grpc

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "packages"))
import helpers  # From packages.

import speculative_decoding  # From packages.

TRITON_DIR = os.path.join("/", "packages", "triton_model_repo")

DRAFT_MODEL_ENGINE_HF = "baseten/specdec-draft-gpt2"
DRAFT_MODEL_TOKENIZER_HF = "gpt2"
DRAFT_MODEL_KEY = "draft_model"
TARGET_MODEL_ENGINE_HF = "baseten/specdec-target-mistral-7B"
TARGET_MODEL_TOKENIZER_HF = "mistralai/Mistral-7B-v0.1"
TARGET_MODEL_KEY = "target_model"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_num_generated_tokens", type=int, default=30)
    parser.add_argument("--num_draft_tokens", type=int, default=4)

    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--runtime_top_k", type=int, default=None)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--bad_word_list", type=str, default=None)
    parser.add_argument("--stop_words_list", type=str, default=None)

    parser.add_argument(
        "--concurrent", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--verbose", action=argparse.BooleanOptionalAction, default=True
    )
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
    wdir = os.path.dirname(os.path.abspath(__file__))
    shutil.copytree(
        src=os.path.join(wdir, "packages", "triton_model_repo"),
        dst=os.path.join("/", "packages", "triton_model_repo"),
        dirs_exist_ok=True,
    )
    request = helpers.GenerationRequest(
        prompt=args.prompt,
        max_num_generated_tokens=args.max_num_generated_tokens,
        bad_word_list=args.bad_word_list,
        top_words_list=args.stop_words_list,
    )
    request.sampling_config.temperature = args.temperature
    request.sampling_config.runtime_top_k = args.runtime_top_k
    request.sampling_config.random_seed = args.random_seed
    request.sampling_config.repetition_penalty = args.repetition_penalty

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
        triton_server = helpers.TritonServer(Path("/packages/triton_model_repo"))
        triton_server.load_server_and_model({})

    async def main():
        client = triton_grpc.InferenceServerClient("localhost:8001")

        target_model = speculative_decoding.ModelWrapper(
            client,
            TARGET_MODEL_KEY,
            transformers.AutoTokenizer.from_pretrained(TARGET_MODEL_TOKENIZER_HF),
        )

        draft_model = speculative_decoding.ModelWrapper(
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

        state_co = speculative_decoding.run_speculative_inference(
            target_model,
            draft_model,
            request,
            max_num_draft_tokens=args.num_draft_tokens,
            request_id="666",
            result_queue=asyncio.Queue(),
            verbose=args.verbose,
            iteration_delay=args.iteration_delay,
        )

        direct_gen_co = speculative_decoding.run_conventional_inference(
            target_model,
            request,
            request_id="123",
        )

        if args.concurrent:
            state = asyncio.ensure_future(state_co)
            direct_gen = asyncio.ensure_future(direct_gen_co)
        else:
            state = state_co
            direct_gen = direct_gen_co

        with helpers.timeit("NEW TOTAL - speculative_gen"):
            state_result = await state
            print(f"SpecDec result:\n{state_result.get_verified_text()}")
            if args.verbose:
                print(
                    f"Average num of accepted draft tokens: "
                    f"{state_result.get_aveage_num_accepted_draft_tokens():.2f}"
                )
        with helpers.timeit("OLD TOTAL - direct_gen"):
            direct_text = await direct_gen
            print(f"Direct Gen text:\n{direct_text}\n`")

        helpers.show_timings()

    state = asyncio.run(main())
