import os

import colorama
import helpers
import huggingface_hub
import spec_dec
import transformers
import tritonclient.grpc as triton_grpc

if __name__ == "__main__":

    colorama.init(autoreset=True)

    TRITON_DIR = os.path.join("/", "triton_model_repo")

    DRAFT_MODEL_ENGINE_HF = "baseten/specdec-draft-gpt2"
    DRAFT_MODEL_TOKENIZER_HF = "gpt2"
    DRAFT_MODEL_KEY = "draft_model"
    TARGET_MODEL_ENGINE_HF = "baseten/specdec-target-mistral-7B"
    TARGET_MODEL_TOKENIZER_HF = "mistralai/Mistral-7B-v0.1"
    TARGET_MODEL_KEY = "target_model"

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

    request = helpers.GenerationRequest(
        # prompt="Once upon a time there was",
        prompt="Once upon",
        max_num_generated_tokens=60,
        request_id="123",
    )
    request.sampling_config.random_seed = 123412
    request.sampling_config.temperature = 3.0

    # Warmup models with unrelated string.
    target_model.generate("What is a computer?", 4, "111", request.sampling_config)
    draft_model.generate("What is a computer?", 4, "111", request.sampling_config)

    helpers.enable_timing()

    with helpers.timeit("A - speculative_gen"):
        state = spec_dec.run_speculative_inference(
            target_model,
            draft_model,
            request,
            max_num_draft_tokens=4,
            verbose=True,
        )

    print(f"Final text:\n{state.get_current_text()}")
    print(
        f"Average num of accepted draft tokens: "
        f"{state.get_aveage_num_accepted_draft_tokens():.2f}"
    )

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
