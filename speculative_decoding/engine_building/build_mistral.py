from huggingface_hub import snapshot_download

MODEL_NAME = "mistral-7B-Instruct"
MODEL_HF_ID = "mistralai/Mistral-7B-Instruct-v0.2"

TRT_LLM_BUILD_SCRIPT = "/app/tensorrt_llm/examples/llama/build.py"
BUILD_PYTOHON_BIN = "/usr/bin/python"


MAX_DRAFT_LEN = 5

HF_DIR = f"/root/workbench/{MODEL_NAME}_hf"
ENGINE_DIR = f"/root/workbench/{MODEL_NAME}_engine_draft_{MAX_DRAFT_LEN}"


snapshot_download(
    MODEL_HF_ID,
    local_dir=HF_DIR,
    local_dir_use_symlinks=False,
    max_workers=8,
    resume_download=True,
)

# Build engine command.
command = [
    BUILD_PYTOHON_BIN,
    TRT_LLM_BUILD_SCRIPT,
    f"--model_dir={HF_DIR}",
    "--dtype=float16",
    "--remove_input_padding",
    "--use_gpt_attention_plugin=float16",
    "--enable_context_fmha",
    "--paged_kv_cache",
    "--use_inflight_batching",
    "--use_gemm_plugin=float16",
    "--multi_block_mode",
    "--max_batch_size=32",
    "--max_input_len=1024",
    "--max_output_len=2048",
    "--use_paged_context_fmha",
    f"--max_draft_len={MAX_DRAFT_LEN}" if MAX_DRAFT_LEN else "",
    f"--output_dir={ENGINE_DIR}",
]
print(" ".join(command))
