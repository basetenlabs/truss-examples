from huggingface_hub import snapshot_download

MODEL_NAME = "gpt2"
MODEL_HF_ID = "gpt2"

TRT_LLM_BUILD_SCRIPT = "/app/tensorrt_llm/examples/gpt/build.py"
TRT_LLM_CONVERT_SCRIPT = "/app/tensorrt_llm/examples/gpt/hf_gpt_convert.py"
BUILD_PYTOHON_BIN = "/usr/bin/python"


HF_DIR = f"/root/workbench/{MODEL_NAME}_hf"
FT_DIR = f"/root/workbench/{MODEL_NAME}_ft"
ENGINE_DIR = f"/root/workbench/{MODEL_NAME}_engine"


snapshot_download(
    MODEL_HF_ID,
    local_dir=HF_DIR,
    local_dir_use_symlinks=False,
    max_workers=4,
)


# Convert weights command.
command = [
    BUILD_PYTOHON_BIN,
    TRT_LLM_CONVERT_SCRIPT,
    f"-i={HF_DIR}",
    f"-o={FT_DIR}",
    "--tensor-parallelism=1",
    "--storage-type=float16",
]
print(" ".join(command))


# Build engine command.
command = [
    BUILD_PYTOHON_BIN,
    TRT_LLM_BUILD_SCRIPT,
    f"--model_dir={FT_DIR}/1-gpu",
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
    f"--output_dir={ENGINE_DIR}",
]
print(" ".join(command))
