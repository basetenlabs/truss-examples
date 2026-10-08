export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/workspace/flash-attention:$PYTHONPATH"
export SGLANG_ENABLE_FP8_QUANTIZATION=0
export SGLANG_DIFFUSION_USE_CUTE_DSL_FLASH_ATTN=1
export SGLANG_DIFFUSION_ENABLE_CUDA_GRAPH_CAPTURE=0

GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | uniq)

if [[ "$GPU_TYPE" == *"H100"* ]]; then
    export TEXT_ENCODER_CPU_OFFLOAD=true
elif [[ "$GPU_TYPE" == *"B200"* ]]; then
    export TEXT_ENCODER_CPU_OFFLOAD=false
else
    echo "Unsupported GPU type: $GPU_TYPE"
    exit 1
fi

unset B10_CPU_MEMORY_SAVING
/workspace/venv/bin/sglang serve \
    --model-path /app/model_cache/FLUX.2-dev \
    --dit-cpu-offload false \
    --text-encoder-cpu-offload $TEXT_ENCODER_CPU_OFFLOAD \
    --image-encoder-cpu-offload false \
    --vae-cpu-offload false \
    --num-gpus 1 \
    --port 8000 \
    --host 0.0.0.0