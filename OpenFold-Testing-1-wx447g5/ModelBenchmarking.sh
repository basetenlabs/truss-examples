uv run --reinstall --with tiktoken --with /Users/tesslipsky/repos/genai-bench genai-bench benchmark \
  --api-backend baseten \
  --api-base "https://model-03y82lew.api.baseten.co/environments/production/sync/v1/chat/completions" \
  --task "text-to-text" \
  --api-model-name "qwen-ai/Qwen3-Next-80B-A3B-Instruct-Dedicated" \
  --model-tokenizer "qwen-ai/Qwen3-Next-80B-A3B-Instruct-Dedicated" \
  --traffic-scenario "N(10000,1000)/(1000,100)" \
  --num-workers 32 \
  --max-requests-per-run 15000 \
  --num-concurrency 512 \
  --max-time-per-run 5

uv run --reinstall --with tiktoken --with /Users/jiegong/jieg/genai-bench genai-bench benchmark \
  --api-backend baseten \
  --api-base "https://model-03y8zylw.api.baseten.co/environments/production/sync/v1/chat/completions" \
  --task "text-to-text" \
  --api-model-name "zai-org/GLM-4.6" \
  --model-tokenizer "zai-org/GLM-4.6" \
  --traffic-scenario "N(10000,1000)/(1000,100)" \
  --num-workers 32 \
  --max-requests-per-run 15000 \
  --num-concurrency 512 \
  --max-time-per-run 5


 uv run --reinstall --with tiktoken --with /Users/tesslipsky/repos/genai-bench genai-bench benchmark \
  --api-backend baseten \
  --api-base "https://model-03y82lew.api.baseten.co/environments/production/sync/v1/chat/completions" \
  --task "text-to-text" \
  --api-model-name "Qwen/Qwen3-Next-80B-A3B-Instruct" \
  --model-tokenizer "Qwen/Qwen3-Next-80B-A3B-Instruct" \
  --traffic-scenario "N(10000,1000)/(1000,100)" \
  --num-workers 32 \
  --max-requests-per-run 15000 \
  --num-concurrency 512 \
  --max-time-per-run 5

export MODEL_NAME="https://model-03y82lew.api.baseten.co/environments/production/sync/v1"
 uv run --reinstall --with tiktoken --with /Users/tesslipsky/repos/genai-bench genai-bench benchmark \
  --api-backend baseten \
  --api-base "https://inference.baseten.co/v1/chat/completions" \
  --task "text-to-text" \
  --api-model-name "${MODEL_NAME}" \
  --model-tokenizer "${MODEL_NAME}" \
  --traffic-scenario "D(100,100)" \
  --num-workers 8 \
  --max-requests-per-run 1000 \
  --max-time-per-run 1 \
  --num-concurrency 2


export MODEL_NAME="https://model-03y82lew.api.baseten.co/development/predict"
uv run --with tiktoken --with git+https://github.com/basetenlabs/genai-bench.git genai-bench benchmark \
  --api-backend baseten \
  --api-base "https://inference.baseten.co/v1/chat/completions" \
  --task "text-to-text" \
  --api-model-name "${MODEL_NAME}" \
  --model-tokenizer "${MODEL_NAME}" \
  --traffic-scenario "D(100,100)" \
  --num-workers 8 \
  --max-requests-per-run 1000 \
  --max-time-per-run 1 \
  --num-concurrency 1 --num-concurrency 2


uv run --reinstall --with tiktoken --with /Users/tesslipsky/repos/genai-bench genai-bench benchmark \
  --api-backend baseten \
  --api-base "https://model-03y82lew.api.baseten.co/environments/production/sync/v1/chat/completions" \
  --task "text-to-text" \
  --api-model-name "qwen-ai/Qwen3-Next-80B-A3B-Instruct-Dedicated" \
  --model-tokenizer "qwen-ai/Qwen3-Next-80B-A3B-Instruct-Dedicated" \
  --traffic-scenario "N(10000,1000)/(1000,100)" \
  --num-workers 8 \
  --max-requests-per-run 1000 \
  --num-concurrency 1 \
  --max-time-per-run 1


  uv run --reinstall --with tiktoken --with /Users/tesslipsky/repos/genai-bench genai-bench benchmark \
  --api-backend baseten \
  --api-base "https://model-03y82lew.api.baseten.co/environments/production/sync/v1/chat/completions" \
  --task "text-to-text" \
  --api-model-name "Qwen/Qwen3-Next-80B-A3B-Instruct" \
  --model-tokenizer "Qwen/Qwen3-Next-80B-A3B-Instruct" \
  --traffic-scenario "N(10000,1000)/(1000,100)" \
  --num-workers 8 \
  --max-requests-per-run 1000 \
  --num-concurrency 1 \
  --max-time-per-run 1