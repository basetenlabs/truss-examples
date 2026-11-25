#!/usr/bin/env bash
# Serve Nemotron-3.5-Transcribe-30B-A4B with NVIDIA's model-card vLLM flags.
# BASETEN_MODEL_PATH is the base checkpoint mount, or the merged checkpoint a
# Baseten Training deployment mounts instead.
set -euo pipefail
: "${BASETEN_MODEL_PATH:?BASETEN_MODEL_PATH must name the checkpoint directory}"

VENV=/opt/nemo-speech/venv
# FlashInfer (ReplaySSM module) and Triton JIT-compile at startup with the
# image's CUDA toolkit; FlashInfer's fused-MoE and sampling kernels come
# prebuilt from flashinfer-jit-cache (config.yaml).
export CUDA_HOME=/usr/local/cuda
export PATH="${VENV}/bin:${CUDA_HOME}/bin:${PATH}"

# Cold engine init takes ~400 s (749 s before the prebuilt kernels). The BDN startup cache
# (ORG_ENABLE_BDN_STARTUP_CACHE) sets the JIT cache paths (VLLM_CACHE_ROOT,
# TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR, FLASHINFER_WORKSPACE_BASE, ...)
# and restores them across replicas and restarts, keyed by GPU type.

# config.json names pretrained_llm as the relative path "llm_backbone".
cd "${BASETEN_MODEL_PATH}"

# Knobs (config.yaml documents them). Every fallback below matches config.yaml's value, so a
# deployment whose environment block lacks a knob still serves the same configuration.
# BASETEN_MODEL_PATH is the NVFP4 mount (/app/checkpoint/nvfp4) or a Training checkpoint.
KV_CACHE_DTYPE="${NEMOTRON_KV_CACHE_DTYPE:-fp8}"
case "${KV_CACHE_DTYPE}" in
    auto|fp8|fp8_e4m3|bfloat16) ;;
    *) echo "NEMOTRON_KV_CACHE_DTYPE must be auto, fp8, fp8_e4m3 or bfloat16, got '${KV_CACHE_DTYPE}'" >&2; exit 1 ;;
esac
GPU_UTIL="${NEMOTRON_GPU_MEMORY_UTILIZATION:-0.88}"
if [[ ! "${GPU_UTIL}" =~ ^0\.[0-9]+$ ]]; then
    echo "NEMOTRON_GPU_MEMORY_UTILIZATION must be a fraction like 0.90, got '${GPU_UTIL}'" >&2
    exit 1
fi

# Speculative decoding: the card's MTP with 8 draft tokens.
SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":8}'
# Audio-encoder CUDA graphs (default on): NEMOTRON_ENCODER_CUDA_GRAPHS=1 turns on NeMo Speech's
# opt-in EncoderGraphRunner (speechlm2/vllm/salm/encoder_graphs.py, read from the model config via
# --hf-overrides "encoder_cuda_graphs"): the ParallelExpertEncoder forward, its long-audio ASR
# windows and streaming Sortformer steps replay from CUDA graphs captured at the first encoder call
# instead of launching ~1.8k small kernels eagerly per call. NEMOTRON_ENCODER_COMPILE=1 (needs
# graphs on) also torch.compiles every encoder block ("encoder_compile"; fuses LayerNorm/RoPE/
# residual ops, bf16-rounding-level numerics change). A failed capture logs "Encoder CUDA graph
# capture failed" and the encoder runs eagerly. Both 0: no --hf-overrides flag.
ENCODER_GRAPHS="${NEMOTRON_ENCODER_CUDA_GRAPHS:-1}"
ENCODER_COMPILE="${NEMOTRON_ENCODER_COMPILE:-1}"
case "${ENCODER_GRAPHS}:${ENCODER_COMPILE}" in
    0:0) ENCODER_OVERRIDES="" ;;
    1:0) ENCODER_OVERRIDES='"encoder_cuda_graphs":true' ;;
    1:1) ENCODER_OVERRIDES='"encoder_cuda_graphs":true,"encoder_compile":true' ;;
    *) echo "NEMOTRON_ENCODER_CUDA_GRAPHS and NEMOTRON_ENCODER_COMPILE must be 0 or 1 (compile needs graphs), got '${ENCODER_GRAPHS}' and '${ENCODER_COMPILE}'" >&2; exit 1 ;;
esac
# With compile on, data/encoder_flex.py forces FlexAttention's main Triton kernel in the encoder
# (inductor's flex_decoding kernel, which it picks for clips under 128 encoder tokens, broke
# short-clip transcripts). A .pth file in the venv loads it into the vLLM processes; the vLLM
# argv is unchanged.
ENCODER_FLEX_PTH="$("${VENV}/bin/python3" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nemotron_encoder_flex.pth"
if [[ "${ENCODER_COMPILE}" == 1 ]]; then
    echo "import sys, importlib.util as u; s = u.spec_from_file_location('encoder_flex', '/app/data/encoder_flex.py'); m = u.module_from_spec(s); sys.modules['encoder_flex'] = m; s.loader.exec_module(m); m.install()" > "${ENCODER_FLEX_PTH}"
else
    rm -f "${ENCODER_FLEX_PTH}"
fi
# Model card: CTC word timestamps through the middleware. The encoder keys go in --hf-overrides.
EXTRA_ARGS=(--middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware)
if [[ -n "${ENCODER_OVERRIDES}" ]]; then
    EXTRA_ARGS+=(--hf-overrides "{${ENCODER_OVERRIDES}}")
fi
# vLLM API server processes (--api-server-count; default 4). "1" passes no flag, so the argv is
# unchanged; N > 1 runs N frontend processes (HTTP, audio decode, NeMo processor and the CTC
# middleware, each loaded per process) on one engine core. vLLM 0.32 then disables its
# periodic stats log lines (Prometheus metrics stay). See README "API server processes".
API_SERVER_COUNT="${NEMOTRON_API_SERVER_COUNT:-4}"
if [[ ! "${API_SERVER_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NEMOTRON_API_SERVER_COUNT must be a positive integer, got '${API_SERVER_COUNT}'" >&2
    exit 1
fi
API_ARGS=()
if [[ "${API_SERVER_COUNT}" != 1 ]]; then
    API_ARGS=(--api-server-count "${API_SERVER_COUNT}")
fi

# Server-side chunking of long audio (data/chunk_proxy.py and data/long_audio.py, ported
# from the H100 batch preset, #1195; README "Long audio"). On by default (600 s chunks above
# 2400 s). NEMOTRON_CHUNK_MAX_SECONDS empty or 0 turns it off: vLLM is exec'd on :8000
# with exactly the flags it had before (unless the guard is on). On: audio longer than
# NEMOTRON_CHUNK_TRIGGER_SECONDS (default 2400) is cut at a pause into chunks of at most
# NEMOTRON_CHUNK_MAX_SECONDS (default 600), each starting
# NEMOTRON_CHUNK_OVERLAP_SECONDS (default 5) before the previous cut; the chunks run at once
# and their transcripts are stitched into one response. vLLM then listens on 127.0.0.1:8001
# behind the proxy on :8000. The proxy validates every knob first (--check), so a bad value
# fails the deploy.
CHUNK_MAX="${NEMOTRON_CHUNK_MAX_SECONDS-600}"
# long_audio.ChunkConfig's own defaults are 3300 / 3600; pin the resolved values for the proxy.
export NEMOTRON_CHUNK_MAX_SECONDS="${CHUNK_MAX}"
export NEMOTRON_CHUNK_TRIGGER_SECONDS="${NEMOTRON_CHUNK_TRIGGER_SECONDS:-2400}"
CHUNKING=1
if [[ "${CHUNK_MAX}" =~ ^0*(\.0*)?$ ]]; then
    CHUNKING=0
fi
# Runaway guard (data/runaway.py, README "Runaway guard"), ported from #1195 (054fc0e). ON
# by default; "0" turns it off. "1" caps each engine request's max_tokens by its
# audio duration and retries/cleans runaway transcripts. It lives in the proxy, so guard on
# puts the proxy in front of vLLM even with chunking off (every single-audio request then
# goes through the proxy's file path). runaway.GuardConfig's own default is on; pin it.
GUARD="${NEMOTRON_RUNAWAY_GUARD:-1}"
export NEMOTRON_RUNAWAY_GUARD="${GUARD}"
case "${GUARD}" in
    0|1) ;;
    *) echo "NEMOTRON_RUNAWAY_GUARD must be 0 or 1, got '${GUARD}'" >&2; exit 1 ;;
esac
PROXY=0
if [[ "${CHUNKING}" == "1" || "${GUARD}" == "1" ]]; then
    PROXY=1
fi
PROXY_PY="${VENV}/bin/python3"
CHUNK_DIR="${NEMOTRON_CHUNK_DIR:-/tmp/nemotron-chunks}"
if [[ "${PROXY}" == "1" ]]; then
    "${PROXY_PY}" /app/data/chunk_proxy.py --check
    mkdir -p "${CHUNK_DIR}"
    VLLM_HOST=127.0.0.1
    VLLM_PORT=8001
    # The proxy hands vLLM each chunk (and each downloaded URL) as a file:// URL in CHUNK_DIR.
    MEDIA_ARGS=(--allowed-local-media-path "${CHUNK_DIR}")
else
    VLLM_HOST=0.0.0.0
    VLLM_PORT=8000
    MEDIA_ARGS=()
fi
# NEMOTRON_PROXY_RETRY=1 (opt-in, default 0; chunk_proxy.py): the proxy retries the audio fetch
# and each engine POST once on a dropped connection (ServerDisconnected HTTP 400s) and keeps
# pooled connections under vLLM's 5 s keep-alive. The proxy validated the value above (--check).
if [[ "${NEMOTRON_PROXY_RETRY:-0}" == "1" && "${PROXY}" == "0" ]]; then
    echo "NEMOTRON_PROXY_RETRY=1 needs the chunk proxy (chunking or the runaway guard on)" >&2
    exit 1
fi
echo "start.sh: model=${BASETEN_MODEL_PATH} kv_cache_dtype=${KV_CACHE_DTYPE} gpu_util=${GPU_UTIL} spec=${SPEC_CONFIG} v2_runner=${VLLM_USE_V2_MODEL_RUNNER-unset} api_servers=${API_SERVER_COUNT} proxy=${PROXY} guard=${GUARD} chunking=${CHUNKING} chunk_max_s=${CHUNK_MAX} chunk_trigger_s=${NEMOTRON_CHUNK_TRIGGER_SECONDS} encoder_graphs=${ENCODER_GRAPHS} encoder_compile=${ENCODER_COMPILE} max_num_seqs=${NEMOTRON_MAX_NUM_SEQS:-384} max_num_batched_tokens=${NEMOTRON_MAX_NUM_BATCHED_TOKENS:-32768} proxy_retry=${NEMOTRON_PROXY_RETRY:-0}" >&2

# Card flags for vLLM 0.32.0: ReplaySSM with the FlashInfer Mamba backend and
# prefix caching off. Deviation from the card: thinking off by default (clients
# that omit chat_template_kwargs get a transcript). NeMo Speech main registers
# the current architecture name. MoE backend: vLLM's auto selection. The served
# name is the BF16 repo id.
VLLM_CMD=("${VENV}/bin/vllm" serve "${BASETEN_MODEL_PATH}" \
    --tokenizer "${BASETEN_MODEL_PATH}" \
    --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --trust-remote-code \
    --dtype bfloat16 \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-model-len 65536 \
    --max-num-seqs "${NEMOTRON_MAX_NUM_SEQS:-384}" \
    --max-num-batched-tokens "${NEMOTRON_MAX_NUM_BATCHED_TOKENS:-32768}" \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --use-replayssm \
    --mamba-backend flashinfer \
    --mamba-cache-mode none \
    --mm-processor-cache-gb 0 \
    --no-async-scheduling \
    --disable-uvicorn-access-log \
    --limit-mm-per-prompt '{"audio":{"count":1,"length":120000000}}' \
    --speculative-config "${SPEC_CONFIG}" \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --reasoning-parser nemotron_v3 \
    --trust-request-mm-kwargs \
    "${EXTRA_ARGS[@]}" \
    "${MEDIA_ARGS[@]}" \
    "${API_ARGS[@]}")

if [[ "${PROXY}" == "0" ]]; then
    exec "${VLLM_CMD[@]}"
fi
# vLLM behind the proxy. If either exits, stop the other and exit non-zero so the pod restarts
# (a proxy without an engine would answer 503 forever).
"${VLLM_CMD[@]}" &
VLLM_PID=$!
"${PROXY_PY}" /app/data/chunk_proxy.py --host 0.0.0.0 --port 8000 --upstream "http://127.0.0.1:${VLLM_PORT}" &
PROXY_PID=$!
trap 'kill -TERM "${VLLM_PID}" "${PROXY_PID}" 2>/dev/null || true' TERM INT
set +e
wait -n "${VLLM_PID}" "${PROXY_PID}"
STATUS=$?
echo "start.sh: vLLM or the chunk proxy exited (status ${STATUS}); stopping both" >&2
kill -TERM "${VLLM_PID}" "${PROXY_PID}" 2>/dev/null
wait
exit $(( STATUS == 0 ? 1 : STATUS ))
