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

# Startup levers (Model Scout startup study; defaults reproduce main).
# NEMOTRON_FLASHINFER_AUTOTUNE: 1 (default) lets vLLM autotune FlashInfer kernels at start
# (92-173 s on RTX PRO 6000 every boot: vLLM 0.32 runs the tuning passes even with a restored cache);
# 0 passes --no-enable-flashinfer-autotune, so FlashInfer uses its heuristic tactics and the
# ReplaySSM autotune warmup is skipped too.
AUTOTUNE="${NEMOTRON_FLASHINFER_AUTOTUNE:-1}"
case "${AUTOTUNE}" in
    1) AUTOTUNE_ARGS=() ;;
    0) AUTOTUNE_ARGS=(--no-enable-flashinfer-autotune) ;;
    *) echo "NEMOTRON_FLASHINFER_AUTOTUNE must be 0 or 1, got '${AUTOTUNE}'" >&2; exit 1 ;;
esac
# NEMOTRON_AUTOTUNE_REUSE: 1 (default) lets a warm start skip FlashInfer tuning for the ops in
# the autotune file the BDN startup cache restored, but only when that file is the one vLLM loads
# for this exact config (same vLLM config hash) and has a checkpointing_ssu entry; serving then
# takes their tactics from the file. FlashInfer re-profiles checkpointing_ssu every start
# otherwise (cold-L2 policy ignores file entries while tuning). The check runs inside vLLM
# (autotune_hook/sitecustomize.py, right before kernel warmup loads the file), since the hash needs
# vLLM's built config. Cold starts and mismatches (another config hash, FlashInfer version or GPU
# name) tune as before; autotune_reuse.py logs why and cites the vLLM/FlashInfer source. Ported
# from the H100 streaming preset. 0 opts out. An explicit VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS wins.
REUSE="${NEMOTRON_AUTOTUNE_REUSE:-1}"
[[ "${REUSE}" =~ ^[01]$ ]] || { echo "NEMOTRON_AUTOTUNE_REUSE must be 0 or 1, got '${REUSE}'" >&2; exit 1; }
SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
# Read VLLM_CACHE_ROOT with `-` plus an emptiness test rather than `:-`: the
# `NAME:` form trips tests/test_registry_jit_cache_dirs.py, which reads it as
# an assignment in a file that mentions /cache/org. vLLM's default otherwise (vllm/envs.py:712).
VLLM_ROOT="${VLLM_CACHE_ROOT-}"
[[ -n "${VLLM_ROOT}" ]] || VLLM_ROOT="${XDG_CACHE_HOME:-${HOME:-/root}/.cache}/vllm"
AUTOTUNE_ROOT="${VLLM_ROOT}/flashinfer_autotune_cache"
USER_SKIP_OPS="${VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS+set}"
REUSE_HOOK_DIR="${SCRIPT_DIR}/autotune_hook"
# Arms the in-vLLM reuse check for the next `vllm serve` (called before each start).
apply_reuse() {
    [[ "${REUSE}" == 1 ]] || return 0
    if [[ "${AUTOTUNE}" != 1 ]]; then
        echo "autotune_reuse: autotune is off, nothing to reuse" >&2
    elif [[ -n "${USER_SKIP_OPS}" ]]; then
        echo "autotune_reuse: VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS already set ('${VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS}'), leaving it" >&2
    else
        export NEMOTRON_AUTOTUNE_REUSE_HOOK=1
        case ":${PYTHONPATH-}:" in
            *":${REUSE_HOOK_DIR}:"*) ;;
            *) export PYTHONPATH="${REUSE_HOOK_DIR}${PYTHONPATH:+:${PYTHONPATH}}" ;;
        esac
    fi
}
# NEMOTRON_DIAG_RESTART_ONCE: 0 (default) starts vLLM once. 1 is a diagnostic for warm-start
# levers (as the H100 preset): boot 1 starts vLLM on 127.0.0.1:8001, waits for its /health, lists
# the autotune file it wrote, stops it (SIGTERM to its process group, SIGKILL after 120 s), then
# boot 2 starts on 0.0.0.0:8000 with the caches boot 1 left on local disk. There is no server
# process in front of vLLM here: readiness is the platform probing docker_server's
# readiness_endpoint (/health) on server_port 8000, which nothing listens on until boot 2 binds it,
# so the replica takes traffic only after boot 2 (the same refused-connection state as any cold
# start's first ~400 s). Liveness probes the same /health; before first readiness the start is
# bounded by startup_threshold_seconds (3000 s), and two boots fit it (cold engine init 400-418 s
# plus a warm start). A background poller logs `diag restart: boot2 ready` when :8000/health
# answers. Measures in-container warm init, not the BDN restore or pod scheduling.
DIAG_RESTART="${NEMOTRON_DIAG_RESTART_ONCE:-0}"
[[ "${DIAG_RESTART}" =~ ^[01]$ ]] || { echo "NEMOTRON_DIAG_RESTART_ONCE must be 0 or 1, got '${DIAG_RESTART}'" >&2; exit 1; }
DIAG_BOOT1_TIMEOUT_S="${NEMOTRON_DIAG_BOOT1_TIMEOUT_S:-1800}"
SERVE_PORT=8000
# NEMOTRON_WEIGHT_PREFETCH: off (default) loads by mmap as main does; early reads the served
# checkpoint into the page cache in parallel while vLLM imports (prefetch_weights.py), and
# fails the start if the checkpoint cannot fit or a read fails; vllm passes
# --safetensors-load-strategy=prefetch (vLLM's own, one reader per shard, from load time).
PREFETCH="${NEMOTRON_WEIGHT_PREFETCH:-off}"
PREFETCH_THREADS="${NEMOTRON_WEIGHT_PREFETCH_THREADS:-16}"
if [[ ! "${PREFETCH_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NEMOTRON_WEIGHT_PREFETCH_THREADS must be a positive integer, got '${PREFETCH_THREADS}'" >&2
    exit 1
fi
LOAD_ARGS=()
case "${PREFETCH}" in
    off) ;;
    vllm) LOAD_ARGS=(--safetensors-load-strategy prefetch) ;;
    early)
        PREFETCH_PY="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/prefetch_weights.py"
        # Synchronous: missing shards or too little memory stop the start here.
        python3 -I "${PREFETCH_PY}" check "${BASETEN_MODEL_PATH}"
        # $$ becomes vllm after the exec below; a failed read stops it.
        python3 -I "${PREFETCH_PY}" read "${BASETEN_MODEL_PATH}" --threads "${PREFETCH_THREADS}" --kill-on-error $$ &
        ;;
    *) echo "NEMOTRON_WEIGHT_PREFETCH must be off, early or vllm, got '${PREFETCH}'" >&2; exit 1 ;;
esac
echo "startup levers: flashinfer_autotune=${AUTOTUNE} weight_prefetch=${PREFETCH} autotune_reuse=${REUSE} diag_restart_once=${DIAG_RESTART}" >&2

# config.json names pretrained_llm as the relative path "llm_backbone".
cd "${BASETEN_MODEL_PATH}"

# Card flags for vLLM 0.32.0: ReplaySSM with the FlashInfer Mamba backend and
# prefix caching off. Deviation from the card: thinking off by default (clients
# that omit chat_template_kwargs get a transcript). NeMo Speech main registers
# the current architecture name, so no --hf-overrides. MoE backend: vLLM's
# auto selection.
VLLM_ARGS=(
    --tokenizer "${BASETEN_MODEL_PATH}"
    --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B
    --trust-remote-code
    --dtype bfloat16
    --gpu-memory-utilization 0.94
    --max-model-len 65536
    --max-num-seqs "${NEMOTRON_MAX_NUM_SEQS:-384}"
    --max-num-batched-tokens "${NEMOTRON_MAX_NUM_BATCHED_TOKENS:-16384}"
    --enable-chunked-prefill
    --no-enable-prefix-caching
    --use-replayssm
    --mamba-backend flashinfer
    --mamba-cache-mode none
    --mm-processor-cache-gb 0
    --no-async-scheduling
    --disable-uvicorn-access-log
    --limit-mm-per-prompt '{"audio":{"count":1,"length":120000000}}'
    --speculative-config '{"method":"mtp","num_speculative_tokens":8}'
    --default-chat-template-kwargs '{"enable_thinking":false}'
    --reasoning-parser nemotron_v3
    --trust-request-mm-kwargs
    --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware
    "${AUTOTUNE_ARGS[@]}"
    "${LOAD_ARGS[@]}"
)

if [[ "${DIAG_RESTART}" == 1 ]]; then
    DIAG_PORT=$((SERVE_PORT + 1))
    apply_reuse
    echo "diag restart: boot1 start (port ${DIAG_PORT}, reuse_hook=${NEMOTRON_AUTOTUNE_REUSE_HOOK-off})" >&2
    setsid "${VENV}/bin/vllm" serve "${BASETEN_MODEL_PATH}" --host 127.0.0.1 --port "${DIAG_PORT}" "${VLLM_ARGS[@]}" &
    BOOT1=$!
    BOOT1_T0=${SECONDS}
    until python3 -I -c "import sys, urllib.request; sys.exit(urllib.request.urlopen(sys.argv[1], timeout=5).status != 200)" "http://127.0.0.1:${DIAG_PORT}/health" 2>/dev/null; do
        if ! kill -0 "${BOOT1}" 2>/dev/null; then
            wait "${BOOT1}" || true
            echo "diag restart: boot1 exited before /health" >&2
            exit 1
        fi
        if (( SECONDS - BOOT1_T0 > DIAG_BOOT1_TIMEOUT_S )); then
            echo "diag restart: boot1 not healthy after ${DIAG_BOOT1_TIMEOUT_S}s" >&2
            kill -KILL -- "-${BOOT1}" 2>/dev/null || true
            exit 1
        fi
        sleep 5
    done
    echo "diag restart: boot1 ready after $((SECONDS - BOOT1_T0))s" >&2
    shopt -s nullglob
    AUTOTUNE_FILES=("${AUTOTUNE_ROOT}"/*/*/*/autotune_configs_dp0_rank0.json)
    shopt -u nullglob
    echo "diag restart: autotune files after boot1: ${AUTOTUNE_FILES[*]:-none}" >&2
    kill -TERM -- "-${BOOT1}" 2>/dev/null || true
    for _ in $(seq 120); do
        kill -0 -- "-${BOOT1}" 2>/dev/null || break
        sleep 1
    done
    if kill -0 -- "-${BOOT1}" 2>/dev/null; then
        echo "diag restart: boot1 still running after 120 s, SIGKILL" >&2
        kill -KILL -- "-${BOOT1}" 2>/dev/null || true
        sleep 5
    fi
    wait "${BOOT1}" 2>/dev/null || true
    echo "diag restart: boot1 stopped" >&2
    apply_reuse
    echo "diag restart: boot2 start (reuse_hook=${NEMOTRON_AUTOTUNE_REUSE_HOOK-off})" >&2
    # Readiness is the platform's probe of :8000/health; log when boot 2 answers it.
    python3 -I -c "import sys, time, urllib.request
for _ in range(int(sys.argv[2]) // 5):
    try:
        if urllib.request.urlopen(sys.argv[1], timeout=5).status == 200:
            print('diag restart: boot2 ready', file=sys.stderr, flush=True)
            break
    except Exception:
        pass
    time.sleep(5)" "http://127.0.0.1:${SERVE_PORT}/health" "${DIAG_BOOT1_TIMEOUT_S}" &
else
    apply_reuse
fi
exec "${VENV}/bin/vllm" serve "${BASETEN_MODEL_PATH}" --host 0.0.0.0 --port "${SERVE_PORT}" "${VLLM_ARGS[@]}"
