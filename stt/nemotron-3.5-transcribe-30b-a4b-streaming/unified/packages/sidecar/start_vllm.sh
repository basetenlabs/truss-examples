#!/usr/bin/env bash
# The engine sidecar for the unified streaming preset. nemotron_stt's build() launches this and
# waits for its /health before returning the backend, so baseten-stt (the image's Python 3.12)
# never imports NeMo or vLLM: those live in their own Python 3.13 venv at /opt/nemo-speech,
# built as the batch preset's (stt/nemotron-3.5-transcribe-30b-a4b/rtx-pro-6000/batch). Same
# weights and NVIDIA model-card flags as that preset's data/start.sh, minus the CTC timestamp
# middleware (turns are plain text); it binds to localhost only, since nothing outside the
# container talks to it.
set -euo pipefail
: "${BASETEN_MODEL_PATH:?BASETEN_MODEL_PATH must name the checkpoint directory}"
: "${NEMOTRON_SIDECAR_PORT:=8000}"

VENV=/opt/nemo-speech/venv
# The slim server image has no CUDA toolkit. FlashInfer JIT-compiles its ReplaySSM module at
# start with nvcc and includes nvrtc.h (fused-MoE and sampling come prebuilt from
# flashinfer-jit-cache), so point it at the CUDA 13.0 wheels in the venv (nvcc,
# crt, nvrtc, runtime all unpack under nvidia/cu13). The venv's bin/ also has the ninja
# FlashInfer runs its JIT build with.
export CUDA_HOME="${VENV}/lib/python3.13/site-packages/nvidia/cu13"
# The server image's interpreter (baseten-stt, nemotron_stt), before the venv's bin/ shadows python3;
# the diagnostic boot-1 warmup runs with it.
SERVER_PYTHON="$(command -v python3)"
export PATH="${VENV}/bin:${CUDA_HOME}/bin:${PATH}"

# As the batch preset: the BDN startup cache (ORG_ENABLE_BDN_STARTUP_CACHE) sets the JIT cache
# paths (VLLM_CACHE_ROOT, TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR, FLASHINFER_WORKSPACE_BASE,
# ...) and restores them across replicas and restarts, keyed by GPU type.
# Never point them at /cache/org: on that shared mount the engine sat idle after "Warming up
# Mamba2" until the startup check killed it (#1077).

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
# NEMOTRON_DIAG_WARMUP_PROFILE: 0 (default) off. 1 is a diagnostic for the STT engine warmup: the
# hook's sitecustomize registers a SIGUSR2 stack dump in every vLLM process (so this directory goes
# on PYTHONPATH even with reuse off), nemotron_stt runs the warmup twice and profiles each turn
# (see nemotron_stt/engine.py _WarmupProfiler), and with NEMOTRON_DIAG_RESTART_ONCE=1 boot 1 runs the
# profiled warmup too before it is stopped, so boot 2's first pass shows what on-disk caches fix.
DIAG_PROFILE="${NEMOTRON_DIAG_WARMUP_PROFILE:-0}"
[[ "${DIAG_PROFILE}" =~ ^[01]$ ]] || { echo "NEMOTRON_DIAG_WARMUP_PROFILE must be 0 or 1, got '${DIAG_PROFILE}'" >&2; exit 1; }
apply_diag_profile() {
    [[ "${DIAG_PROFILE}" == 1 ]] || return 0
    case ":${PYTHONPATH-}:" in
        *":${REUSE_HOOK_DIR}:"*) ;;
        *) export PYTHONPATH="${REUSE_HOOK_DIR}${PYTHONPATH:+:${PYTHONPATH}}" ;;
    esac
}
# NEMOTRON_DIAG_RESTART_ONCE: 0 (default) starts vLLM once. 1 is a diagnostic for warm-start
# levers (as the H100 preset): boot 1 starts vLLM on a private port (NEMOTRON_SIDECAR_PORT + 1,
# 8001; nemotron_stt's /health poll only watches NEMOTRON_SIDECAR_PORT, so build() does not return
# and /health/ready stays unready), waits for its /health, lists the autotune file it wrote, stops
# it (SIGTERM to its process group, SIGKILL after 120 s), then boot 2 starts on the real port with
# the caches boot 1 left on local disk. Only boot 2 can make build() return, so the replica
# reports ready after boot 2. Measures in-container warm init, not the BDN restore or pod
# scheduling. Two boots fit SIDECAR_STARTUP_TIMEOUT_S (2700 s) and startup_threshold_seconds
# (3000 s): a cold RTX engine init measured 315-328 s.
DIAG_RESTART="${NEMOTRON_DIAG_RESTART_ONCE:-0}"
[[ "${DIAG_RESTART}" =~ ^[01]$ ]] || { echo "NEMOTRON_DIAG_RESTART_ONCE must be 0 or 1, got '${DIAG_RESTART}'" >&2; exit 1; }
DIAG_BOOT1_TIMEOUT_S="${NEMOTRON_DIAG_BOOT1_TIMEOUT_S:-1800}"
SERVE_PORT="${NEMOTRON_SIDECAR_PORT}"
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
# NEMOTRON_ATTENTION_BACKEND: unset or "" (config.yaml default) lets vLLM pick, which on RTX PRO 6000 (sm120) is
# FLASH_ATTN with FlashAttention 2. vLLM 0.32's _vllm_fa2_C ships sm_80 SASS plus compute_80 PTX
# only, so every FA2 kernel variant is PTX-JIT-compiled by the driver at first use (exec-13: the
# 1 s and 8 s warmup turns spent 27.6 s and 53.3 s in the MTP draft's flash_attn_varlen_func,
# engine CPU 100%, GPU 0%, driver cache +19.2 / +31.5 MiB). A name (TRITON_ATTN, FLASHINFER) passes
# --attention-backend for the backbone. exec-14 (FLASHINFER, TRITON_ATTN): that alone leaves the MTP
# draft on FA2, because vLLM 0.32 never lets the draft inherit the target's backend
# (v1/spec_decode/llm_base_proposer.py _create_draft_vllm_config sets the draft's
# attention_config.backend to speculative_config.attention_backend, None = auto-select), and the
# warmup stall stayed (pass1 75.9 / 74.2 s). So a name also sets "attention_backend" in
# --speculative-config. NEMOTRON_DRAFT_ATTENTION_BACKEND overrides the draft's backend separately:
# unset or "" follows NEMOTRON_ATTENTION_BACKEND, "auto" lets vLLM pick, or a backend name.
ATTN_BACKEND="${NEMOTRON_ATTENTION_BACKEND-}"
ATTN_ARGS=()
case "${ATTN_BACKEND}" in
    "") ;;
    TRITON_ATTN|FLASHINFER|FLASH_ATTN|FLEX_ATTENTION) ATTN_ARGS=(--attention-backend "${ATTN_BACKEND}") ;;
    *) echo "NEMOTRON_ATTENTION_BACKEND must be empty, TRITON_ATTN, FLASHINFER, FLASH_ATTN or FLEX_ATTENTION, got '${ATTN_BACKEND}'" >&2; exit 1 ;;
esac
DRAFT_ATTN_BACKEND="${NEMOTRON_DRAFT_ATTENTION_BACKEND:-${ATTN_BACKEND}}"
SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":8}'
case "${DRAFT_ATTN_BACKEND}" in
    ""|auto) DRAFT_ATTN_BACKEND="" ;;
    TRITON_ATTN|FLASHINFER|FLASH_ATTN|FLEX_ATTENTION)
        SPEC_CONFIG="${SPEC_CONFIG%\}},\"attention_backend\":\"${DRAFT_ATTN_BACKEND}\"}" ;;
    *) echo "NEMOTRON_DRAFT_ATTENTION_BACKEND must be empty, auto, TRITON_ATTN, FLASHINFER, FLASH_ATTN or FLEX_ATTENTION, got '${DRAFT_ATTN_BACKEND}'" >&2; exit 1 ;;
esac
echo "startup levers: attention_backend=${ATTN_BACKEND:-auto} draft_attention_backend=${DRAFT_ATTN_BACKEND:-auto} flashinfer_autotune=${AUTOTUNE} weight_prefetch=${PREFETCH} autotune_reuse=${REUSE} diag_restart_once=${DIAG_RESTART} diag_warmup_profile=${DIAG_PROFILE}" >&2
apply_diag_profile

# config.json names pretrained_llm as the relative path "llm_backbone".
cd "${BASETEN_MODEL_PATH}"

# The batch preset's card flags for vLLM 0.32.0: ReplaySSM with the FlashInfer Mamba backend
# and prefix caching off. Deviation from the card: thinking off by default (the backend also
# sends it off). NeMo Speech main registers the current architecture name, so no
# --hf-overrides.
VLLM_ARGS=(
    --tokenizer "${BASETEN_MODEL_PATH}"
    --served-model-name nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B
    --trust-remote-code
    --dtype bfloat16
    --gpu-memory-utilization 0.94
    --max-model-len 65536
    --max-num-seqs 192
    --max-num-batched-tokens 4096
    --enable-chunked-prefill
    --no-enable-prefix-caching
    --use-replayssm
    --mamba-backend flashinfer
    --mamba-cache-mode none
    --mm-processor-cache-gb 0
    --no-async-scheduling
    --disable-uvicorn-access-log
    --limit-mm-per-prompt '{"audio":{"count":1,"length":120000000}}'
    --speculative-config "${SPEC_CONFIG}"
    --default-chat-template-kwargs '{"enable_thinking":false}'
    --reasoning-parser nemotron_v3
    "${AUTOTUNE_ARGS[@]}"
    "${ATTN_ARGS[@]}"
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
    if [[ "${DIAG_PROFILE}" == 1 ]]; then
        # The STT warmup against boot 1, profiled, from the server's interpreter (nemotron_stt is
        # on its PYTHONPATH). Diagnostic only: a failure is logged and boot 2 still starts.
        "${SERVER_PYTHON}" -c 'import sys; from nemotron_stt.engine import diag_warmup; diag_warmup(int(sys.argv[1]), int(sys.argv[2]), "boot1")' \
            "${DIAG_PORT}" "${BOOT1}" || echo "diag restart: boot1 warmup profile failed" >&2
    fi
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
else
    apply_reuse
fi
exec "${VENV}/bin/vllm" serve "${BASETEN_MODEL_PATH}" --host 127.0.0.1 --port "${SERVE_PORT}" "${VLLM_ARGS[@]}"
