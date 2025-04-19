#!/bin/bash
HF_TOKEN=$(cat /secrets/hf_access_token) vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 --served-model-name llama --max-model-len 340000 --tensor-parallel-size 8 --distributed-executor-backend mp --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --limit-mm-per-prompt image=10 --override-generation-config='{"attn_temperature_tuning": true}'
