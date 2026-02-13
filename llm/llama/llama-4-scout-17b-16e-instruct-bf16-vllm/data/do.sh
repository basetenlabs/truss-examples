#!/bin/bash
HF_TOKEN=$(cat /secrets/hf_access_token) vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct --served-model-name llama --max-model-len 131072 --tensor-parallel-size 4 --distributed-executor-backend mp --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --limit-mm-per-prompt image=10 --override-generation-config='{"attn_temperature_tuning": true}'
