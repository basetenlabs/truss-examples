# vLLM Truss: Deploy a Chat Completion Model

## Overview

This Truss example offers a **codeless, OpenAI-compatible solution** to run a vLLM server within a Truss container. With minimal configuration, you can deploy powerful language models on our cloud—just update your settings and Truss will handle the rest.

---

## Configuration Guide

All deployment options are controlled via the `config.yaml` file. Follow the instructions below based on your GPU requirements:

### 🚀 Basic: Single GPU Deployment

To deploy a model using a single GPU, simply modify the following parameters in `config.yaml`:
- `model_name`
- `repo_id`
- `accelerator`

No additional changes are required.

---

### 🖥️ Advanced: Multi-GPU Deployment (Tensor Parallelism)

If your model requires multiple GPUs, such as for tensor parallelism, you’ll need to configure:

- `accelerator`  
  Example for 4 H100 GPUs:  
  ```yaml
  accelerator: H100:4
  ```
- `tensor_parallel_size`
- `distributed_executor_backend`

These last two are arguments for the `vllm serve` command within `config.yaml`. Add to the command as follows: `--tensor-parallel-size 4 --distributed-executor-backend mp`