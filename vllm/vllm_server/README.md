# vLLM Truss to deploy chat completion model

## What is this Truss example doing

This is a general purpose [Truss](https://truss.baseten.co/) that can deploy an asynchronous vLLM engine([AsyncLLMEngine](https://docs.vllm.ai/en/latest/dev/engine/async_llm_engine.html#asyncllmengine)) of any customized configuration with [all compatible chat completion models](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Configure your Truss by modifying the config.yaml

### Basic options using 1 GPU

To deploy a model using 1 GPU, the only config parameters you need to change are:
- `model_name`
- `repo_id`
- `accelerator`

### Basic options using multiple GPUs

If your model needs more than 1 GPU to run using tensor parallel, you will need to change `accelerator`, and to set `tensor_parallel_size` and `distributed_executor_backend` accordingly.

`tensor_parallel_size` and `distributed_executor_backend` are each arguments for the vllm serve command in the `config.yaml`.

If you are using 4 GPUs for inference for example, you need to add these arguments to the `vllm serve` command.

`--tensor-parallel-size 4 --distributed-executor-backend mp`

### Other ways to customize

See this [doc](https://docs.vllm.ai/en/v0.7.2/serving/engine_args.html) for all the ways you can customize the `vllm serve` command. These parameters give you control over the level of compilation, quantization, and much more.