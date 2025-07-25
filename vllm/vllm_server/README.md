# vLLM Truss to deploy chat completion model

## What is this Truss example doing

This is a codeless, easy OpenAI compatible solution to run a vllm server in a truss. Run a vllm server simply by modifying configurations, we'll handle the rest.

## Configure your Truss by modifying the config.yaml

### Basic options using 1 GPU

To deploy a model using 1 GPU, the only config parameters you need to change are:
- `model_name`
- `repo_id`
- `accelerator`

### Basic options using multiple GPUs

If your model needs more than 1 GPU to run using tensor parallel, you will need to change `accelerator`, and to set `tensor_parallel_size` and `distributed_executor_backend` accordingly.

`tensor_parallel_size` and `distributed_executor_backend` are each arguments for the vllm serve command in the `config.yaml`.

If you are using 4 GPUs for inference for example, you need to add the arguments `--tensor-parallel-size 4 --distributed-executor-backend mp` to the `vllm serve` command as well as indicating this new quantity under `accelerator: H100:4`.

### Customize the vLLM server

This container starts by calling the `vllm serve` command under `start_command` in `config.yaml`.

See this [doc](https://docs.vllm.ai/en/v0.7.2/serving/engine_args.html) for all the ways you can customize the `vllm serve` command. These parameters give you control over the level of compilation, quantization, and much more.