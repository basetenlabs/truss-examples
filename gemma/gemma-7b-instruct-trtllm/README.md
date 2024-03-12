# Gemma 7B Instruct

At Baseten, we care a lot about model performance. On a fundamental level, if you're building with LLMs, the latecy of your responses really matters. Years of UX research have shown that even slight speed increases yield much happier users. GPUs are also really, really expensive so improving their efficiency and utilization translates to less spent per request.

We're open-sourcing this optimized implementation of Gemma 7B and explaining how we optimize models.

[insert graph of benchmarks, compared to other guys]

The cool thing is that this is a standalone example that you can run on your own GPUs (or deploy to Baseten if you'd like this + industry-exceeding reliability and infrastructure).

## Usage

If you just want to run this model, you have two options:
1. Running on Baseten
2. Running outside of Baseten

Running this on Baseten is pretty straightforward.
1. Update the `hf_access_token` secret inside the `config.yaml`. We need this to download the correct tokenizer. Please make sure you've been given access to the `google/gemma-7b-it` [repository on Hugging Face](https://huggingface.co/google/gemma-7b-it).
2. `pip install truss`
3. Inside this directory, just run `truss push --publish ./` and Baseten will build and deploy this truss into an endpoint you can use from anywhere.

If you'd like this to run outside of Baseten, follow the same steps as above except for the last step. You'll want to produce the underlying Docker container that this Truss creates which you can do by running the following inside this directory.

```
truss image build
```

## Strategies for making models run faster

You've probably heard that language models are "memory-bound". What this really means is that the performance of these models is often limited more by memory bandwidth of the GPU than by raw computational power. In simpler terms, the model spends more time waiting for data to be moved around (to and from memory) than it does on actual computations like matrix multiplications. This is particularly true for autoregressive models (including LLMs like Gemma), which generate one token at a time and thus have a sequential dependency in their operations.

### Strategy 1: Quantization

Quantizing the weights reduces the size of the model, which is useful in making those memory read/writes take less time. Certain GPUs also have dedicated hardware for computing matrix multiplcations for certain data types faster which is useful.

### Strategy 2: Memory-level optimizations

There's an entire class of optimizations around optimizing the memory read/writes on the GPU. For example, kernel fusion works by taking multiple, common operations (like a matmul followed by some reduction) and fusing them into a single kernel. This means that instead of reading the matrix from memory, computing some operation, writing the result back to memory, then reading that result back into memory to perform the reduction, the entire multiplcation and reduction happen in one forward pass.

Outside of the benefits of reducing memory read/writes, kernel fusion also reduces the overhead of invoking multiple kernels. When using PyTorch, for example, the Python API will invoke a CUDA kernel on a GPU. However, each invocation comes with some overhead, where data needs to be transferred from CPU to GPU which is not ideal. Kernel fusion reduces this overhead as well.

Other examples of memory-level optimizations are [Flash Attention](https://arxiv.org/abs/2205.14135) and [PagedAttention](https://arxiv.org/abs/2309.06180).

### Strategy 3: Batching

If we're forced to load in each layer for each token, we may as well try to run as many concurrent requests per computation to balance the ratio of memory boundedness to compute boundedness. One of the key ideas here is continous batching presneted in the [Orca](https://www.usenix.org/system/files/osdi22-yu.pdf) paper. As requests come in sporadically, we can run a batcher at each time-step of the LLM runtime and identify requests within the batch that have completed. If they've completed, we can replace those rows with requests that we've just recieved.

### Strategy 4: Parallelism

By splitting the model or the data across multiple processors, you can process more data at the same time, reducing the overall time required. One implementation we've seen be quite useful is tensor parallelism (TP). With TP, we split a tensor (or matrix) into `N` parts across some dimension and place each part on a particular GPU. Because we can compute an `all-gather` operation across the GPUs, we can process an input in parallel across the GPUs and then communicate our results to some master node. The catch is that you want fast intra-GPU bandwidth as the system will run this `all-gather` at each layer of your model.

## Making Gemma fast

This particular implementation uses [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/). We converted the engine into a TensorRT-LLM binary with `INT8` AWQ and then ran the following command to generate the engine:

```
trtllm-build --checkpoint_dir
    ./int8-gemma-weights/
    --gemm_plugin bfloat16
    --gpt_attention_plugin bfloat16
    --max_batch_size 64
    --max_input_len 3000
    --max_output_len 1000
    --context_fmha enable
    --output_dir ./engines
```

We use optimized GPT attention implementations and utilize optimized `GEMM` kernels. On an 80GB A100, we can fit up to 64 concurrent requests that at 3k input, 1k requested outputs. We host the engine binary [here](https://huggingface.co/baseten/gemma-7b-it-trtllm-3k-1k-64bs).

## Serving Gemma

As part of releasing this engine, we're also releasing our high-performance Triton [truss](https://truss.baseten.co/). [Triton Inference Server](https://github.com/triton-inference-server/tensorrtllm_backend) is a high-performance model server built by Nvidia. The TensorRT-LLM team provides support for a Triton backend that we can use to serve this engine. The backend leverages the C++ runtime that comes with TensorRT-LLM (vs. the Python runtime) which we've seen is usually faster and has support for continous batching.

The catch is that Triton Inference Server mostly operates on protobuf inputs and they're not the cleanest to interact with. A lot of this Truss is built to provide a simple JSON interface for consumers of this service while minimizing the performance overhead of the proxy server and maximizing TPS. We have a couple core utilities to help with this, namely the `TritonServer` and `TritonClient` classes.


- The `TritonSever` class helps manage the lifecycle of the underlying Triton Server instance that's actually running the optimized engine binary. We provide helpers to `start()` and `stop()` the server as well as helpful properties if the instance is `alive` and/or `ready` for inferencing.
- The `TritonClient` class contains all the logic for managing GRPC inference streams. The Triton Server provides async GRPC streams as an interface which we use between the proxy server (defined in the `model.py`) and the underlying server. We also manage the process of converting from JSON request params to the appropriate protobuf that Triton expects.

The `model.py` is the entrypoint to this truss. When a truss starts, we first invoke the code in `load` and then for each request, invoke the `predict` method. In our `load`, we instantiate the Triton server with the correct engine and then proxy requests over GRPC when requests come in.

## Extensibility

When using this truss, you may see that the `config.yaml` provides information such as the `engine_repository` and `tokenizer_repository`. _This truss is compatabile with all GPT-style TensorRT-LLM engines!_. This means that if there's an engine for a model you're interested in that lives on the Baseten Huggingface repository [here](https://huggingface.co/baseten), you can swap out the `engine_repository` and `tokenizer_repository` fields and run this truss. One think worth noting is that each engine is built for a specific GPU so you'll also want to pay attention to the `resources` field and update accordingly.
