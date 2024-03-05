

# TRTLLM

### Overview
This Truss adds support for TRT-LLM engines via Triton Inference Server. TRT-LLM is a highly-performant language model runtime. We leverage the C++ runtime to take advantage of in-flight batching (aka continous batching).

### Prerequisites

To use this Truss, your engine must be built with in-flight batching support. Refer to your architecture-specific `build.py` re: how to build with in-flight-batching support.

### Config

This Truss is primarily config driven. This means that most settings you'll need to edit are located in the `config.yaml`. These settings are all located underneath the `model_metadata` key.

- `tensor_parallelism` (int): If you built your model with tensor parallelism support, you'll need to set this value with the same value used during the build engine step. This value should be the same as the number of GPUs in the `resources` section.

*Pipeline parallelism is not supported in this version but will be added later. As noted from Nvidia, pipeline parallelism reduces the need for high-bandwidth communication but may incur load-balancing issues and may be less efficient in terms of GPU utilization.*

- `engine_repository` (str): We expect engines to be uploaded to Huggingface with a flat directory structure (i.e the engine and associated files are not underneath a folder structure). This value is the full `{org_name}/{repo_name}` string. Engines can be private or public.

- `tokenizer_repository` (str): Engines do not come bundled with their own tokenizer. This is the Huggingface repository where we can find a tokenizer. Tokenizers can be private or public.

If the engine and repository tokenizers are private, you'll need to update the `secrets` section of the `config.yaml` as follows:

```
secrets:
 hf_access_token: "my_hf_api_key"
```

### Performance

TRT-LLM engines are designed to be highly performant. Once your Truss has been deployed, you may find that you're not fully utilizing the GPU. The following are levers to improve performance but require trial-and-error to identify appropriates. All of these values live inside the `config.pbtxt` for a given ensemble model.

#### Preprocessing / Postprocessing

```
instance_group [
    {
        count: 1
        kind: KIND_CPU
    }
]
```
By default, we load 1 instance of the pre/post models. If you find that the tokenizer is a bottleneck, increasing the `count` variable here will load more replicas of these models and Triton will automatically load balance across model instances.

### Tensorrt LLM
```
parameters: {
  key: "max_tokens_in_paged_kv_cache"
  value: {
    string_value: "10000"
  }
}
```
By default, we set the `max_tokens_in_paged_kv_cache` to 10000. For a 13B model on 1 A100 with a batch size of 8, we have over 60GB of GPU memory left over. We can increase this value to 100k comfortably and allow for more tokens in the KV cache. Your mileage will vary based on the size of your model and the hardware you're running on.

```
parameters: {
  key: "kv_cache_free_gpu_mem_fraction"
  value: {
    string_value: "0.1"
  }
}
```
By default, if `max_tokens_in_paged_kv_cache` is unset, Triton Inference Server will attempt to preallocate `kv_cache_free_gpu_mem_fraction` fraction of free gpu memory for the KV cache.

```
parameters: {
  key: "max_num_sequences"
  value: {
    string_value: "64"
  }
}
```
The `max_num_sequences` param is the maximum numbers of requests that the inference server can maintain state for at a given time (state = KV cache + decoder state).
See this [comment](https://github.com/NVIDIA/TensorRT-LLM/issues/65#issuecomment-1774332446) for more details. Setting this value higher allows for more parallel processing but uses more GPU memory.

### API

We expect requests will the following information:


- ```prompt``` (str): The prompt you'd like to complete
- ```max_tokens``` (int, default: 50): The max token count. This includes the number of tokens in your prompt so if this value is less than your prompt, you'll just recieve a truncated version of the prompt.
- ```beam_width``` (int, default:50): The number of beams to compute. This must be 1 for this version of TRT-LLM. Inflight-batching does not support beams > 1.
- ```bad_words_list``` (list, default:[]): A list of words to not include in generated output.
- ```stop_words_list``` (list, default:[]): A list of words to stop generation upon encountering.
- ```repetition_penalty``` (float, defualt: 1.0): A repetition penalty to incentivize not repeating tokens.

This Truss will stream responses back. Responses will be buffered chunks of text.
