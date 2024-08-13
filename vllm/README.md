# vLLM Truss

## What is this Truss example doing

This is a generic [Truss](https://truss.baseten.co/) that can deploy an asynchronous vLLM engine([AsyncLLMEngine](https://docs.vllm.ai/en/latest/dev/engine/async_llm_engine.html#asyncllmengine)) of any customized configuration with [all compatible models](https://docs.vllm.ai/en/latest/models/supported_models.html). We create this example to give you the most codeless experience, so you can configure all vLLM engine parameters in `config.yaml`, without making code changes in `model.py` for most of the use cases.

## Configure your Truss by modifying the config.yaml

### Basic options using 1 GPU

Here is the minimum config file you will need to deploy a model using vLLM on 1 GPU.
The only parameters you need to touch are:
- `model_name`
- `repo_id`
- `accelerator`

```
model_name: "Llama 3.1 8B Instruct VLLM"
python_version: py311
model_metadata:
  example_model_input: {"prompt": "what is the meaning of life"}
  repo_id: meta-llama/Meta-Llama-3.1-8B-Instruct
  openai_compatible: true
  vllm_config: null
requirements:
  - vllm==0.5.4
resources:
  accelerator: A100
  use_gpu: true
runtime:
  predict_concurrency: 128
secrets:
  hf_access_token: null
```

### Basic options using multiple GPUs

If your model needs more than 1 GPU to run using tensor parallel, you will need to change `accelerator`, and to set `tensor_parallel_size` and `distributed_executor_backend` accordingly.

```
model_name: "Llama 3.1 8B Instruct VLLM"
python_version: py311
model_metadata:
  example_model_input: {"prompt": "what is the meaning of life"}
  repo_id: meta-llama/Meta-Llama-3.1-8B-Instruct
  openai_compatible: false
  vllm_config:
    tensor_parallel_size: 4
    max_model_len: 4096
    distributed_executor_backend: mp
requirements:
  - vllm==0.5.4
resources:
  accelerator: A10G:4
  use_gpu: true
runtime:
  predict_concurrency: 128
secrets:
  hf_access_token: null
```

### Use vLLM's OpenAI compatible server

To use vLLM in [OpenAI compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) mode, simply set `openai_compatible: true` under `model_metadata`.

### Customize vLLM engine parameters

For advanced users who want to override [vLLM engine arguments](https://docs.vllm.ai/en/latest/models/engine_args.html), you can add all arguments to `vllm_config`.

#### Example 1: using model quantization

```
model_name: Mistral 7B v2 vLLM AWQ - T4
environment_variables: {}
external_package_dirs: []
model_metadata:
  repo_id: TheBloke/Mistral-7B-Instruct-v0.2-AWQ
  vllm_config:
    quantization: "awq"
    dtype: "float16"
    max_model_len: 8000
    max_num_seqs: 8
python_version: py310
requirements:
  - vllm==0.5.4
resources:
  accelerator: T4
  use_gpu: true
secrets:
  hf_access_token: null
system_packages: []
runtime:
  predict_concurrency: 128
```

#### Example 2: using customized vLLM image

You can even override with your own customized vLLM docker image to work with models that are not supported yet by vanilla vLLM.

```
model_name: Ultravox v0.2
base_image:
  image: vshulman/vllm-openai-fixie:latest
  python_executable_path: /usr/bin/python3
model_metadata:
  repo_id: fixie-ai/ultravox-v0.2
  vllm_config:
    audio_token_id: 128002
environment_variables: {}
external_package_dirs: []
python_version: py310
runtime:
  predict_concurrency: 512
requirements:
  - httpx
resources:
  accelerator: A100
  use_gpu: true
secrets:
  hf_access_token: null
system_packages:
- python3.10-venv
```

## Deploy your Truss

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`
3. With `vllm` as your working directory, you can deploy the model with:

    ```sh
    truss push --trusted
    ```

    Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Call your model

Once your deployment is up, there are [many ways](https://docs.baseten.co/invoke/quickstart) to call your model.

### curl command

#### If you are NOT using OpenAI compatible server

```
curl -X POST https://model-<YOUR_MODEL_ID>.api.baseten.co/development/predict \
     -H "Authorization: Api-Key $BASETEN_API_KEY" \
     -d '{"prompt": "what is the meaning of life"}'
```


#### If you are using OpenAI compatible server

```
curl -X POST "https://model-<YOUR_MODEL_ID>.api.baseten.co/development/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {BASETEN_API_KEY}' \
     -d '{
           "messages": [{"role": "user", "content": "What even is AGI?"}],
           "max_tokens": 256
         }'
```

To access [production metrics](https://docs.vllm.ai/en/latest/serving/metrics.html) out of OpenAI compatible server, simply add `metrics: true` to the request.

```
curl -X POST "https://model-<YOUR_MODEL_ID>.api.baseten.co/development/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {BASETEN_API_KEY}' \
     -d '{
           "metrics": true
         }'
```

### OpenAI SDK (if you are using OpenAI compatible server)

```
from openai import OpenAI
import os

model_id = "a2345678" # Replace with your model ID

client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url=f"https://bridge.baseten.co/{model_id}/v1/direct"
)

response = client.chat.completions.create(
  model="meta-llama/Meta-Llama-3.1-8B-Instruct",
  messages=[
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)
print(response.choices[0].message.content)

```

For more information, see [API reference](https://docs.baseten.co/api-reference/openai).

## Support

If you have any questions or need assistance, please open an issue in this repository or contact our support team.
