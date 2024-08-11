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

### Customize vLLM engine parameters

For advanced users who want to override [vLLM engine arguments](https://docs.vllm.ai/en/latest/models/engine_args.html), you can add all arguments to `vllm_config`.

#### Customized vLLM config example 1: prefix caching

#### Customized vLLM config example 2: model quantization

### Use vLLM's OpenAI compatible server

To use vLLM in [OpenAI compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) mode, simply set `openai_compatible: true` under `model_metadata`.

### Base Image

todo: You can use any vLLM compatible base image.

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
           "model": "meta-llama/Meta-Llama-3.1-8B-Instruct"
           "messages": [{"role": "user", "content": "What even is AGI?"}],
           "max_tokens": 256
         }'
```

### OpenAI SDK

#### If you are NOT using OpenAI compatible server
todo: verify the code snippet correctness

```
from openai import OpenAI
import os

model_id = "abcd1234" # Replace with your model ID
deployment_id = "4321cbda" # [Optional] Replace with your deployment ID

client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url=f"https://bridge.baseten.co/v1/direct"
)

response = client.chat.completions.create(
  model=f"baseten/{model_id}/{deployment_id}",
  messages=[
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)

print(response.choices[0].message.content)

```

#### If you are using OpenAI compatible server
todo: verify the code snippet correctness

```
from openai import OpenAI
import os

model_id = "abcd1234" # Replace with your model ID
deployment_id = "4321cbda" # [Optional] Replace with your deployment ID

client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url=f"https://bridge.baseten.co/v1/direct"
)

response = client.chat.completions.create(
  model=f"baseten/{model_id}/{deployment_id}",
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
