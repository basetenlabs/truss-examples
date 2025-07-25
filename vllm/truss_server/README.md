# vLLM Truss: Deploy Chat Completion Models

## Overview

This repository demonstrates how to deploy [vLLM](https://github.com/vllm-project/vllm) using a Truss server.  
**Use this approach only if you need custom inference logic or flexibility.**  
For most users, we recommend the easier [vLLM server example](https://github.com/basetenlabs/truss-examples/tree/main/vllm/vllm_server), which is also OpenAI-compatible.

This Truss works with asynchronous vLLM engines ([AsyncLLMEngine](https://docs.vllm.ai/en/v0.6.5/dev/engine/async_llm_engine.html#asyncllmengine)) and [all supported chat completion models](https://docs.vllm.ai/en/latest/models/supported_models.html).

---

## Configure Your Truss (`config.yaml`)

### Single GPU Example

To deploy on a single GPU, update these fields:
- `model_name`
- `repo_id`
- `accelerator`

<details>
<summary>Minimal config example</summary>

```yaml
model_name: "Llama 3.1 8B Instruct VLLM"
python_version: py311
model_metadata:
  example_model_input: {"prompt": "what is the meaning of life"}
  repo_id: meta-llama/Llama-3.1-8B-Instruct
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
</details>

---

### Multi-GPU Example (Tensor Parallelism)

For multi-GPU deployments, set:
- `accelerator` (e.g., `A10G:4`)
- `model_metadata.vllm_config.tensor_parallel_size`
- `model_metadata.vllm_config.distributed_executor_backend`

<details>
<summary>Multi-GPU config example</summary>

```yaml
model_name: "Llama 3.1 8B Instruct VLLM"
python_version: py311
model_metadata:
  example_model_input: {"prompt": "what is the meaning of life"}
  repo_id: meta-llama/Llama-3.1-8B-Instruct
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
</details>

---

### Custom vLLM Engine Parameters

Override any [vLLM engine argument](https://docs.vllm.ai/en/latest/models/engine_args.html) by adding it to `vllm_config` in `model_metadata`.

#### Example: Model Quantization

<details>
<summary></summary>

```yaml
model_name: Mistral 7B v2 vLLM AWQ - T4
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
runtime:
  predict_concurrency: 128
secrets:
  hf_access_token: null
```
</details>

#### Example: Custom Docker Image

<details>
<summary></summary>

```yaml
model_name: Ultravox v0.2
base_image:
  image: vshulman/vllm-openai-fixie:latest
  python_executable_path: /usr/bin/python3
model_metadata:
  repo_id: fixie-ai/ultravox-v0.2
  vllm_config:
    audio_token_id: 128002
python_version: py310
requirements:
  - httpx
resources:
  accelerator: A100
  use_gpu: true
runtime:
  predict_concurrency: 512
secrets:
  hf_access_token: null
system_packages:
  - python3.10-venv
```
</details>

---

## Deploy Your Truss

1. [Sign up for Baseten](https://app.baseten.co/signup) and get an [API key](https://app.baseten.co/settings/account/api_keys).
2. Install Truss:  
   ```sh
   pip install --upgrade truss
   ```
3. Deploy your model from the `vllm` directory:
   ```sh
   truss push
   ```
   Enter your API key when prompted.

[Truss documentation →](https://truss.baseten.co)

---

## Call Your Model

After deploying, invoke your model via [many methods](https://docs.baseten.co/invoke/quickstart).

### Curl: Not OpenAI Compatible

```sh
curl -X POST https://model-<YOUR_MODEL_ID>.api.baseten.co/development/predict \
     -H "Authorization: Api-Key $BASETEN_API_KEY" \
     -d '{"prompt": "what is the meaning of life"}'
```

### Curl: OpenAI Compatible

```sh
curl -X POST "https://model-<YOUR_MODEL_ID>.api.baseten.co/development/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {BASETEN_API_KEY}' \
     -d '{
           "messages": [{"role": "user", "content": "What even is AGI?"}],
           "max_tokens": 256
         }'
```

**Production Metrics:**  
Add `"metrics": true` to your request for detailed metrics:

```sh
curl -X POST "https://model-<YOUR_MODEL_ID>.api.baseten.co/development/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {BASETEN_API_KEY}' \
     -d '{"metrics": true}'
```

---

### OpenAI SDK (OpenAI-Compatible Only)

```python
from openai import OpenAI
import os

model_id = "abcd1234"  # Replace with your model ID
deployment_id = "4321cbda"  # [Optional]

client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url=f"https://bridge.baseten.co/{model_id}/v1/direct"
)

response = client.chat.completions.create(
  model="meta-llama/Llama-3.1-8B-Instruct",
  messages=[
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ],
  extra_body={
    "baseten": {
      "model_id": model_id,
      "deployment_id": deployment_id
    }
  }
)
print(response.choices[0].message.content)
```

[API Reference →](https://docs.baseten.co/api-reference/openai)

---

## Support

Need help?  
Open an issue in this repository or [contact Baseten support](https://www.baseten.co/contact).
