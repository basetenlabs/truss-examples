# Qwen3-Omni-30B-Thinker (vLLM) on Baseten

This is a [Truss](https://truss.baseten.co/) for serving the "thinker" part of **Qwen/Qwen3-Omni-30B-A3B-Intruct** with **vLLM** on Baseten. It exposes an **OpenAI-compatible /v1/chat/completions** endpoint that accepts **text, image, audio, and video** inputs and returns low-latency text responses.

**Why this deployment**

* *Multimodal, end-to-end*: text, images, audio, and video inputs in a single chat request; text responses today (audio TTS output is not compatible with vLLM at the moment).
* *OpenAI-compatible API*: drop-in with `openai` libraries and many tools.
* *Production-ready on Baseten*: autoscaling, logs, metrics, and zero-downtime publishes.

---

# Overview

**Model**: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
**Modalities**:

* **Input**: text, image(s), audio, video
* **Output**: text

---

# Deploy with Truss

Before deployment:

1. Create a [Baseten account](https://app.baseten.co/signup) and an [API key](https://app.baseten.co/settings/account/api_keys).
2. Install Truss: `pip install --upgrade truss`

Clone the examples repo (or your project) and cd into your working directory:

```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd qwen/qwen-3-30b-omni-thinker
```

Publish:

```sh
truss push --publish
# ✨ Model Qwen3 Omni 30B Instruct (Thinker Only) was successfully pushed ✨
```

---

# Call your model

Your deployment is OpenAI-compatible. Replace `model-xxxxxx` and include your Baseten API key.

## API schema (Chat Completions)

**POST** `https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/chat/completions`

**Request (multimodal example)**:

```json
{
  "model": "qwen3-omni",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe what you see and hear."},
        {
          "type": "image_url",
          "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"}
        },
        {
          "type": "audio_url",
          "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"}
        }
      ]
    }
  ],
  "max_tokens": 2048,
  "temperature": 0.7,
  "stream": false
}
```

**Response (truncated example)**:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "qwen3-omni",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "I see several parked cars in front of a building and hear a short cough."
      }
    }
  ],
  "usage": {
    "prompt_tokens": 512,
    "completion_tokens": 24,
    "total_tokens": 536
  }
}
```

### curl

```bash
curl -X POST https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/chat/completions \
  -H "Authorization: Api-Key YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"qwen3-omni",
    "messages":[
      {"role":"system","content":"You are a helpful assistant."},
      {"role":"user","content":[
        {"type":"text","text":"Describe this image and audio content."},
        {"type":"image_url","image_url":{"url":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"}},
        {"type":"audio_url","audio_url":{"url":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"}}
      ]}
    ],
    "max_tokens":2048,
    "temperature":0.7,
    "stream":false
  }'
```

### OpenAI Python SDK

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ["BASETEN_API_KEY"],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync/v1"
)

resp = client.chat.completions.create(
    model="qwen3-omni",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": "Describe this image and audio content."},
            {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"}},
            {"type": "audio_url", "audio_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"}}
        ]}
    ],
    max_tokens=2048,
    temperature=0.7,
    stream=False,
)
print(resp.choices[0].message.content)
```

### Streaming (server-sent events)

```python
stream = client.chat.completions.create(
    model="qwen3-omni",
    messages=[{"role":"user","content":"Summarize this short clip."}],
    stream=True
)
for event in stream:
    if event.choices and event.choices[0].delta:
        print(event.choices[0].delta.content or "", end="", flush=True)
```

### requests (Python)

```python
import os, requests, json

url = "https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/chat/completions"
headers = {
    "Authorization": "Api-Key " + os.environ["BASETEN_API_KEY"],
    "Content-Type": "application/json"
}
payload = {
    "model": "qwen3-omni",
    "messages": [{"role":"user","content":[{"type":"text","text":"Briefly describe the image."},
        {"type":"image_url","image_url":{"url":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"}}]}],
    "max_tokens": 512
}
print(requests.post(url, headers=headers, data=json.dumps(payload)).json())
```

> The OpenAPI schema is available at:
> `https://model-xxxxxx.api.baseten.co/environments/production/sync/openapi.json`

---

# Advanced usage notes

* **Images, audio, video**: Provide remote URLs via `image_url` / `audio_url` / `video_url`, or configure `--allowed-local-media-path` to allow local file ingestion.
* **Batching/throughput**: Configure Baseten `predict_concurrency` for request-level concurrency. vLLM will also batch internally.
* **Long contexts**: `--max-model-len 65,536` in the server command; adjust based on memory.
* **Multi-GPU**: Add `-tp N` (tensor parallelism) in the vLLM command to shard across GPUs.
* **Audio output**: If you require generated speech, verify your vLLM build and model variant support it; otherwise pipe text into a TTS stage.

---

**Notes**

* The container image `qwenllm/qwen3-omni:3-cu124` bundles vLLM and dependencies.
* For **multi-GPU** boxes, add `-tp <num_gpus>` to the `vllm serve` command.

---

# Examples

### 1) Pure text

```json
{
  "model":"qwen3-omni",
  "messages":[{"role":"user","content":"Give me three creative app ideas for teachers."}],
  "max_tokens":512
}
```

### 2) Image + instruction

```json
{
  "model":"qwen3-omni",
  "messages":[
    {"role":"user","content":[
      {"type":"image_url","image_url":{"url":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"}},
      {"type":"text","text":"Describe the scene in one sentence."}
    ]}
  ]
}
```

### 3) Audio + question

```json
{
  "model":"qwen3-omni",
  "messages":[
    {"role":"user","content":[
      {"type":"audio_url","audio_url":{"url":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"}},
      {"type":"text","text":"What do you hear?"}
    ]}
  ]
}
```

---

# Support

If you have questions or need help, open an issue in this repository or contact Baseten support.
