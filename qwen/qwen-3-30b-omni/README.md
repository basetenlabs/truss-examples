# Qwen3-Omni-30B-Thinker (vLLM) on Baseten

This is a [Truss](https://truss.baseten.co/) for serving **Qwen/Qwen3-Omni-30B-A3B-Intruct** with **transformers** on Baseten. It exposes an endpoint that accepts **text, image, audio, and video** inputs and returns low-latency text and audio responses. 

**Why this deployment**

* *Multimodal, end-to-end*: text, images, audio, and video inputs in a single chat request; text and spoken audio responses.
* *Production-ready on Baseten*: autoscaling, logs, metrics, and zero-downtime publishes.

---

# Overview

**Model**: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
**Modalities**:

* **Input**: text, image(s), audio, video
* **Output**: text, audio

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
# ✨ Model Qwen3 Omni 30B Instruct was successfully pushed ✨
```

---

# Call your model

Replace `model-xxxxxx` and include your Baseten API key.

## API schema (Chat Completions)

**POST** `https://model-xxxxxx.api.baseten.co/development/predict`

**Request example**:

```json
{
  "speaker": "Chelsie",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Hello how are you?"}
      ]
    }
  ]
}
```

**Response (truncated example)**:

```json
{
  "text": "Hello! I'm doing well, thank you. How can I assist you today?",
  "audio": "UklGRs5hAwBXQVZFZm10IBAAAAABAAEAwF0AAIC7AAACABAAZGF0YaphAwAE..."
}
```

### curl

```bash
curl -X POST https://model-xxxxxx.api.baseten.co/development/predict \
  -H "Authorization: Api-Key YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "speaker":"Aiden",
    "messages":[
      {"role":"system","content":"You are a helpful assistant."},
      {"role":"user","content":[
        {"type":"text","text":"Describe this image and audio content."},
        {"type":"image","image":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"},
        {"type":"audio","audio":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"}
      ]}
    ]
  }'
```

### requests (Python)

```python
import os, requests, json

url = "https://model-xxxxxx.api.baseten.co/development/predict"
headers = {
    "Authorization": "Api-Key " + os.environ["BASETEN_API_KEY"],
    "Content-Type": "application/json"
}
payload = {
    "speaker": "Ethan",
    "messages": [{"role":"user","content":[{"type":"text","text":"Briefly describe the image."},
        {"type":"image","image":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"}]}]
}
print(requests.post(url, headers=headers, data=json.dumps(payload)).json())
```

---

**Notes**

* The container image `qwenllm/qwen3-omni:3-cu124` bundles flash attention and dependencies.

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
      {"type":"image","image":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"},
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
      {"type":"audio","audio":"https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"},
      {"type":"text","text":"What do you hear?"}
    ]}
  ]
}
```

---

# Support

If you have questions or need help, open an issue in this repository or contact Baseten support.
