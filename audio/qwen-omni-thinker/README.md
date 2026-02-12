# Qwen3 Omni 30B Instruct (Thinker Only)

Deploy Qwen3 Omni 30B Instruct (Thinker Only) on Baseten using a vLLM engine.

| Property | Value |
|----------|-------|
| Model | [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) |
| Task | Audio |
| Engine | vLLM |
| GPU | H100 |
| OpenAI compatible | Yes |

## Deploy

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/v1/chat/completions \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Describe this image and audio content."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"
          }
        },
        {
          "type": "audio_url",
          "audio_url": {
            "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"
          }
        },
        {
          "type": "text",
          "text": "What can you see and hear? Answer in one sentence."
        }
      ]
    }
  ],
  "stream": false,
  "model": "qwen3-omni",
  "max_tokens": 2048,
  "temperature": 0.7
}'
```

## Configuration highlights

- Base image: `qwenllm/qwen3-omni:3-cu124`
- Predict concurrency: **32**
