# Qwen3-ASR-1.7B

Deploy Qwen3-ASR-1.7B on Baseten using a vLLM engine.

| Property | Value |
|----------|-------|
| Model | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| Task | Audio |
| Engine | vLLM |
| GPU | H100_40GB:1 |
| OpenAI compatible | Yes |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/v1/chat/completions \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "stream": false,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "audio_url",
          "audio_url": {
            "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"
          }
        }
      ]
    }
  ]
}'
```

## Configuration highlights

- Base image: `vllm/vllm-openai:nightly-070c811d6f74c55302557878f5982411a3346b4d`
- Predict concurrency: **256**
- System packages: `python3.10-venv, ffmpeg, openmpi-bin, libopenmpi-dev`
