# PaddleOCR-VL

Deploy [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) using a custom server configuration on Baseten.

| Property | Value |
|----------|-------|
| Model | [PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) |
| Task | Infrastructure / Custom server |
| Engine | vLLM |
| GPU | H100_40GB |
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
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png"
          }
        },
        {
          "type": "text",
          "text": "OCR:"
        }
      ]
    }
  ],
  "model": "PaddlePaddle/PaddleOCR-VL",
  "max_tokens": 4096,
  "temperature": 0.0
}'
```

## Configuration highlights

- Base image: `public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:0bf29fadf5f8b28817fbccb037fb70adaef3f7f1`
- Predict concurrency: **128**
