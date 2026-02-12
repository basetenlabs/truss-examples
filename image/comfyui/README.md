# ComfyUI Workflow

Deploy a ComfyUI workflow as a Truss

| Property | Value |
|----------|-------|
| Task | Image generation |
| Engine | Custom (Truss) |
| GPU | A10G |
| Python | py39 |

## Deploy

```sh
truss push
```

## Invoke

```sh
curl -X POST https://model-<model_id>.api.baseten.co/predict \
  -H "Authorization: Api-Key YOUR_BASETEN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
  "workflow_values": {
    "controlnet_image": "https://storage.googleapis.com/logos-bucket-01/baseten_logo.png",
    "negative_prompt": "blurry, text, low quality",
    "positive_prompt": "An igloo on a snowy day, 4k, hd"
  }
}'
```

## Configuration highlights

- Base image: `bolabaseten/comfyui-truss-base:6a7bc35`
- System packages: `ffmpeg, libgl1-mesa-glx`
