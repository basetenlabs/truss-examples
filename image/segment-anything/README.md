# Segment Anything

Deploy Segment Anything for image segmentation on Baseten.

| Property | Value |
|----------|-------|
| Task | Image segmentation |
| Engine | Custom (Truss) |
| GPU | A10G |
| Python | py310 |

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
  "image_url": "https://as2.ftcdn.net/v2/jpg/00/66/26/87/1000_F_66268784_jccdcfdpf2vmq5X8raYA8JQT0sziZ1H9.jpg"
}'
```

## Configuration highlights

- System packages: `python3-opencv`
