# Stable Diffusion XL with LoRA Swapping

This Truss provides an example on how to hot-swap LoRAs with Stable Diffusion XL. This is useful when you have a bunch of customers with different fine-tunes but you want to use the same GPU instance.

On Baseten, you can expect LoRA downloading + loading to take ~4s for a 120 MB LoRA and ~2s for a standard 20 MB LoRA. Generation time will be ~6s on an A100.

## Deploying the Truss

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd sdxl-lora-swapping
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `sdxl-lora-swapping` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Using the model

The model takes a JSON payload with two main fields:

- `prompt`: Text describing the desired image. Make sure the prompt includes key phrases to activate the LoRA if needed. For example, many pixel style LoRAs require the words "pixel style" to be present in the prompt.
- `lora`: Dict with two keys, `repo_id` and `weights`. An example `lora` dict would like this:

```lora.json
{
    "prompt": "pixel art, an baby giraffe",
    "lora": {"repo_id": "nerijs/pixel-art-xl", "weights": "pixel-art-xl.safetensors"}
}
```

There are many ojptional fields including negative prompt and size that you can find in the `predict` function of the `model.py`.

It returns a JSON object with the `result` field containing the generated image.

## Example Usage

You can use the Truss CLI to generate an image:

```
truss predict --model PRIMARY_MODEL_ID -f lora.json
```

Here, `lora.json` should contain the data payload for the request.

You can also invoke the model via REST API:

```bash
curl -X POST "https://app.baseten.co/model_versions/VERSION_ID/predict" \
     -H "Content-Type: application/json" \
     -H "Authorization: Api-Key {API_KEY}" \
     -d '{"prompt": "pixel art, painting of a cat"}'
```

The API will return a JSON response containing the generated image encoded in base64.
