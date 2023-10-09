# Vicunlocked-Alpaca-30B Truss

This repository packages [Vicunlocked-Alpaca-30B](https://huggingface.co/Aeala/VicUnlocked-alpaca-30b) as a [Truss](https://truss.baseten.co).

Utilizing this model for inference can be challenging given the hardware requirements. With Baseten and Truss, inference is dead simple.

## Deploying Vicunlocked-Alpaca-30B

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `vicunlocked-alpaca-30b` as your working directory, you can deploy the model with:

```
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Hardware notes

We found this model runs reasonably fast on A100s; you can configure the hardware you'd like in the config.yaml.

```yaml
...
resources:
  cpu: "3"
  memory: 14Gi
  use_gpu: true
  accelerator: A100
...
```

## Invoking Vicunlocked-Alpaca-30B

The usual GPT-style parameters will pass right through to the inference point:

```python
truss predict -d '{"prompt": "Write a movie plot about vicunas planning to over the world", "do_sample": True, "max_new_tokens": 300}'
```

You can also invoke your model via a REST API

```
curl -X POST "https://app.baseten.co/models/YOUR_MODEL_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "Write a movie plot about vicunas planning to over the world",
           "do_sample": True,
           "max_new_tokens": 300,
           "temperature": 0.3
         }'
```
