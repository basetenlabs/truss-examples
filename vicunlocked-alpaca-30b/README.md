# Vicunlocked-Alpaca-30B Truss

This repository packages [Vicunlocked-Alpaca-30B](https://huggingface.co/Aeala/VicUnlocked-alpaca-30b) as a [Truss](https://truss.baseten.co).

Utilizing this model for inference can be challenging given the hardware requirements. With Baseten and Truss, inference is dead simple.

## Deploying Vicunlocked-Alpaca-30B

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

Before deployment:

1. Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup).
2. Install Truss and the Baseten Python client: `pip install --upgrade baseten truss`
3. Authenticate your development environment with `baseten login`

Deploying the Truss is easy; simply load it and push from a Python script:

```python
import baseten
import truss

vicunlocked_truss = truss.load('.')
baseten.deploy(vicunlocked_truss)
```

## Invoking Vicunlocked-Alpaca-30B

The usual GPT-style parameters will pass right through to the inference point:

```python
import baseten
model = baseten.deployed_model_id('YOUR MODEL ID')
model.predict({"prompt": "Write a movie plot about vicunas planning to over the world", "do_sample": True, "max_new_tokens": 300})
```

You can also invoke your model via a REST API

```
curl -X POST " https://app.baseten.co/models/YOUR_MODEL_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "Write a movie plot about vicunlockeds planning to over the world",
           "do_sample": True,
           "max_new_tokens": 300,
           "temperature": 0.3
         }'
```

