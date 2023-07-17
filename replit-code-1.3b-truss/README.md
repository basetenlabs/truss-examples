# Replic Code 1.3B Truss

This repository packages [Replit Code 1.3B](https://huggingface.co/replit/replit-code-v1-3b) as a [Truss](https://truss.baseten.co).

Replit Code 1.3B is an LLM released by Replit, optimized and trained for generating code autocompletions. 

## Deploying Replit Code 1.3B

We found this model runs reasonably fast on A10Gs; you can configure the hardware you'd like in the config.yaml.

```yaml
...
resources:
  cpu: "3"
  memory: 14Gi
  use_gpu: true
  accelerator: A10G
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

replit_code_truss = truss.load('.')
baseten.deploy(replit_code_truss)
```

## Invoking Replit-1.3B

The usual GPT-style parameters will pass right through to the inference point:

* max_new_tokens (_default_: 64)
* temperature (_default_: 0.5)
* top_p (_default_: 0.9)
* top_k (_default_: 0)
* num_beams (_default_: 4)
* do_sample (_default_: False)

Note that we recommend setting `do_sample` to `True` for best results, and
increasing the `max_new_tokens` parameter to 200-300.


```python
import baseten
model = baseten.deployed_model_id('YOUR MODEL ID')
model.predict({"prompt": "def fib(n):", "do_sample": True, "max_new_tokens": 300})
```

You can also invoke your model via a REST API

```
curl -X POST " https://app.baseten.co/models/YOUR_MODEL_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "def fib(n):",
           "do_sample": True,
           "max_new_tokens": 300,
         }'
```
