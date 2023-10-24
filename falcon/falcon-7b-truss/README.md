# Falcon-7B Truss

This repository packages [Falcon-7B](https://huggingface.co/tiiuae/falcon-7b) as a [Truss](https://truss.baseten.co).

Falcon 7B is an LLM released by Technology Innovation Institute (TII).

Utilizing this model for inference can be challenging given the hardware requirements. With Baseten and Truss, inference is dead simple.

## Deploying Falcon-7B

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd falcon-7b-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `falcon-7b-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Hardware notes

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

## Invoking Falcon-7B

The usual GPT-style parameters will pass right through to the inference point:

* max_new_tokens (_default_: 64)
* temperature (_default_: 0.5)
* top_p (_default_: 0.9)
* top_k (_default_: 0)
* num_beams (_default_: 4)
* do_sample (_default_: False)

Note that we recommend setting `do_sample` to `True` for best results, and
increasing the `max_new_tokens` parameter to 200-300.


```sh
truss predict -d '{"prompt": "Write a movie plot about falcons planning to over the world", "do_sample": True, "max_new_tokens": 300}'
```

You can also invoke your model via a REST API

```
curl -X POST " https://app.baseten.co/models/YOUR_MODEL_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "Write a movie plot about falcons planning to over the world",
           "do_sample": True,
           "max_new_tokens": 300,
           "temperature": 0.3
         }'
```
