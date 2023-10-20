# WizardLM Truss w/ vLLM

This repository packages [WizardLM](https://github.com/nlpxucan/WizardLM) as a [Truss](https://trussml.com) using vLLM as the model server.

WizardLM is a instruction-following LLM tuned using the Evol-Instruct method. Evol-Instruct is a novel method using LLMs instead of humans to automatically mass-produce open-domain instructions of various difficulty levels and skills range, to improve the performance of LLMs.

## Deploying WizardLM

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `wizardlm-vllm` as your working directory, you can deploy the model with:

```
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Hardware notes

We found this model runs reasonably fast on A10Gs; hardware is configured as follows in `config.yaml`:

```yaml
...
resources:
  cpu: "3"
  memory: 14Gi
  use_gpu: true
  accelerator: A10G
...
```

## Invoking WizardLM

Once the model is deployed, you can invoke it with:

```sh
truss predict -d '{"prompt": "What is the difference between a wizard and a sorcerer?"}'
```

You can also invoke your model via a REST API

```
curl -X POST "https://app.baseten.co/models/YOUR_MODEL_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "What is the difference between a wizard and a sorcerer?",
           "temperature": 0.3
         }'
```

The usual GPT-style parameters will pass right through to the inference point:

* max_new_tokens (_default_: 64)
* temperature (_default_: 0.5)
* top_p (_default_: 0.9)
* top_k (_default_: 0)
* num_beams (_default_: 4)