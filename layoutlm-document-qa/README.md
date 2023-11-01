 # LayoutLM Document QA Truss

This repository packages [LayoutLM Document QA](https://huggingface.co/impira/layoutlm-document-qa) as a [Truss](https://truss.baseten.co).

This multimodal model takes an image of an invoice (PNG or JPEG) and extracts information from it in response to natural language prompts.

## Deploying LayoutLM Document QA

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd layoutlm-document-qa-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `layoutlm-document-qa-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Invoking LayoutLM Document QA

LayoutLM takes a dictionary with:

* `url`: The URL of a PNG or JPEG of an invoice
* `prompt`: The question to ask of the invoice

Example invocation:

```sh
truss predict -d '{"url": "https://templates.invoicehome.com/invoice-template-us-neat-750px.png", "prompt": "What is the invoice number?"}'
```

Expected response:

```python
[{'answer': '9.06', 'end': 73, 'score': 0.9910207986831665, 'start': 73}]
```

You can also invoke your model via a REST API

```
curl -X POST "https://app.baseten.co/models/YOUR_MODEL_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "url": "https://templates.invoicehome.com/invoice-template-us-neat-750px.png",
           "prompt": "What is the invoice number?"
         }'
```

## Hardware notes

We found this model runs reasonably fast with 4 vCPUs and 16 GiB of RAM, no GPU needed. Invocation times are usually <10 seconds for the first prompt on an image (to account for image download times) then <3 seconds thereafter.

Default config:

```yaml
...
resources:
  cpu: "4"
  memory: 16Gi
  use_gpu: false
  accelerator: null
...
```
