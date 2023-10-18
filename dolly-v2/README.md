## Dolly-v2 Truss

This is a [Truss](https://truss.baseten.co/) for Dolly-v2, an instruction-following large language model based on `pythia-12b` . This README will walk you through how to deploy this Truss on Baseten to get your own instance of Dolly-v2.

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.

## Deploying Dolly-v2

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd dolly-v2-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `dolly-v2-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

#### Example Usage

```sh
truss predict -d '{"prompt": "Explain to me the difference between nuclear fission and fusion."}'
```

You can also invoke your model via a REST API
```
curl -X POST https://app.baseten.co/model_versions/<YOUR_MODEL_VERSION_ID>/predict \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "Explain to me the difference between nuclear fission and fusion."
         }'
```
