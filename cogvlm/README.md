# CogVLM Truss

This repository packages [CogVLM](https://github.com/THUDM/CogVLM) as a [Truss](https://truss.baseten.co/).

CogVLM is a highly performant open-source vision language model with capabilities similar to GPT-4V.

## Deploying CogVLM

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd cogvlm
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `cogvlm` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Invoking CogVLM

CogVLM takes in two inputs, a `query` and a base64 encoded `image`. CogVLM will respond to the `query` conditioned on the `image`. The output is a JSON blob with a single key, `result`, that answers the `query`.


```sh
truss predict -d '{"query": "Describe this picture in detail.", "image": "data:image/png;base64,iVBORw0KGgoA..."}'
```

You can also invoke your model via a REST API
```
curl -X POST https://app.baseten.co/model_versions/<YOUR_MODEL_VERSION_ID>/predict \
     -H "Content-Type: application/json" \
     -d '{
           "query": "Describe this picture in detail.",
           "image": "data:image/png;base64,iVBORw0KGgoA..."
         }'
```
