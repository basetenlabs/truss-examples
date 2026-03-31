# LLaVA v1.5 Truss

This repository packages [LLaVA 1.5](https://github.com/haotian-liu/LLaVA/) as a [Truss](https://truss.baseten.co/).

LLaVA (Large Language and Vision Assistant) is a highly performant open-source vision language model with capabilities similar to GPT-4V.

## Deploying LLaVA

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd llava/llava-v1.5-7b
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `llava-v1.5-7b` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Invoking LLaVA

LLaVA takes in two inputs, a `query` and a base64 encoded `image`. LLaVA will respond to the `query` conditioned on the `image`. The output is a JSON blob with a single key, `result`, that answers the `query`.


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
