# IP Adapter

This is a [Truss](https://truss.baseten.co/) for [IP Adapter](https://github.com/tencent-ailab/IP-Adapter). IP Adapter can create variations of a given input image based on a prompt, while retaining the aesthetic of the origina image.


## Deploying IP Adapter

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd ip_adapter
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `ip_adapter` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Invoking IP Adapter

IP Adapter takes in two inputs, an optional `prompt` and a base64 encoded `image`. The output is a JSON blob with a single key, `result` with another base64 encoded image.


```sh
truss predict -d '{"image": "data:image/png;base64,iVBORw0KGgoA..."}'
```

You can also invoke your model via a REST API
```
curl -X POST https://app.baseten.co/model_versions/<YOUR_MODEL_VERSION_ID>/predict \
     -H "Content-Type: application/json" \
     -d '{
           "image": "data:image/png;base64,iVBORw0KGgoA..."
         }'
```
