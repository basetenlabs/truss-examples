# Qwen VL Truss

This is a [Truss](https://truss.baseten.co/) for [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL) which is a visual language model. Qwen is a family of models developed by Alibaba Cloud. This LLM supports both English and Chinese.

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.


## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd qwen-vl
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `qwen-vl` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).


## Qwen-VL API documentation

This section provides an overview of the Qwen-VL model, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided prompt.

### API route: `predict`

The predict route is the primary method for generating text completions based on a given prompt. It takes several parameters:

- __prompt__: The instruction the model will follow to extract the data from the image.
- __image__ : The input image in the form of a URL or a base64 string.


## Example usage

```python
import requests

data = {
  "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
  "prompt": "Generate the caption in English with grounding"
}

res = requests.post(
    "https://model-<model-id>.api.baseten.co/development/predict",
    headers={"Authorization": "Api-Key <BASETEN-API-KEY>"},
    json=data,
)

print(res.json())
```

## Example Output

```json
{"output": "Picture 1: <img>/tmp/tmpw6m_zmbk.png</img>\nGenerate the caption in English with grounding<ref> A maltese dog</ref><box>(385,361),(783,934)</box> in a flower garden<|endoftext|>"}
```
