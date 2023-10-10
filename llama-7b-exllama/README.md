# ExLlama V2 Truss

This is a [Truss](https://truss.baseten.co/) for models supported by ExLlama V2, including Mistral, Llama 2, and more. The Truss supports `.safetensors`, `GPTQ`, and `EXL2` file formats, whcih means you should be able to run many of the models you love, as long as they have a Llama-style architecture.

## Benchmarks

Using ExLlama V2, we found that we were able to:
- run 7B parameter models on a T4 at 40 tok/s
- run 7B parameter models on an A10G at 80 tok/s
- run 7B parameter models on an A100 at 100 tok/s
- run 70B parameter models on a **single** A100 at 30 tok/s

Note: for all four tests, we used Llama 2 quantized to 4 bits via GPTQ.

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp)). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.


## Deployment

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd llama-2-7b-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `llama-2-7b-truss` as your working directory, you can deploy the model with:

```sh
truss push --trusted
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Hardware notes

You can select different hardware based on the quantization of the model itself. For example, if you're running a Llama 2 7B model in half-precision, you will need to pick a GPU with 14 GB of VRAM. This leaves you with deciding between an A100, which has 80 GB of VRAM, and a A10G, which has 24 GB of VRAM. If you quantize Llama 2 7B to 4 bits per weight, you can drop to a GPU with just 7 GB of VRAM. You can run this quantized model comfortably on a T4 (which has 16 GB of VRAM).


## Example usage

```sh
truss predict -d '{"prompt": "What is the meaning of life?", "max_new_tokens": 4096}'
```

You can also invoke your model via a REST API:

```
curl -X POST " https://app.baseten.co/model_versions/YOUR_MODEL_VERSION_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "What's the meaning of life?",
           "max_new_tokens": 4096
         }'
```