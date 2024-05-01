# DBRX Truss

This truss makes the [DBRX](https://huggingface.co/databricks/dbrx-instruct) model available on the Baseten platform for efficient inference. DBRX is an open-source large language model trained by Databricks. It is a 132B parameter model capable of instruction following and general language tasks.

## Setup

This truss requires Python 3.11 and the dependencies listed in `requirements.txt`. It is configured to run on A10G GPUs for optimal performance.

## Usage

Once deployed on Baseten, the truss exposes an endpoint for making prediction requests to the model.

### Request Format

Requests should be made with a JSON payload in the following format:

```json
{
  "prompt": "What is machine learning?"
}
```

### Parameters

The following inference parameters can be configured in `config.yaml`:

- `max_new_tokens`: Max number of tokens to generate in the response (default: 100)
- `temperature`: Controls randomness of output (default: 0.7)
- `top_p`: Nucleus sampling probability threshold (default: 0.95)
- `top_k`: Number of highest probability vocabulary tokens to keep (default: 50)
- `repetition_penalty`: Penalty for repeated tokens (default: 1.01)

## Original Model

DBRX was developed and open-sourced by Databricks. For more information, see:

- [DBRX Model Card](https://github.com/databricks/dbrx/blob/master/MODEL_CARD_dbrx_instruct.md)
- [Databricks Blog Post](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)
- [HuggingFace Model Page](https://huggingface.co/databricks/dbrx-instruct)

## About Baseten

This truss was created by [Baseten](https://www.baseten.co/) to enable easy deployment and serving of the open-source DBRX model at scale. Baseten is a platform for building powerful AI apps.
