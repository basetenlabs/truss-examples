from dataclasses import field
from pathlib import Path
from typing import Any

from pydantic import dataclasses
from truss.base.trt_llm_config import (
    CheckpointRepository,
    CheckpointSource,
    TRTLLMConfiguration,
    TrussTRTLLMBuildConfiguration,
    TrussTRTLLMModel,
    TrussTRTLLMQuantizationType,
)
from truss.base.truss_config import Accelerator, Resources, TrussConfig

REPO_URL = "https://github.com/basetenlabs/truss-examples"
SUBFOLDER = Path("11-embeddings-reranker-classification-tensorrt")
ROOT_NAME = Path(REPO_URL.split("/")[-1])


@dataclasses.dataclass
class Task:
    purpose: str
    client_usage: str
    model_identification: str
    model_metadata: dict[str, Any]


@dataclasses.dataclass
class Embedder(Task):
    purpose: str = (
        " is a text-embeddings model, producing a 1D embeddings vector, given an input.\n"
        "It's frequently used for downstream tasks like clustering, used with vector databases."
    )
    model_identification: str = (
        "Suitable models need to have the configurations of the `sentence-transformers` library, which are used for embeddings. "
        "Such repos contain e.g. a `sbert_config.json` or a `1_Pooling/config.json` file besides the fast-tokenizer and the safetensors file."
    )
    model_metadata: dict = field(
        default_factory=lambda: dict(
            example_model_input=dict(
                input="text string", encoding_format="float", model="model"
            )
        )
    )
    client_usage: str = r"""
### API-Schema:
POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/embeddings`
```json
{
  "encoding_format": "float", # or base64
  "input": "string", # can be list of strings for multiple embeddings
  "model": "null",
  "user": "null"
}
```

Returns:
```json
{
  "data": [
    {
      "embedding": [
        0
      ],
      "index": 0,
      "object": "embedding"
    }
  ],
  "model": "thenlper/gte-base",
  "object": "list",
  "usage": {
    "prompt_tokens": 512,
    "total_tokens": 512
  }
}
```
The OpenAPI.json is available under https://model-xxxxxx.api.baseten.co/environments/production/sync/openapi.json for more details.

#### Advanced:
You may also use Baseten's async jobs API, which returns a request_id, which you can use to query the status of the job and get the results.

POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/async/v1/embeddings`
Read more about [Baseten's Async API here](https://docs.baseten.co/invoke/async)

### curl
```bash
curl -X POST https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/embeddings \
        -H "Authorization: Api-Key YOUR_API_KEY" \
        -d '{"input": "text string", "model": "model"}'
```

### OpenAI compatible client library
```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ['BASETEN_API_KEY'],
    api_url="https://model-xxxxxx.api.baseten.co/environments/production/sync"
)

embedding = client.embeddings.create(
    input="Baseten Embeddings are fast",
    model="model"
)
```
### requests python library

```python
import os
import requests

resp = requests.post(
    "https://model-xxxxxx.api.baseten.co/environments/production/sync/v1/embeddings",
    headers={"Authorization": "Api-Key " + str(os.environ['BASETEN_API_KEY'])},
    json={"input": ["text string", "second string"]},
)

print(resp.json())
```
"""


@dataclasses.dataclass
class Reranker(Task):
    purpose: str = (
        " is a reranker model, used to re-rank a list of items, given a query. \\n"
        "It is frequently used in search engines, recommendation systems, and more."
    )
    model_identification: str = (
        "Suitable models can be identified by the `ForSequenceClassification` suffix in the model name. "
        "Reranker models may have at most one label, which contains the score of the reranking."
    )
    model_metadata: dict = field(
        default_factory=lambda: dict(
            example_model_input=dict(
                input="This redirects to the embedding endpoint. Use the /sync API to reach /rerank"
            )
        )
    )
    client_usage: str = r"""
### API-Schema:
POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/sync/rerank`:
```json
{
  "query": "What is Baseten?",
  "raw_scores": false,
  "return_text": false,
  "texts": [
    "Deep Learning is ...", "Baseten is a fast inference provider"
  ],
  "truncate": false,
  "truncation_direction": "right"
}
```

Returns:
```json
[
  {
    "index": 0,
    "score": 1,
    "text": "Deep Learning is ..."
  }
]
```
The OpenAPI.json is available under https://model-xxxxxx.api.baseten.co/environments/production/sync/openapi.json for more details.

#### Advanced:
You may also use Baseten's async jobs API, which returns a request_id, which you can use to query the status of the job and get the results.

POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/async/rerank`
Read more about [Baseten's Async API here](https://docs.baseten.co/invoke/async)

### OpenAI compatible client library
OpenAI.com does not have a rerank endpoint, therefore no client library is available.
"""


@dataclasses.dataclass
class Predictor(Task):
    purpose: str = (
        " is a text-classification model, used to classify a text into a category. \\n"
        "It is frequently used in sentiment analysis, spam detection, and more. It's also used for deployment of chat rating models, e.g. RLHF reward models or toxicity detection models."
    )
    model_identification: str = (
        "Suitable models can be identified by the `ForSequenceClassification` suffix in the model name. "
        "Prediction models may have one or more labels, which are returned with the prediction."
    )
    model_metadata: dict = field(
        default_factory=lambda: dict(
            example_model_input=dict(
                input="This redirects to the embedding endpoint. Use the /sync API to reach /sync/predict endpoint."
            )
        )
    )
    client_usage: str = r"""
### API-Schema:
POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/sync/predict`
```json
{
  "inputs": "Baseten is a fast inference provider",
  "raw_scores": false,
  "truncate": false,
  "truncation_direction": "right"
}
```

Returns:
```json
[
  {
    "label": "excitement",
    "score": 0.99
  }
]
```
Important, this is different from the `predict` route: https://model-xxxxxx.api.baseten.co/environments/production/predict
The OpenAPI.json is available under https://model-xxxxxx.api.baseten.co/environments/production/sync/openapi.json for more details.

#### Advanced:
You may also use Baseten's async jobs API, which returns a request_id, which you can use to query the status of the job and get the results.

POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/async/predict`
Read more about [Baseten's Async API here](https://docs.baseten.co/invoke/async)

### OpenAI compatible client library
OpenAI does not have a classification endpoint, therefore no client library is available.
"""


@dataclasses.dataclass
class Deployment:
    name: str
    hf_model_id: str
    accelerator: Accelerator
    task: Task
    is_gated: bool = False
    is_fp8: bool = False


def name_generator(dp: Deployment):
    return "BEI-" + dp.name.replace(" ", "-").replace("/", "-").lower()


def generate_bei_deployment(dp: Deployment):
    from transformers import AutoConfig

    root = Path(__file__).parent.parent.parent
    assert root.name == ROOT_NAME.name, "This script has been moved"
    folder_name = name_generator(dp)

    folder_relative_path = SUBFOLDER / folder_name
    full_folder_path = root / folder_relative_path
    model_nickname = folder_name + "-truss-example"
    is_gated = (
        "Note: [This is a gated/private model] Retrieve your Hugging Face token from the [settings](https://huggingface.co/settings/tokens). "
        "Set your Hugging Face token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_access_key`."
        if dp.is_gated
        else ""
    )

    hf_cfg = AutoConfig.from_pretrained(dp.hf_model_id)
    max_position_embeddings = hf_cfg.max_position_embeddings
    architecture = hf_cfg.architectures[0]

    max_num_tokens = max(16384, max_position_embeddings)
    quantization_disclaimer = (
        "\nThis model is quantized to FP8 for deployment, which is supported by Nvidia's newest GPUs e.g. H100, H100_40GB or L4. "
        "Quantization is optional, but leads to higher efficiency."
        if dp.is_fp8
        else ""
    )
    quantization_removal = "If you want to remove the quantization, remove the `quantization_type` field or set it to `no_quant` for float16."

    config = TrussConfig(
        model_metadata=dp.task.model_metadata,
        trt_llm=TRTLLMConfiguration(
            build=TrussTRTLLMBuildConfiguration(
                base_model=TrussTRTLLMModel.ENCODER,
                checkpoint_repository=CheckpointRepository(
                    repo=dp.hf_model_id,
                    revision="main",
                    source=CheckpointSource.HF,
                ),
                max_seq_len=1000001,
                max_num_tokens=max_num_tokens,
                **(
                    {
                        "quantization_type": TrussTRTLLMQuantizationType.FP8,
                        # give more resources / cpu ram + vram on build if the model uses not-mig
                        "num_builder_gpus": (
                            2
                            if dp.accelerator in [Accelerator.H100, Accelerator.L4]
                            else 1
                        ),
                    }
                    if dp.is_fp8
                    else {}
                ),
            )
        ),
        resources=Resources(
            accelerator=dp.accelerator,
            use_gpu=True,
            memory="8Gi",
        ),
        model_name=model_nickname,
    )

    # Writes
    full_folder_path.mkdir(parents=True, exist_ok=True)
    config.write_to_yaml_file(full_folder_path / "config.yaml", verbose=False)
    config_yaml_as_str = Path(full_folder_path / "config.yaml").read_text()

    README_SUBREPO = f"""# Baseten-Embeddings-Inference with {dp.name}

This is a Deployment for BEI (Baseten-Embeddings-Inference) with {dp.name}. BEI is Baseten's solution for production-grade deployments via TensorRT-LLM.

With BEI you get the following benefits:
- *Lowest-latency inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama)<sup>1</sup>
- *Highest-throughput inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama) - thanks to XQA kernels, FP8 and dynamic batching.<sup>2</sup>
- High parallelism: up to 1400 client embeddings per second
- Cached model weights for fast vertical scaling and high availability - no Hugging Face hub dependency at runtime

# Examples:
This deployment is specifically designed for the Hugging Face model [{dp.hf_model_id}](https://huggingface.co/{dp.hf_model_id}).
It will also work for fine-tuned models that have the architecture of {architecture} specified in their Hugging Face transformers config.
{dp.task.model_identification}

{dp.hf_model_id} {dp.task.purpose}
{quantization_disclaimer}

## Deployment with Truss

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`
{is_gated}

First, clone this repository:
```sh
git clone https://github.com/basetenlabs/truss-examples.git
cd {folder_relative_path.as_posix()}
```

With `{folder_relative_path.as_posix()}` as your working directory, you can deploy the model with the following command. Paste your Baseten API key if prompted.

```sh
truss push --publish
# prints:
# ✨ Model {model_nickname} was successfully pushed ✨
# 🪵  View logs for your deployment at https://app.baseten.co/models/yyyyyy/logs/xxxxxx
```

## Call your model
{dp.task.client_usage}

## Config.yaml
By default, the following configuration is used for this deployment. {quantization_removal}

```yaml
{config_yaml_as_str}
```

## Support
If you have any questions or need assistance, please open an issue in this repository or contact our support team.
"""
    (full_folder_path / "README.md").write_text(README_SUBREPO)


DEPLOYMENTS_BEI = [
    Deployment(
        "BAAI/bge-large-en-v1.5-embedding",
        "BAAI/bge-large-en-v1.5",
        Accelerator.L4,
        Embedder(),
    ),
    Deployment(
        "WhereIsAI/UAE-Large-V1-embedding",
        "WhereIsAI/UAE-Large-V1",
        Accelerator.L4,
        Embedder(),
    ),
    Deployment(
        "mixedbread-ai/mxbai-embed-large-v1-embedding",
        "mixedbread-ai/mxbai-embed-large-v1",
        Accelerator.L4,
        Embedder(),
    ),
    Deployment(
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        Accelerator.A100,  # Bert has long-context issues (>8K tokens on 24Gb Ram machines. Using 80B therefore)
        Embedder(),
    ),
    Deployment(
        "BAAI/bge-m3-embedding-dense",
        "BAAI/bge-m3",
        Accelerator.A100,  # Bert has long-context issues (>8K tokens on 24Gb Ram machines. Using 80B therefore)
        Embedder(),
    ),
    Deployment(
        "Linq-AI-Research/Linq-Embed-Mistral",
        "Linq-AI-Research/Linq-Embed-Mistral",
        Accelerator.H100_40GB,
        Embedder(),
        is_fp8=True,
    ),
    Deployment(
        "BAAI/bge-multilingual-gemma2-multilingual-embedding",
        "BAAI/bge-multilingual-gemma2",
        Accelerator.H100_40GB,
        Embedder(),
        # no fp8 support
    ),
    Deployment(
        "Salesforce/SFR-Embedding-Mistral",
        "Salesforce/SFR-Embedding-Mistral",
        Accelerator.H100_40GB,
        Embedder(),
        is_fp8=True,
    ),
    Deployment(
        "BAAI/bge-en-icl-embedding",
        "BAAI/bge-en-icl",
        Accelerator.H100_40GB,
        Embedder(),
        is_fp8=True,
    ),
    Deployment(
        "intfloat/e5-mistral-7b-instruct-embedding",
        "intfloat/e5-mistral-7b-instruct",
        Accelerator.H100_40GB,
        Embedder(),
        is_fp8=True,
    ),
    Deployment(
        "SamLowe/roberta-base-go_emotions-classification",
        "SamLowe/roberta-base-go_emotions",
        Accelerator.L4,
        Predictor(),
    ),
    Deployment(
        "papluca/xlm-roberta-base-language-detection-classification",
        "papluca/xlm-roberta-base-language-detection",
        Accelerator.L4,
        Predictor(),
    ),
    Deployment(
        "BAAI/bge-reranker-large-classification",
        "BAAI/bge-reranker-large",
        Accelerator.L4,
        Reranker(),
    ),
    Deployment(
        "BAAI/bge-reranker-v2-m3-multilingual",
        "BAAI/bge-reranker-v2-m3",
        Accelerator.A100,  # Bert has long-context issues (>8K tokens on 24Gb Ram machines. Using 80B therefore)
        Reranker(),
    ),
    Deployment(
        "ncbi/MedCPT-Cross-Encoder-reranker",
        "ncbi/MedCPT-Cross-Encoder",
        Accelerator.L4,
        Reranker(),
    ),
    Deployment(
        "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2-Reward-Model",
        "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        Accelerator.H100_40GB,
        Predictor(),
        is_fp8=True,
    ),
    Deployment(
        "allenai/Llama-3.1-Tulu-3-8B-Reward-Model",
        "allenai/Llama-3.1-Tulu-3-8B-RM",
        Accelerator.H100_40GB,
        Predictor(),
        is_fp8=True,
    ),
    Deployment(
        "baseten/example-Meta-Llama-3-70B-InstructForSequenceClassification",
        "baseten/example-Meta-Llama-3-70B-InstructForSequenceClassification",
        Accelerator.H100,
        Predictor(),
        is_fp8=True,
    ),
]

if __name__ == "__main__":
    for dp in DEPLOYMENTS_BEI:
        generate_bei_deployment(dp)

    all_deployments = DEPLOYMENTS_BEI

    def format_filter(dps: list[Deployment], type_):
        sorted_filter = sorted(
            [dp for dp in dps if isinstance(dp.task, type_)], key=lambda x: x.name
        )
        names = [
            f"[{dp.name}]({REPO_URL}/tree/main/{SUBFOLDER}/{name_generator(dp)})"
            for dp in sorted_filter
        ]
        names_fmt = "\n - ".join(names)
        names_fmt = " - " + names_fmt
        return names_fmt

    embedders_names_fmt = format_filter(all_deployments, Embedder)
    rerankers_names_fmt = format_filter(all_deployments, Reranker)
    predictors_names_fmt = format_filter(all_deployments, Predictor)

    readme = f"""
# BEI with Baseten

This is a collection of BEI (Baseten Embeddings Inference) model implementations for deployment to Baseten. BEI is Baseten's solution for production-grade embeddings/re-ranking and classification inference using TensorRT-LLM.

With BEI you get the following benefits:
- *Lowest-latency inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama)<sup>1</sup>
- *Highest-throughput inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama) - thanks to XQA kernels, FP8 and dynamic batching.<sup>2</sup>
- High parallelism: up to 1400 client embeddings per second
- Cached model weights for fast vertical scaling and high availability - no Hugging Face hub dependency at runtime

# Examples:
You can find the following deployments in this repository:

## Embedding Model Deployments:
{embedders_names_fmt}

## Reranker Deployments:
{rerankers_names_fmt}

## Text Sequence Classification Deployments:
{predictors_names_fmt}

<sup>1</sup> measured on H100-HBM3 (bert-large-335M, for MistralModel-7B: 9ms)
<sup>2</sup> measured on H100-HBM3 (leading model architecture on MTEB, MistralModel-7B)
"""
    (Path(__file__).parent.parent / "README.md").write_text(readme)
    print(readme)
