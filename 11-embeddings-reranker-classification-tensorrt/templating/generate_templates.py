from dataclasses import field
from pathlib import Path
from typing import Any, Optional

from pydantic import dataclasses
from transformers import AutoConfig
from truss.base.trt_llm_config import (
    CheckpointRepository,
    CheckpointSource,
    TRTLLMConfiguration,
    TrussTRTLLMBuildConfiguration,
    TrussTRTLLMModel,
    TrussTRTLLMPluginConfiguration,
    TrussTRTLLMQuantizationType,
    TrussTRTLLMRuntimeConfiguration,
)
from truss.base.truss_config import Accelerator, AcceleratorSpec, Resources, TrussConfig

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
class Solution:
    name: str
    benefits: str
    nickname: str
    trt_config: Optional[TRTLLMConfiguration] = None

    def make_headline(self, dp: "Deployment"):
        # e.g. BEI (Baseten-Embeddings-Inference) with {dp.name}.
        return f"{self.name} with {dp.name}"

    def make_truss_config(self, dp: "Deployment") -> TrussConfig:
        pass


@dataclasses.dataclass
class BEI(Solution):
    name: str = "BEI (Baseten-Embeddings-Inference)"
    nickname: str = "BEI"
    benefits: str = """BEI is Baseten's solution for production-grade deployments via TensorRT-LLM for (text) embeddings, reranking models and prediction models.

With BEI you get the following benefits:
- *Lowest-latency inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama)<sup>1</sup>
- *Highest-throughput inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama) - thanks to XQA kernels, FP8 and dynamic batching.<sup>2</sup>
- High parallelism: up to 1400 client embeddings per second
- Cached model weights for fast vertical scaling and high availability - no Hugging Face hub dependency at runtime
"""

    def make_truss_config(self, dp: "Deployment") -> TrussConfig:
        hf_cfg = AutoConfig.from_pretrained(
            dp.hf_model_id, trust_remote_code=True
        )  # make sure model is available
        max_position_embeddings = hf_cfg.max_position_embeddings

        max_num_tokens = max(16384, max_position_embeddings)
        return TrussConfig(
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
            model_name=dp.model_nickname,
        )


@dataclasses.dataclass
class HFTEI(Solution):
    name: str = "Huggingface's text-embeddings-inference"
    nickname: str = "TEI"
    benefits: str = """TEI is huggingface's solution for (text) embeddings, reranking models and prediction models.

Supported models are tagged here: https://huggingface.co/models?other=text-embeddings-inference&sort=trending


For TEI you have to perform a manual selection of the Docker Image. We have mirrored the following images:
```
CPU	baseten/text-embeddings-inference-mirror:cpu-1.6
Turing (T4, ...)	baseten/text-embeddings-inference-mirror:turing-1.6
Ampere 80 (A100, A30)	baseten/text-embeddings-inference-mirror:1.6
Ampere 86 (A10, A10G, A40, ...)	baseten/text-embeddings-inference-mirror:86-1.6
Ada Lovelace (L4, ...)	baseten/text-embeddings-inference-mirror:89-1.6
Hopper (H100/H100 40GB/H200)	baseten/text-embeddings-inference-mirror:hopper-1.6
```
"""

    def make_truss_config(self, dp: "Deployment") -> TrussConfig:
        try:
            AutoConfig.from_pretrained(
                dp.hf_model_id, trust_remote_code=True
            )  # make sure model is available
        except ImportError:
            pass
        docker_image = {
            Accelerator.L4: "baseten/text-embeddings-inference-mirror:89-1.6",
            Accelerator.A100: "baseten/text-embeddings-inference-mirror:1.6",
            Accelerator.H100: "baseten/text-embeddings-inference-mirror:89-1.6",
            Accelerator.H100_40GB: "baseten/text-embeddings-inference-mirror:hopper-1.6",
            Accelerator.A10G: "baseten/text-embeddings-inference-mirror:86-1.6",
            Accelerator.T4: "baseten/text-embeddings-inference-mirror:turing-1.6",
            Accelerator.H200: "baseten/text-embeddings-inference-mirror:hopper-1.6",
            Accelerator.V100: "baseten/text-embeddings-inference-mirror:1.6",
        }[dp.accelerator]

        predict_endpoint = (
            "/v1/embeddings"
            if isinstance(dp.task, Embedder)
            else "/predict" if isinstance(dp.task, Predictor) else "/rerank"
        )

        return TrussConfig(
            base_image=dict(image=docker_image),
            model_metadata=dp.task.model_metadata,
            docker_server=dict(
                start_command=f"text-embeddings-router --port 7997 --model-id /data/local-model --max-client-batch-size 128 --max-concurrent-requests 40 --max-batch-tokens 16384",
                readiness_endpoint="/health",
                liveness_endpoint="/health",
                predict_endpoint=predict_endpoint,
                server_port=7997,
            ),
            resources=Resources(
                accelerator=dp.accelerator,
                use_gpu=True,
            ),
            model_name=dp.model_nickname,
            build_commands=[
                f"git clone https://huggingface.co/{dp.hf_model_id} /data/local-model # optional step to download the weights of the model into the image, otherwise specify the --model-id {dp.hf_model_id} directly `start_command`",
            ],
            runtime=dict(
                predict_concurrency=40,
            ),
        )


@dataclasses.dataclass
class Briton(Solution):
    name: str = "TensorRT-LLM Briton"
    nickname: str = "Briton"
    benefits: str = """Briton is Baseten's solution for production-grade deployments via TensorRT-LLM for Causal Language Models models. (e.g. LLama, Qwen, Mistral)

With Briton you get the following benefits by default:
- *Lowest-latency* latency, beating frameworks such as vllm
- *Highest-throughput* inference, automatically using XQA kernels, paged kv caching and inflight batching.
- *distributed inference* run large models (such as LLama-405B) tensor-parallel
- *json-schema based structured output for any model*
- *chunked prefilling* for long generation tasks

Optionally, you can also enable:
- *speculative decoding* using an external draft model or self-speculative decoding
- *fp8 quantization* deployments on H100, H200 and L4 GPUs
"""

    def make_truss_config(self, dp):
        hf_cfg = AutoConfig.from_pretrained(
            dp.hf_model_id, trust_remote_code=True
        )  # make sure model is available
        max_position_embeddings = hf_cfg.max_position_embeddings
        assert self.trt_config is not None
        self.trt_config.build.max_seq_len = max(max_position_embeddings, 512)

        return TrussConfig(
            model_metadata=dp.task.model_metadata,
            resources=Resources(
                accelerator=AcceleratorSpec(
                    accelerator=dp.accelerator,
                    count=max(1, self.trt_config.build.tensor_parallel_count),
                ).to_str(),
                use_gpu=True,
            ),
            model_name=dp.model_nickname,
            trt_llm=self.trt_config,
        )


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
class TextGen(Task):
    purpose: str = (
        " is a text-generation model, used to generate text given a prompt. \\n"
        "It is frequently used in chatbots, text completion, structured output and more."
    )
    model_identification: str = (
        "Suitable models can be identified by the `ForCausalLM` suffix in the model name. "
        "Currently we support e.g. LLama, Qwen, Mistral models."
    )
    model_metadata: dict = field(
        default_factory=lambda: dict(
            tags=["openai-compatible"],
            example_model_input=dict(
                messages=[
                    {
                        "role": "user",
                        "content": "Tell me everything you know about optimized inference.",
                    },
                ],
                temperature=0.5,
                max_tokens=512,
                stream=True,
            ),
        )
    )
    client_usage: str = r"""
### OpenAI compatible inference
Briton is OpenAI compatible, which means you can use the OpenAI client library to interact with the model.

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ['BASETEN_API_KEY'],
    api_url="https://model-xxxxxx.api.baseten.co/environments/production/sync"
)

# Default completion
response_completion = client.completions.create(
    model="not_required",
    prompt="Q: Tell me everything about Baseten.co! A:",
    temperature=0.3,
    max_tokens=100,
)

# Chat completion
response_chat = client.chat.completions.create(
    model="",
    messages=[
        {"role": "user", "content": "Tell me everything about Baseten.co!"}
    ],
    temperature=0.3,
    max_tokens=100,
)

# Structured output
from pydantic import BaseModel

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

completion = client.beta.chat.completions.parse(
    model="not_required",
    messages=[
        {"role": "system", "content": "Extract the event information."},
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ],
    response_format=CalendarEvent,
)

event = completion.choices[0].message.parsed

# If you model supports tool-calling, you can use the following example:
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia"
                }
            },
            "required": [
                "location"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}]

completion = client.chat.completions.create(
    model="not_required",
    messages=[{"role": "user", "content": "What is the weather like in Paris today?"}],
    tools=tools
)

print(completion.choices[0].message.tool_calls)
```
"""


@dataclasses.dataclass
class Deployment:
    name: str
    hf_model_id: str
    accelerator: Accelerator
    task: Task
    solution: Solution
    is_gated: bool = False
    is_fp8: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.solution.trt_config is not None:
            self.is_fp8 = (
                "fp8" in self.solution.trt_config.build.quantization_type.value
            )

        try:
            AutoConfig.from_pretrained(self.hf_model_id, token="invalid")
            self.is_gated = False
        except Exception:
            try:
                # has only access with permissions
                AutoConfig.from_pretrained(self.hf_model_id)
                self.is_gated = True
            except Exception:
                raise

    @property
    def folder_name(self):
        return (
            self.solution.nickname
            + "-"
            + self.name.replace(" ", "-").replace("/", "-").lower()
            + ("-fp8" * self.is_fp8)
        )

    @property
    def model_nickname(self):
        return self.folder_name + "-truss-example"


def generate_bei_deployment(dp: Deployment):
    root = Path(__file__).parent.parent.parent
    assert root.name == ROOT_NAME.name, "This script has been moved"

    folder_relative_path = SUBFOLDER / dp.folder_name
    full_folder_path = root / folder_relative_path
    is_gated = (
        "Note: [This is a gated/private model] Retrieve your Hugging Face token from the [settings](https://huggingface.co/settings/tokens). "
        "Set your Hugging Face token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_access_key`."
        if dp.is_gated
        else ""
    )

    quantization_disclaimer = (
        "\nThis model is quantized to FP8 for deployment, which is supported by Nvidia's newest GPUs e.g. H100, H100_40GB or L4. "
        "Quantization is optional, but leads to higher efficiency."
        if dp.is_fp8
        else ""
    )

    config = dp.solution.make_truss_config(dp)
    if (
        config.trt_llm is not None
        and config.trt_llm.build.quantization_type
        != TrussTRTLLMQuantizationType.NO_QUANT
    ):
        quantization_removal = (
            f" This config uses `quantization_type={config.trt_llm.build.quantization_type.value}`. "
            "This is optional, remove the `quantization_type` field or set it to `no_quant` for float16/bfloat16."
        )
    else:
        quantization_removal = ""

    # Writes
    full_folder_path.mkdir(parents=True, exist_ok=True)
    config.write_to_yaml_file(full_folder_path / "config.yaml", verbose=False)
    config_yaml_as_str = Path(full_folder_path / "config.yaml").read_text()

    README_SUBREPO = f"""# {dp.solution.make_headline(dp)}

This is a Deployment for {dp.solution.make_headline(dp)}. {dp.solution.benefits}

# Examples:
This deployment is specifically designed for the Hugging Face model [{dp.hf_model_id}](https://huggingface.co/{dp.hf_model_id}).
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
# ✨ Model {dp.model_nickname} was successfully pushed ✨
# 🪵  View logs for your deployment at https://app.baseten.co/models/yyyyyy/logs/xxxxxx
```

## Call your model
{dp.task.client_usage}

## Config.yaml
By default, the following configuration is used for this deployment.{quantization_removal}

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
        solution=BEI(),
    ),
    Deployment(
        "WhereIsAI/UAE-Large-V1-embedding",
        "WhereIsAI/UAE-Large-V1",
        Accelerator.L4,
        Embedder(),
        solution=BEI(),
    ),
    Deployment(
        "mixedbread-ai/mxbai-embed-large-v1-embedding",
        "mixedbread-ai/mxbai-embed-large-v1",
        Accelerator.L4,
        Embedder(),
        solution=BEI(),
    ),
    Deployment(
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        "Snowflake/snowflake-arctic-embed-l-v2.0",
        Accelerator.A100,  # Bert has long-context issues (>8K tokens on 24Gb Ram machines. Using 80B therefore)
        Embedder(),
        solution=BEI(),
    ),
    Deployment(
        "BAAI/bge-m3-embedding-dense",
        "BAAI/bge-m3",
        Accelerator.A100,  # Bert has long-context issues (>8K tokens on 24Gb Ram machines. Using 80B therefore)
        Embedder(),
        solution=BEI(),
    ),
    # Deployment( # no slidig window support for >4096
    # this PR needs to be merged first: or use this revision https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral/discussions/7
    #     "Linq-AI-Research/Linq-Embed-Mistral",
    #     "Linq-AI-Research/Linq-Embed-Mistral",
    #     Accelerator.H100_40GB,
    #     Embedder(),
    #     is_fp8=True,
    Deployment(
        "BAAI/bge-multilingual-gemma2-multilingual-embedding",
        "BAAI/bge-multilingual-gemma2",
        Accelerator.H100_40GB,
        Embedder(),
        # no fp8 support for Gemma in ModelOPT quant, but requires 18GB+ RAM -> A100 or H100Mig
        solution=BEI(),
    ),
    Deployment(
        "Salesforce/SFR-Embedding-Mistral",
        "Salesforce/SFR-Embedding-Mistral",
        Accelerator.H100_40GB,
        Embedder(),
        is_fp8=True,
        solution=BEI(),
    ),
    Deployment(
        "BAAI/bge-en-icl-embedding",
        "BAAI/bge-en-icl",
        Accelerator.H100,
        Embedder(),
        is_fp8=True,
        solution=BEI(),
    ),
    Deployment(
        "intfloat/e5-mistral-7b-instruct-embedding",
        "intfloat/e5-mistral-7b-instruct",
        Accelerator.H100,
        Embedder(),
        is_fp8=True,
        solution=BEI(),
    ),
    Deployment(
        "SamLowe/roberta-base-go_emotions-classification",
        "SamLowe/roberta-base-go_emotions",
        Accelerator.L4,
        Predictor(),
        solution=BEI(),
    ),
    Deployment(
        "papluca/xlm-roberta-base-language-detection-classification",
        "papluca/xlm-roberta-base-language-detection",
        Accelerator.L4,
        Predictor(),
        solution=BEI(),
    ),
    Deployment(
        "BAAI/bge-reranker-large",
        "BAAI/bge-reranker-large",
        Accelerator.L4,
        Reranker(),
        solution=BEI(),
    ),
    Deployment(
        "BAAI/bge-reranker-v2-m3-multilingual",
        "BAAI/bge-reranker-v2-m3",
        Accelerator.A100,  # Bert has long-context issues (>8K tokens on 24Gb Ram machines. Using 80B therefore)
        Reranker(),
        solution=BEI(),
    ),
    Deployment(
        "ncbi/MedCPT-Cross-Encoder-reranker",
        "ncbi/MedCPT-Cross-Encoder",
        Accelerator.L4,
        Reranker(),
        solution=BEI(),
    ),
    Deployment(
        "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2-Reward-Model",
        "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        Accelerator.L4,
        Predictor(),
        is_fp8=True,
        solution=BEI(),
    ),
    Deployment(
        "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2-Reward-Model",
        "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        Accelerator.H100_40GB,
        Predictor(),
        is_fp8=False,
        solution=BEI(),
    ),
    Deployment(
        "allenai/Llama-3.1-Tulu-3-8B-Reward-Model",
        "allenai/Llama-3.1-Tulu-3-8B-RM",
        Accelerator.H100_40GB,
        Predictor(),
        is_fp8=True,
        solution=BEI(),
    ),
    Deployment(
        "baseten/example-Meta-Llama-3-70B-InstructForSequenceClassification",
        "baseten/example-Meta-Llama-3-70B-InstructForSequenceClassification",
        Accelerator.H100,
        Predictor(),
        is_fp8=True,
        solution=BEI(),
    ),
]

DEPLOYMENTS_HFTEI = [  # models that don't yet run on BEI
    Deployment(  #
        name="jina-ai/jina-embeddings-v2-base-en",
        hf_model_id="jinaai/jina-embeddings-v2-base-en",
        accelerator=Accelerator.L4,
        task=Embedder(),
        solution=HFTEI(),
    ),
    Deployment(
        name="jinaai/jina-embeddings-v2-base-code",
        hf_model_id="jinaai/jina-embeddings-v2-base-code",
        accelerator=Accelerator.L4,
        task=Embedder(),
        solution=HFTEI(),
    ),
    Deployment(  #
        name="sentence-transformers/all-MiniLM-L6-v2-embedding",
        hf_model_id="sentence-transformers/all-MiniLM-L6-v2",
        accelerator=Accelerator.L4,
        task=Embedder(),
        solution=HFTEI(),
    ),
    Deployment(  #
        name="nomic-ai/nomic-embed-text-v1.5",
        hf_model_id="nomic-ai/nomic-embed-text-v1.5",
        accelerator=Accelerator.L4,
        task=Embedder(),
        solution=HFTEI(),
    ),
    Deployment(  #
        name="Alibaba-NLP/gte-Qwen2-1.5B-instruct-embedding",
        hf_model_id="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        accelerator=Accelerator.L4,
        task=Embedder(),
        solution=HFTEI(),
    ),
    Deployment(  #
        name="intfloat/multilingual-e5-large-instruct",
        hf_model_id="intfloat/multilingual-e5-large-instruct",
        accelerator=Accelerator.L4,
        task=Embedder(),
        solution=HFTEI(),
    ),
    Deployment(  #
        name="Alibaba-NLP/gte-multilingual-reranker-base",
        hf_model_id="Alibaba-NLP/gte-multilingual-reranker-base",
        accelerator=Accelerator.L4,
        task=Reranker(),
        solution=HFTEI(),
    ),
]


def llamalike_config(
    quant: TrussTRTLLMQuantizationType = TrussTRTLLMQuantizationType.FP8_KV,
    tp=1,
    repoid="meta-llama/Llama-3.3-70B-Instruct",
):
    # config for meta-llama/Llama-3.3-70B-Instruct (FP8)
    build_kwargs = dict()
    if quant != TrussTRTLLMQuantizationType.NO_QUANT:
        if tp == 1:
            build_kwargs["num_builder_gpus"] = 4
    if quant == TrussTRTLLMQuantizationType.FP8_KV:
        build_kwargs["plugin_configuration"] = TrussTRTLLMPluginConfiguration(
            use_fp8_context_fmha=True
        )

    return TRTLLMConfiguration(
        build=TrussTRTLLMBuildConfiguration(
            base_model=TrussTRTLLMModel.LLAMA,
            checkpoint_repository=CheckpointRepository(
                repo=repoid,
                revision="main",
                source=CheckpointSource.HF,
            ),
            max_seq_len=1000001,  # dummy for now
            quantization_type=quant,
            tensor_parallel_count=tp,
            **build_kwargs,
        ),
        runtime=TrussTRTLLMRuntimeConfiguration(
            enable_chunked_context=True,
        ),
    )


DEPLOYMENTS_BRITON = [
    Deployment(
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(repoid="meta-llama/Llama-3.3-70B-Instruct")
        ),
    ),
    Deployment(
        "meta-llama/Llama-3.3-70B-Instruct-tp2",
        "meta-llama/Llama-3.3-70B-Instruct",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="meta-llama/Llama-3.3-70B-Instruct", tp=2
            )
        ),
        is_gated=True,
    ),  # meta-llama/Llama-3.1-405B tp8
    Deployment(
        "meta-llama/Llama-3.1-405B",
        "meta-llama/Llama-3.1-405B",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(repoid="meta-llama/Llama-3.1-405B", tp=8)
        ),
        is_gated=True,
    ),
    Deployment(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                quant=TrussTRTLLMQuantizationType.FP8,  # no KV quantization fgor Qwen
            )
        ),
    ),
    Deployment(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                quant=TrussTRTLLMQuantizationType.FP8_KV,
                tp=2,
            )
        ),
    ),
    # Qwen/Qwen2.5-72B-Instruct
    Deployment(
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(repoid="Qwen/Qwen2.5-72B-Instruct", tp=4)
        ),
    ),
    # mistralai/Mistral-Small-24B-Instruct-2501
    Deployment(
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="mistralai/Mistral-Small-24B-Instruct-2501"
            )
        ),
        is_gated=True,
    ),  # unsloth/phi-4
    Deployment(
        "microsoft/phi-4",
        "unsloth/phi-4",
        Accelerator.L4,
        TextGen(),
        solution=Briton(trt_config=llamalike_config(repoid="unsloth/phi-4")),
    ),
    Deployment(
        "tiiuae/Falcon3-10B-Instruct",
        "tiiuae/Falcon3-10B-Instruct",
        Accelerator.L4,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(repoid="tiiuae/Falcon3-10B-Instruct")
        ),
        is_gated=True,
    ),
    Deployment(
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.3",
        Accelerator.A10G,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="mistralai/Mistral-7B-Instruct-v0.3",
                quant=TrussTRTLLMQuantizationType.NO_QUANT,
                tp=2,
            )
        ),
    ),
]


ALL_DEPLOYMENTS = DEPLOYMENTS_BEI + DEPLOYMENTS_HFTEI + DEPLOYMENTS_BRITON

if __name__ == "__main__":
    for dp in ALL_DEPLOYMENTS:
        generate_bei_deployment(dp)

    def format_filter(dps: list[Deployment], type_):
        sorted_filter = sorted(
            [dp for dp in dps if isinstance(dp.task, type_)], key=lambda x: x.name
        )
        names = [
            f"[{dp.name}-{dp.solution.nickname}]({REPO_URL}/tree/main/{SUBFOLDER}/{dp.folder_name})"
            for dp in sorted_filter
        ]
        names_fmt = "\n - ".join(names)
        names_fmt = " - " + names_fmt
        return names_fmt

    embedders_names_fmt = format_filter(ALL_DEPLOYMENTS, Embedder)
    rerankers_names_fmt = format_filter(ALL_DEPLOYMENTS, Reranker)
    predictors_names_fmt = format_filter(ALL_DEPLOYMENTS, Predictor)
    generation_names_fmt = format_filter(ALL_DEPLOYMENTS, TextGen)

    readme = f"""
# Performance Section
Below are a example deployments of optimized models for the Baseten platform.

# Baseten Embeddings Inference (BEI)

Collection of BEI (Baseten Embeddings Inference) model implementations for deployment to Baseten. BEI is Baseten's solution for production-grade embeddings/re-ranking and classification inference using TensorRT-LLM.

With BEI you get the following benefits:
- *Lowest-latency inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama)<sup>1</sup>
- *Highest-throughput inference* across any embedding solution (vLLM, SGlang, Infinity, TEI, Ollama) - thanks to XQA kernels, FP8 and dynamic batching.<sup>2</sup>
- High parallelism: up to 1400 client embeddings per second
- Cached model weights for fast vertical scaling and high availability - no Hugging Face hub dependency at runtime

Architectures that are not supported on BEI are deployed with Huggingface's text-embeddings-inference (TEI) solution.

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

# Text-Generation - Briton
Briton is Baseten's solution for production-grade deployments via TensorRT-LLM for Text-generation models. (e.g. LLama, Qwen, Mistral)

With Briton you get the following benefits by default:
- *Lowest-latency* latency, beating frameworks such as vllm
- *Highest-throughput* inference - tensorrt-llm will automatically use XQA kernels, paged kv caching and inflight batching.
- *distributed inference* run large models (such as LLama-3-405B) in tensor-parallel
- *json-schema based structured output for any model*
- *chunked prefilling* for long generation tasks

Optionally, you can also enable:
- *speculative decoding* using external draft models or lookahead decoding
- *fp8 quantization* on new GPUS such as H100, H200 and L4 GPUs

Examples:
{generation_names_fmt}
"""
    (Path(__file__).parent.parent / "README.md").write_text(readme)
    print(readme)
