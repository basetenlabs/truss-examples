from dataclasses import field
from functools import cached_property
from pathlib import Path
from typing import Any, Optional
import os
import requests
from pydantic import dataclasses
from transformers import AutoConfig
from truss.base.trt_llm_config import (
    CheckpointRepository,
    CheckpointSource,
    TRTLLMConfiguration,
    TRTLLMConfigurationV2,
    TrussSpeculatorConfiguration,
    TrussTRTLLMBuildConfiguration,
    TrussTRTLLMModel,
    TrussTRTLLMPluginConfiguration,
    TrussTRTLLMQuantizationType,
    TrussTRTLLMRuntimeConfiguration,
    TRTLLMRuntimeConfigurationV2,
    TrussTRTQuantizationConfiguration,
    VersionsOverrides,
)
from truss.base.truss_config import (
    Accelerator,
    AcceleratorSpec,
    ModelCache,
    ModelRepo,
    Resources,
    TrussConfig,
)
import yaml
import copy

REPO_URL = "https://github.com/basetenlabs/truss-examples"
SUBFOLDER = Path("11-embeddings-reranker-classification-tensorrt")
ROOT_NAME = Path(REPO_URL.split("/")[-1])
BEI_VERSION = os.environ.get("BEI")
ENGINE_BUILDER_VERSION = os.environ.get("ENGINE_BUILDER")
BRITON_VERSION = os.environ.get("BRITON")


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
    make_fp8: bool = False
    make_fp4: bool = False
    use_bei_bert: bool = False

    def __post_init__(self):
        if self.make_fp8 and self.make_fp4:
            raise ValueError("make_fp8 and make_fp4 cannot both be True")
        if self.use_bei_bert and (self.make_fp4 or self.make_fp8):
            raise ValueError("BEI BERT does not support FP4 or FP8 quantization")

    def suffix(self):
        if self.use_bei_bert:
            return "-bert"
        else:
            return ""

    def make_truss_config(self, dp: "Deployment") -> TrussConfig:
        hf_cfg = AutoConfig.from_pretrained(
            dp.hf_model_id, trust_remote_code=True
        )  # make sure model is available
        max_position_embeddings = hf_cfg.max_position_embeddings

        max_num_tokens = max(16384, max_position_embeddings)

        num_builder_gpus = 1
        if dp.accelerator in [Accelerator.H100]:
            num_builder_gpus = 2
        elif dp.accelerator in [Accelerator.L4]:
            num_builder_gpus = 4
        endpoint = (
            "/v1/embeddings"
            if isinstance(dp.task, Embedder)
            else "/predict"
            if isinstance(dp.task, Predictor)
            else "/rerank"
        )
        try:
            trt_llm = TRTLLMConfiguration(
                build=TrussTRTLLMBuildConfiguration(
                    base_model=TrussTRTLLMModel.ENCODER_BERT
                    if self.use_bei_bert
                    else TrussTRTLLMModel.ENCODER,
                    checkpoint_repository=CheckpointRepository(
                        repo=dp.hf_model_id,
                        revision="main",
                        source=CheckpointSource.HF,
                    ),
                    max_num_tokens=max_num_tokens,
                    **(
                        {
                            "quantization_type": TrussTRTLLMQuantizationType.FP8,
                            # give more resources / cpu ram + vram on build if the model uses not-mig
                            "num_builder_gpus": num_builder_gpus,
                        }
                        if self.make_fp8
                        else (
                            {
                                "quantization_type": TrussTRTLLMQuantizationType.FP4,
                                "num_builder_gpus": num_builder_gpus,
                            }
                            if self.make_fp4
                            else {}
                        )
                    ),
                ),
                runtime=TrussTRTLLMRuntimeConfiguration(
                    webserver_default_route=endpoint,
                ),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create TRTLLMConfiguration for model {dp.hf_model_id}"
            ) from e
        overrides_engine_builder = ENGINE_BUILDER_VERSION
        overrides_bei = BEI_VERSION
        if overrides_engine_builder is not None or overrides_bei is not None:
            trt_llm.root.version_overrides = VersionsOverrides(
                engine_builder_version=overrides_engine_builder,
                bei_version=overrides_bei,
            )

        return TrussConfig(
            model_metadata=dp.task.model_metadata,
            trt_llm=trt_llm,
            resources=Resources(
                accelerator=dp.accelerator,
                memory="10Gi",
            ),
            model_name=dp.model_nickname,
        )


@dataclasses.dataclass
class BEIBert(BEI):
    name: str = "BEI-Bert (Baseten-Embeddings-Inference-BERT)"
    nickname: str = "BEI-Bert"
    use_bei_bert: bool = True
    make_fp8: bool = False
    make_fp4: bool = False

    def __post_init__(self):
        if self.make_fp8 or self.make_fp4:
            raise ValueError("BEI BERT does not support FP4 or FP8 quantization")
        self.use_bei_bert = True
        self.name = "BEI-Bert (Baseten-Embeddings-Inference-BERT)"
        self.nickname = "BEI-Bert"


@dataclasses.dataclass
class HFTEI(Solution):
    name: str = "Huggingface's text-embeddings-inference"
    nickname: str = "TEI"
    benefits: str = """TEI is huggingface's solution for (text) embeddings, reranking models and prediction models.

Supported models are tagged here: https://huggingface.co/models?other=text-embeddings-inference&sort=trending

For TEI you have to perform a manual selection of the Docker Image. We have mirrored the following images:
```
CPU	baseten/text-embeddings-inference-mirror:cpu-1.8.3
Turing (T4, ...)	baseten/text-embeddings-inference-mirror:turing-1.8.3
Ampere 80 (A100, A30)	baseten/text-embeddings-inference-mirror:1.8.3
Ampere 86 (A10, A10G, A40, ...)	baseten/text-embeddings-inference-mirror:86-1.8.3
Ada Lovelace (L4, ...)	baseten/text-embeddings-inference-mirror:89-1.8.3
Hopper (H100/H100 40GB/H200)	baseten/text-embeddings-inference-mirror:hopper-1.8.3
```

As we are deploying mostly tiny models (<1GB), we are downloading the model weights into the docker image.
For larger models, we recommend downloading the weights at runtime for faster autoscaling, as the weights don't need to go through decompression of the docker image.
"""

    def make_truss_config(self, dp: "Deployment") -> TrussConfig:
        try:
            AutoConfig.from_pretrained(
                dp.hf_model_id, trust_remote_code=True
            )  # make sure model is available
        except ImportError:
            pass
        version = "1.8.3"
        docker_image = {
            Accelerator.L4: f"baseten/text-embeddings-inference-mirror:89-{version}",
            Accelerator.A100: f"baseten/text-embeddings-inference-mirror:{version}",
            Accelerator.H100: f"baseten/text-embeddings-inference-mirror:hopper-{version}",
            Accelerator.H100_40GB: f"baseten/text-embeddings-inference-mirror:hopper-{version}",
            Accelerator.A10G: f"baseten/text-embeddings-inference-mirror:86-{version}",
            Accelerator.T4: f"baseten/text-embeddings-inference-mirror:turing-{version}",
            Accelerator.H200: f"baseten/text-embeddings-inference-mirror:hopper-{version}",
            Accelerator.V100: f"baseten/text-embeddings-inference-mirror:{version}",
        }[dp.accelerator]

        predict_endpoint = (
            "/v1/embeddings"
            if isinstance(dp.task, Embedder)
            else "/predict"
            if isinstance(dp.task, Predictor)
            else "/rerank"
        )
        low_cpu_instructions = ""
        if dp.accelerator in [Accelerator.L4, Accelerator.A10G]:
            low_cpu_instructions = " --tokenization-workers 3"
        return TrussConfig(
            base_image=dict(image=docker_image),
            model_metadata=dp.task.model_metadata,
            model_cache=ModelCache(
                [
                    ModelRepo(
                        revision="main",
                        repo_id=dp.hf_model_id,
                        use_volume=True,
                        volume_folder="cached_model",
                        ignore_patterns=["*.pt", "*.ckpt", "*.onnx"],
                    )
                ]
            ),
            docker_server=dict(
                start_command=f'bash -c "truss-transfer-cli && text-embeddings-router --port 7997 --model-id /app/model_cache/cached_model --max-client-batch-size 128 --max-concurrent-requests 1024 --max-batch-tokens 16384 --auto-truncate{low_cpu_instructions}"',
                readiness_endpoint="/health",
                liveness_endpoint="/health",
                predict_endpoint=predict_endpoint,
                server_port=7997,
            ),
            resources=Resources(
                accelerator=dp.accelerator,
            ),
            model_name=dp.model_nickname,
            runtime=dict(
                predict_concurrency=32,
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
        self.trt_config.build.max_seq_len = max_position_embeddings
        assert max_position_embeddings >= 512, "Model needs to have at least 512 tokens"
        if self.trt_config.build.speculator is not None:
            self.trt_config.build.max_seq_len = min(
                self.trt_config.build.max_seq_len, 32768
            )
            self.trt_config.build.max_num_tokens = self.trt_config.build.max_seq_len
            self.trt_config.runtime.enable_chunked_context = False

        if (
            hf_cfg.model_type in ["qwen2", "qwen2_moe"]
            and self.trt_config.build.quantization_type is not None
        ):
            if (
                self.trt_config.build.quantization_type
                == TrussTRTLLMQuantizationType.FP8_KV
            ):
                raise ValueError(
                    f"Qwen2 models do not support FP8_KV quantization / have quality issues with this dtype - please use regular FP8 for now in the model library {dp.hf_model_id}"
                )
            # increase the quantization example size for qwen2 models
            self.trt_config.build.quantization_config = (
                TrussTRTQuantizationConfiguration(
                    calib_size=2048,
                    calib_max_seq_length=min(2048, self.trt_config.build.max_seq_len),
                )
            )

        overrides_engine_builder = ENGINE_BUILDER_VERSION
        overrides_briton = BRITON_VERSION

        if overrides_engine_builder is not None or overrides_briton is not None:
            version_overrides = VersionsOverrides(
                engine_builder_version=overrides_engine_builder,
                briton_version=overrides_briton,
            )
            self.trt_config.root.version_overrides = version_overrides

        return TrussConfig(
            model_metadata=dp.task.model_metadata,
            resources=Resources(
                accelerator=AcceleratorSpec(
                    accelerator=dp.accelerator,
                    count=max(1, self.trt_config.build.tensor_parallel_count),
                ),
                memory="10Gi",
            ),
            model_name=dp.model_nickname,
            trt_llm=self.trt_config,
        )


@dataclasses.dataclass
class BISV2(Solution):
    name: str = "Baseten Inference Stack"
    nickname: str = "BISV2"
    benefits: str = """Baseten Inference Stack is Baseten's solution for production-grade deployments via TensorRT-LLM for Causal Language Models models. (e.g. LLama, Qwen, Mistral)

With Baseten Inference Stack you get the following benefits by default:
- *Lowest-latency* latency, beating frameworks such as vllm
- *Highest-throughput* inference, automatically using XQA kernels, paged kv caching and inflight batching.
- *distributed inference* run large models (such as LLama-405B) tensor-parallel
- *json-schema based structured output for any model*
- *chunked prefilling* for long generation tasks

Optionally, you can also enable:
- *speculative decoding* using an external draft model or self-speculative decoding
- *fp8 quantization* deployments on H100, H200 and L4 GPUs
- *fp4 quantization* deployments on B200 GPUs to get even more speed
"""

    def make_truss_config(self, dp):
        hf_cfg = AutoConfig.from_pretrained(
            dp.hf_model_id, trust_remote_code=True
        )  # make sure model is available
        max_position_embeddings = hf_cfg.max_position_embeddings
        assert self.trt_config is not None
        self.trt_config.runtime.max_seq_len = max_position_embeddings
        assert max_position_embeddings >= 512, "Model needs to have at least 512 tokens"
        if self.trt_config.runtime is not None:
            self.trt_config.runtime.max_seq_len = min(
                self.trt_config.runtime.max_seq_len, 32768
            )
            self.trt_config.runtime.max_num_tokens = self.trt_config.runtime.max_seq_len

        if (
            hf_cfg.model_type in ["qwen2", "qwen2_moe"]
            and self.trt_config.build.quantization_type is not None
        ):
            if (
                self.trt_config.build.quantization_type
                == TrussTRTLLMQuantizationType.FP8_KV
            ):
                raise ValueError(
                    f"Qwen2 models do not support FP8_KV quantization / have quality issues with this dtype - please use regular FP8 for now in the model library {dp.hf_model_id}"
                )
            # increase the quantization example size for qwen2 models
            self.trt_config.build.quantization_config = (
                TrussTRTQuantizationConfiguration(
                    calib_size=2048,
                    calib_max_seq_length=min(2048, self.trt_config.runtime.max_seq_len),
                )
            )

        overrides_engine_builder = ENGINE_BUILDER_VERSION
        overrides_briton = BRITON_VERSION

        if overrides_engine_builder is not None or overrides_briton is not None:
            version_overrides = VersionsOverrides(
                engine_builder_version=overrides_engine_builder,
                briton_version=overrides_briton,
            )
            self.trt_config.root.version_overrides = version_overrides

        return TrussConfig(
            model_metadata=dp.task.model_metadata,
            resources=Resources(
                accelerator=AcceleratorSpec(
                    accelerator=dp.accelerator,
                    count=1,
                ),
                memory="10Gi",
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

### Baseten Performance Client

```bash
pip install baseten-performance-client
```

```python
from baseten_performance_client import PerformanceClient

client = PerformanceClient(
    api_key=os.environ['BASETEN_API_KEY'],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync"
)
texts = ["Hello world", "Example text", "Another sample"]
response = client.embed(
    input=texts,
    model="my_model",
    batch_size=4,
    max_concurrent_requests=32,
    timeout_s=360,
    # dimensions=1536 # optional for fp8 models.
)
print(response.numpy())
```

Read more on the [Baseten Performance Client Blog](https://www.baseten.co/blog/your-client-code-matters-10x-higher-embedding-throughput-with-python-and-rust/)

### OpenAI compatible client library
```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ['BASETEN_API_KEY'],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync/v1"
)

embedding = client.embeddings.create(
    input="Baseten Embeddings are fast",
    model="model"
    # dimensions=1536 # optional for MRL models.
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
            example_model_input={
                "query": "What is Baseten?",
                "raw_scores": True,
                "return_text": True,
                "texts": [
                    "Deep Learning is ...",
                    "Baseten is a fast inference provider",
                ],
                "truncate": True,
                "truncation_direction": "Right",
            }
        )
    )
    client_usage: str = r"""
### API-Schema:
POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/sync/rerank`:
```json
{
    "query": "What is Baseten?",
    "raw_scores": true,
    "return_text": false,
    "texts": [
        "Deep Learning is ...", "Baseten is a fast inference provider"
    ],
    "truncate": true,
    "truncation_direction": "Right"
}
```

### Baseten Performance Client

Read more on the [Baseten Performance Client Blog](https://www.baseten.co/blog/your-client-code-matters-10x-higher-embedding-throughput-with-python-and-rust/)

```python
from baseten_performance_client import PerformanceClient

client = PerformanceClient(
    api_key=os.environ['BASETEN_API_KEY'],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync"
)
response = client.rerank(
    query="What is Baseten?",
    texts=["Deep Learning is ...", "Baseten is a fast inference provider"],
    raw_scores=True,
    return_text=False,
    truncate=True,
)
print(response.data)
```

Sometimes, you may want to apply a custom template to the texts before reranking them and call the predict endpoint instead:

```python
from baseten_performance_client import PerformanceClient

client = PerformanceClient(
    api_key=os.environ['BASETEN_API_KEY'],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync"
)
def template(text: list[str]) -> list[str]:
    # Custom template function to apply to the texts
    # a popular template might be "{query}\n{document}"
    # or also chat-style templates like "User: {query}\nDocument: {document}"
    apply = lambda x: f"Custom template: {x}"
    return [apply(t) for t in text]

response = client.predict(
    inputs=template(["What is baseten? A: Baseten is a fast inference provider", "Classify this separately."]),
    raw_scores=True,
    truncate=True,
)
print(response.data)
```


### Requests python library

```python
import requests
import os

headers = {
    f"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"
}

requests.post(
    headers=headers,
    url="https://model-xxxxxx.api.baseten.co/environments/production/sync/rerank",
    json={
    "query": "What is Baseten?",
    "raw_scores": True,
    "return_text": False,
    "texts": [
        "Deep Learning is ...", "Baseten is a fast inference provider"
    ],
    "truncate": True,
    "truncation_direction": "Right"
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
            example_model_input={
                "inputs": [
                    ["Baseten is a fast inference provider"],
                    ["Classify this separately."],
                ],
                "raw_scores": True,
                "truncate": True,
                "truncation_direction": "Right",
            }
        )
    )
    client_usage: str = r"""
### API-Schema:
POST-Route: `https://model-xxxxxx.api.baseten.co/environments/production/sync/predict`
```json
{
  "inputs": "Baseten is a fast inference provider",
  "raw_scores": true,
  "truncate": true,
  "truncation_direction": "Right"
}
```


### Baseten Performance Client

Read more on the [Baseten Performance Client Blog](https://www.baseten.co/blog/your-client-code-matters-10x-higher-embedding-throughput-with-python-and-rust/)


```bash
pip install baseten-performance-client
```

```python
from baseten_performance_client import PerformanceClient

client = PerformanceClient(
    api_key=os.environ['BASETEN_API_KEY'],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync"
)
def template(text: list[str]) -> list[str]:
    apply = lambda x: f"Custom template: {x}"
    return [apply(t) for t in text]

response = client.predict(
    inputs=template(["Baseten is a fast inference provider", "Classify this separately."]),
    raw_scores=True,
    truncate=True,
)
print(response.data)
```

### Requests python library
```python
import requests
import os

headers = {
    f"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"
}

requests.post(
    headers=headers,
    url="https://model-xxxxxx.api.baseten.co/environments/production/sync/predict",
    json={
        "inputs": [["Baseten is a fast inference provider", ["classify this separately."]],
        "raw_scores": True,
        "truncate": True,
        "truncation_direction": "Right"
    }
)
```
Returns:
```json
[
  [
    {
        "label": "excitement",
        "score": 0.99
    }
  ],
  [
    {
        "label": "excitement",
        "score": 0.01
    }
  ]
]
```
Important, this is different from the `predict` route that you usually call. (https://model-xxxxxx.api.baseten.co/environments/production/predict), it contains an additional `sync` before that.
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
This solution is OpenAI compatible, which means you can use the OpenAI client library to interact with the model.

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ['BASETEN_API_KEY'],
    base_url="https://model-xxxxxx.api.baseten.co/environments/production/sync/v1"
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

    @cached_property
    def is_fp8(self):
        if self.solution.trt_config is not None:
            return "fp8" in self.solution.trt_config.build.quantization_type.value
        elif hasattr(self.solution, "make_fp8"):
            return self.solution.make_fp8
        else:
            return False

    @cached_property
    def is_mlp_only(self):
        if self.solution.trt_config is not None:
            return "mlp_only" in self.solution.trt_config.build.quantization_type.value
        else:
            return False

    @cached_property
    def suffix(self):
        return ""

    @cached_property
    def is_fp4(self):
        if self.solution.trt_config is not None:
            return "fp4" in self.solution.trt_config.build.quantization_type.value
        elif hasattr(self.solution, "make_fp4"):
            return self.solution.make_fp4
        else:
            return False

    @cached_property
    def hf_config(self):
        return AutoConfig.from_pretrained(self.hf_model_id, trust_remote_code=True)

    @cached_property
    def is_gated(self):
        # make sure the model is available via AutoConfig
        assert self.hf_config is not None

        # model_name = "unsloth/phi-4"
        # Attempt to fetch the weights file rather than config.json
        url = f"https://huggingface.co/{self.hf_model_id}/resolve/main/config.json"

        response = requests.get(url)
        if response.status_code == 200:
            return False
        elif response.status_code == 401:
            return True
        else:
            raise ValueError(f"Received HTTP status code: {response.status_code}")

    @property
    def folder_name(self):
        return (
            self.solution.nickname
            + "-"
            + self.name.replace(" ", "-").replace("/", "-").lower()
            + ("-fp8" * self.is_fp8)
            + ("-fp4" * self.is_fp4)
            + ("-mlp-only" * self.is_mlp_only)
            + self.suffix
        )

    @property
    def model_nickname(self):
        return self.folder_name + "-truss-example"


def add_inference_v2_stack(path: Path, dep: Deployment) -> None:
    """
    Edits the YAML at `path` in-place:
      - Only if `should_inject` is True
      - Adds `inference_stack: v2` INSIDE the `trt_llm` mapping
    """
    if not isinstance(dep.solution, BISV2):
        return

    data = yaml.safe_load(path.read_text())
    trt_llm = data.get("trt_llm")
    if isinstance(trt_llm, dict):
        # Build new dict with inference_stack first
        new_trt = {"inference_stack": "v2"}
        new_trt.update({k: v for k, v in trt_llm.items() if k != "inference_stack"})
        data["trt_llm"] = new_trt
        path.write_text(yaml.safe_dump(data, sort_keys=False))


def add_base_model_override(path: Path, dep: Deployment) -> None:
    """
    Edits the YAML at `path` in-place:
      - Only if `should_inject` is True
      - Adds `base_model: ...` INSIDE the `trt_llm` mapping
    """
    if not isinstance(dep.solution, BISV2):
        return

    data = yaml.safe_load(path.read_text())
    build_details = data.get("trt_llm").get("build")
    if isinstance(build_details, dict):
        # Build new dict with base_model first
        new_build = {"base_model": "decoder"}
        new_build.update({k: v for k, v in build_details.items() if k != "base_model"})
        data["trt_llm"]["build"] = new_build
        path.write_text(yaml.safe_dump(data, sort_keys=False))


def generate_deployment(dp: Deployment):
    root = Path(__file__).parent.parent.parent
    assert root.name == ROOT_NAME.name, "This script has been moved"

    folder_relative_path = SUBFOLDER / dp.folder_name
    full_folder_path = root / folder_relative_path
    is_gated_notice = (
        "Note: [This is a gated/private model] Retrieve your Hugging Face token from the [settings](https://huggingface.co/settings/tokens). "
        "Set your Hugging Face token as a Baseten secret [here](https://app.baseten.co/settings/secrets) with the key `hf_access_token`. "
        "Do not set the actual value of key in the config.yaml. `hf_access_token: null` is fine - the true value will be fetched from the secret store."
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
    config_yaml_path = full_folder_path / "config.yaml"
    config.write_to_yaml_file(config_yaml_path, verbose=False)
    config_yaml_as_str = Path(config_yaml_path).read_text()
    header = "# this file was autogenerated by `generate_templates.py` - please do change via template only\n"
    Path(config_yaml_path).write_text(header + config_yaml_as_str)

    add_inference_v2_stack(config_yaml_path, dp)
    # add_base_model_override(config_yaml_path, dp)

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
{is_gated_notice}

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
{is_gated_notice}
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
        "jinaai/jina-code-embeddings-0.5b",
        "jinaai/jina-code-embeddings-0.5b",
        Accelerator.H100_40GB,
        Embedder(),
        solution=BEI(make_fp8=True),
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
        Accelerator.H100,  # Bert has long-context issues (>8K tokens on 24Gb Ram machines. Using 80B therefore)
        Embedder(),
        solution=BEI(),
    ),
    Deployment(
        "BAAI/bge-m3-embedding-dense",
        "BAAI/bge-m3",
        Accelerator.H100,  # Bert has long-context issues (>8K tokens on 24Gb Ram machines. Using 80B therefore)
        Embedder(),
        solution=BEI(),
    ),
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
        solution=BEI(make_fp8=True),
    ),
    Deployment(
        "BAAI/bge-en-icl-embedding",
        "BAAI/bge-en-icl",
        Accelerator.H100,
        Embedder(),
        solution=BEI(make_fp8=True),
    ),
    Deployment(
        "intfloat/e5-mistral-7b-instruct-embedding",
        "intfloat/e5-mistral-7b-instruct",
        Accelerator.H100,
        Embedder(),
        solution=BEI(make_fp8=True),
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
        Accelerator.H100,  # Bert has long-context issues (>8K tokens on 24Gb Ram machines. Using 80B therefore)
        Reranker(),
        solution=BEI(),
    ),
    Deployment(
        "ncbi/MedCPT-Cross-Encoder-reranker",
        "ncbi/MedCPT-Cross-Encoder",
        Accelerator.A10G,
        Reranker(),
        solution=BEI(),
    ),
    Deployment(
        "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2-Reward-Model",
        "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        Accelerator.H100_40GB,
        Predictor(),
        solution=BEI(make_fp8=True),
    ),
    Deployment(
        "allenai/Llama-3.1-Tulu-3-8B-Reward-Model",
        "allenai/Llama-3.1-Tulu-3-8B-RM",
        Accelerator.H100_40GB,
        Predictor(),
        solution=BEI(make_fp8=True),
    ),
    Deployment(
        "mixedbread-ai/mxbai-rerank-large-v2-reranker",
        "michaelfeil/mxbai-rerank-large-v2-seq",
        Accelerator.L4,
        Predictor(),
        solution=BEI(make_fp8=True),
    ),
    Deployment(
        "mixedbread-ai/mxbai-rerank-base-v2-reranker",
        "michaelfeil/mxbai-rerank-base-v2-seq",
        Accelerator.L4,
        Predictor(),
        solution=BEI(make_fp8=True),
    ),
    Deployment(
        "baseten/example-Meta-Llama-3-70B-InstructForSequenceClassification",
        "baseten/example-Meta-Llama-3-70B-InstructForSequenceClassification",
        Accelerator.H100,
        Predictor(),
        solution=BEI(make_fp8=True),
    ),
    Deployment(
        "nomic-ai/nomic-embed-code",
        "nomic-ai/nomic-embed-code",
        Accelerator.H100_40GB,
        Embedder(),
        solution=BEI(make_fp8=True),
    ),
    Deployment(
        "Qwen/Qwen3-Embedding-8B",
        "michaelfeil/Qwen3-Embedding-8B-auto",
        Accelerator.H100_40GB,
        Embedder(),
        solution=BEI(make_fp8=True),
    ),
    Deployment(
        "Qwen/Qwen3-Embedding-4B",
        "michaelfeil/Qwen3-Embedding-4B-auto",
        Accelerator.H100_40GB,
        Embedder(),
        solution=BEI(make_fp8=True),
    ),
    Deployment(
        "Qwen/Qwen3-Embedding-0.6B",
        "michaelfeil/Qwen3-Embedding-0.6B-auto",
        Accelerator.L4,
        Embedder(),
        solution=BEI(make_fp8=True),
    ),
    Deployment(
        "Qwen/Qwen3-Reranker-0.6B",
        "michaelfeil/Qwen3-Reranker-0.6B-seq",
        Accelerator.L4,
        Predictor(),
        solution=BEI(make_fp8=True),
    ),
    Deployment(
        "Qwen/Qwen3-Reranker-4B",
        "michaelfeil/Qwen3-Reranker-4B-seq",
        Accelerator.H100_40GB,
        Predictor(),
        solution=BEI(make_fp8=True),
    ),
    Deployment(
        "Qwen/Qwen3-Reranker-8B",
        "michaelfeil/Qwen3-Reranker-8B-seq",
        Accelerator.H100_40GB,
        Predictor(),
        solution=BEI(make_fp8=True),
    ),
    Deployment(
        "Qwen/Qwen3-Embedding-4B",
        "michaelfeil/Qwen3-Embedding-4B-auto",
        Accelerator.B200,
        Embedder(),
        solution=BEI(make_fp4=True),
    ),
    Deployment(
        "Qwen/Qwen3-Reranker-8B",
        "michaelfeil/Qwen3-Reranker-8B-seq",
        Accelerator.B200,
        Predictor(),
        solution=BEI(make_fp4=True),
    ),
]

DEPLOYMENTS_HFTEI = [  # models that don't yet run on BEI
    Deployment(  #
        name="BAAI/bge-reranker-large",
        hf_model_id="BAAI/bge-reranker-large",
        accelerator=Accelerator.H100,
        task=Reranker(),
        solution=HFTEI(),
    ),
    Deployment(  #
        name="Alibaba-NLP/gte-modernbert-base-embedding",
        hf_model_id="Alibaba-NLP/gte-modernbert-base",
        accelerator=Accelerator.L4,
        task=Embedder(),
        solution=HFTEI(),
    ),
    Deployment(  #
        name="google/embeddinggemma-300m",
        hf_model_id="google/embeddinggemma-300m",
        accelerator=Accelerator.L4,
        task=Embedder(),
        solution=HFTEI(),
    ),
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
        accelerator=Accelerator.T4,
        task=Embedder(),
        solution=HFTEI(),
    ),
    Deployment(  #
        name="nomic-ai/nomic-embed-text-v1.5",
        hf_model_id="nomic-ai/nomic-embed-text-v1.5",
        accelerator=Accelerator.A10G,
        task=Embedder(),
        solution=HFTEI(),
    ),
    Deployment(  #
        name="TaylorAI/bge-micro-v2",
        hf_model_id="TaylorAI/bge-micro-v2",
        accelerator=Accelerator.A10G,
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
        name="Alibaba-NLP/gte-Qwen2-7B-instruct-embedding",
        hf_model_id="Alibaba-NLP/gte-Qwen2-7B-instruct",
        accelerator=Accelerator.H100_40GB,
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
    Deployment(
        "mixedbread-ai/mxbai-embed-large-v1-embedding",
        "mixedbread-ai/mxbai-embed-large-v1",
        Accelerator.L4,
        Embedder(),
        solution=HFTEI(),
    ),
    Deployment(
        "Alibaba-NLP/gte-reranker-modernbert-base",
        "Alibaba-NLP/gte-reranker-modernbert-base",
        Accelerator.L4,
        Reranker(),
        solution=HFTEI(),
    ),
    Deployment(
        "nomic-ai/nomic-embed-text-v2-moe",
        "nomic-ai/nomic-embed-text-v2-moe",
        Accelerator.L4,
        Embedder(),
        solution=HFTEI(),
    ),
    Deployment(
        "redis/langcache-embed-v2",
        "redis/langcache-embed-v2",
        Accelerator.L4,
        Embedder(),
        solution=HFTEI(),
    ),
]
DEPLOYMENTS_BEI_BERT = []
for dep in DEPLOYMENTS_HFTEI:
    dep = copy.deepcopy(dep)
    dep.solution = BEIBert()
    if dep.accelerator == Accelerator.T4:
        dep.accelerator = Accelerator.L4
    DEPLOYMENTS_BEI_BERT.append(dep)


def llamalike_config(
    quant: TrussTRTLLMQuantizationType = TrussTRTLLMQuantizationType.FP8_KV,
    tp=1,
    repoid="meta-llama/Llama-3.3-70B-Instruct",
    batch_scheduler_policy: None = None,
    base_model: TrussTRTLLMModel = TrussTRTLLMModel.DECODER,
    calib_dataset: str = None,
):
    # config for meta-llama/Llama-3.3-70B-Instruct (FP8)
    build_kwargs = dict()
    runtime_kwargs = dict()
    if quant != TrussTRTLLMQuantizationType.NO_QUANT and tp in [1, 2]:
        if tp == 1:
            build_kwargs["num_builder_gpus"] = 4
    if quant == TrussTRTLLMQuantizationType.FP8_KV:
        build_kwargs["plugin_configuration"] = TrussTRTLLMPluginConfiguration(
            use_fp8_context_fmha=True
        )
    if batch_scheduler_policy:
        runtime_kwargs["batch_scheduler_policy"] = batch_scheduler_policy

    if calib_dataset is not None:
        build_kwargs["quantization_config"] = dict(calib_dataset=calib_dataset)

    config = TRTLLMConfiguration(
        build=TrussTRTLLMBuildConfiguration(
            base_model=base_model,
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
            **runtime_kwargs,
        ),
    )

    if quant in [
        TrussTRTLLMQuantizationType.WEIGHTS_INT4_KV_INT8,
    ]:
        config.build.plugin_configuration.use_paged_context_fmha = False
        config.build.plugin_configuration.use_fp8_context_fmha = False
        config.runtime.enable_chunked_context = False
    return config


def llamalike_lookahead(
    quant: TrussTRTLLMQuantizationType = TrussTRTLLMQuantizationType.FP8_KV,
    tp=1,
    repoid="meta-llama/Llama-3.3-70B-Instruct",
    use_dynamic_lengths: bool = False,
    **kwargs,
):
    config = llamalike_config(quant, tp, repoid, **kwargs)
    config.build.speculator = TrussSpeculatorConfiguration(
        # settings from https://arxiv.org/pdf/2402.02057
        speculative_decoding_mode="LOOKAHEAD_DECODING",
        lookahead_windows_size=3 if not use_dynamic_lengths else 1,
        lookahead_ngram_size=8 if not use_dynamic_lengths else 32,
        lookahead_verification_set_size=3 if not use_dynamic_lengths else 1,
        enable_b10_lookahead=True,  #
    )
    config.build.max_batch_size = 64
    return config


def llamalike_spec_dec(
    quant: TrussTRTLLMQuantizationType = TrussTRTLLMQuantizationType.FP8_KV,
    tp=1,
    repoid="meta-llama/Llama-3.3-70B-Instruct",
    spec_repo="meta-llama/Llama-3.2-1B-Instruct",
):
    config = llamalike_config(quant, tp, repoid)
    config.build.speculator = TrussSpeculatorConfiguration(
        speculative_decoding_mode="DRAFT_TOKENS_EXTERNAL",
        num_draft_tokens=10,
        checkpoint_repository=CheckpointRepository(
            repo=spec_repo,
            revision="main",
            source=CheckpointSource.HF,
        ),
    )
    config.build.max_batch_size = 64
    # the draft model and its kv cache live in the free memory of the target model's left-over KV cache
    config.runtime.kv_cache_free_gpu_mem_fraction = 0.45

    config_regular_hf = AutoConfig.from_pretrained(repoid, trust_remote_code=True)
    config_spec_hf = AutoConfig.from_pretrained(spec_repo, trust_remote_code=True)
    # verify vocab size is the same
    assert config_regular_hf.vocab_size == config_spec_hf.vocab_size, (
        f"vocab size mismatch for spec-dec: {config_regular_hf.vocab_size} vs {config_spec_hf.vocab_size}"
    )
    return config


def llamalike_config_v2(
    quant: TrussTRTLLMQuantizationType = TrussTRTLLMQuantizationType.FP8_KV,
    repoid="meta-llama/Llama-3.3-70B-Instruct",
    max_batch_size: int = 32,
    calib_size: Optional[int] = None,
):
    # config for meta-llama/Llama-3.3-70B-Instruct (FP8)
    build_kwargs = dict()
    runtime_kwargs = dict()

    if calib_size is not None:
        build_kwargs["quantization_config"] = dict(calib_size=calib_size)

    config = TRTLLMConfigurationV2(
        build=TrussTRTLLMBuildConfiguration(
            checkpoint_repository=CheckpointRepository(
                repo=repoid,
                revision="main",
                source=CheckpointSource.HF,
            ),
            quantization_type=quant,
            **build_kwargs,
        ),
        runtime=TRTLLMRuntimeConfigurationV2(
            max_seq_len=1000001,  # dummy for now
            max_batch_size=max_batch_size,
            **runtime_kwargs,
        ),
    )

    if quant in [
        TrussTRTLLMQuantizationType.WEIGHTS_INT4_KV_INT8,
    ]:
        config.build.plugin_configuration.use_paged_context_fmha = False
        config.build.plugin_configuration.use_fp8_context_fmha = False
        config.runtime.enable_chunked_context = False
    return config


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
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        Accelerator.B200,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="meta-llama/Llama-3.3-70B-Instruct",
                quant=TrussTRTLLMQuantizationType.FP4,
            )
        ),
    ),
    Deployment(
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        Accelerator.B200,
        TextGen(),
        solution=BISV2(
            trt_config=llamalike_config_v2(
                repoid="meta-llama/Llama-3.3-70B-Instruct",
                quant=TrussTRTLLMQuantizationType.FP4,
            )
        ),
    ),
    Deployment(
        "meta-llama/Llama-3.3-70B-Instruct-tp4",
        "meta-llama/Llama-3.3-70B-Instruct",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="meta-llama/Llama-3.3-70B-Instruct", tp=4
            )
        ),
    ),
    Deployment(
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        Accelerator.H100_40GB,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="meta-llama/Llama-3.2-3B-Instruct",
                tp=1,
                quant=TrussTRTLLMQuantizationType.NO_QUANT,
            )
        ),
    ),
    Deployment(
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        Accelerator.H100_40GB,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="meta-llama/Llama-3.2-3B-Instruct",
                tp=1,
                quant=TrussTRTLLMQuantizationType.FP8_KV,
            )
        ),
    ),
    Deployment(
        "meta-llama/Llama-3.2-3B-Instruct-calib-dataset",
        "meta-llama/Llama-3.2-3B-Instruct",
        Accelerator.H100_40GB,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="meta-llama/Llama-3.2-3B-Instruct",
                tp=1,
                quant=TrussTRTLLMQuantizationType.FP8_KV,
                calib_dataset="baseten/quant_calibration_dataset_v1",
            )
        ),
    ),
    Deployment(
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        Accelerator.H100_40GB,
        TextGen(),
        solution=BISV2(
            trt_config=llamalike_config_v2(
                repoid="meta-llama/Llama-3.2-3B-Instruct",
                quant=TrussTRTLLMQuantizationType.FP8_KV,
            )
        ),
    ),
    Deployment(
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        Accelerator.B200,
        TextGen(),
        solution=BISV2(
            trt_config=llamalike_config_v2(
                repoid="meta-llama/Llama-3.2-3B-Instruct",
                quant=TrussTRTLLMQuantizationType.FP4_MLP_ONLY,
            )
        ),
    ),
    Deployment(
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        Accelerator.H100_40GB,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="meta-llama/Llama-3.2-1B-Instruct",
                tp=1,
                quant=TrussTRTLLMQuantizationType.FP8_KV,
                batch_scheduler_policy="max_utilization",
            )
        ),
    ),
    Deployment(
        "google/gemma-3-270m-it",
        "google/gemma-3-270m-it",
        Accelerator.H100_40GB,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="google/gemma-3-270m-it",
                tp=1,
                quant=TrussTRTLLMQuantizationType.NO_QUANT,
                batch_scheduler_policy="max_utilization",
            )
        ),
    ),
    Deployment(
        "google/gemma-3-27b-it",
        "baseten/gemma-3-27b-causallm-it",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="baseten/gemma-3-27b-causallm-it",
                tp=1,
                quant=TrussTRTLLMQuantizationType.NO_QUANT,
                batch_scheduler_policy="max_utilization",
            )
        ),
    ),
    Deployment(
        "google/gemma-3-27b-it-speculative-lookahead",
        "baseten/gemma-3-27b-causallm-it",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_lookahead(
                repoid="baseten/gemma-3-27b-causallm-it",
                tp=1,
                quant=TrussTRTLLMQuantizationType.NO_QUANT,
                batch_scheduler_policy="max_utilization",
            )
        ),
    ),
    Deployment(
        "google/gemma-3-1b-it",
        "unsloth/gemma-3-1b-it",
        Accelerator.H100_40GB,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="unsloth/gemma-3-1b-it",
                tp=1,
                quant=TrussTRTLLMQuantizationType.NO_QUANT,
                batch_scheduler_policy="max_utilization",
            )
        ),
    ),
    Deployment(
        "Qwen/Qwen3-30B-A3B",
        "Qwen/Qwen3-30B-A3B",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="Qwen/Qwen3-30B-A3B",
                tp=1,
                quant=TrussTRTLLMQuantizationType.FP8,
                batch_scheduler_policy="max_utilization",
            )
        ),
    ),
    # Deployment(
    #     "Qwen/Qwen3-30B-A3B",
    #     "Qwen/Qwen3-30B-A3B",
    #     Accelerator.B200,
    #     TextGen(),
    #     solution=BISV2(
    #         trt_config=llamalike_config_v2(
    #             repoid="Qwen/Qwen3-30B-A3B",
    #             quant=TrussTRTLLMQuantizationType.FP8,
    #             calib_size=4096,
    #         )
    #     ),
    # ),
    Deployment(
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-32B",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="Qwen/Qwen3-32B",
                tp=1,
                quant=TrussTRTLLMQuantizationType.FP8_KV,
                batch_scheduler_policy="max_utilization",
            )
        ),
    ),
    Deployment(
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-32B",
        Accelerator.B200,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="Qwen/Qwen3-32B",
                tp=1,
                quant=TrussTRTLLMQuantizationType.FP4_KV,
                batch_scheduler_policy="max_utilization",
            )
        ),
    ),
    Deployment(
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-32B",
        Accelerator.B200,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="Qwen/Qwen3-32B",
                tp=1,
                quant=TrussTRTLLMQuantizationType.FP4_MLP_ONLY,
                batch_scheduler_policy="max_utilization",
            )
        ),
    ),
    Deployment(
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3-32B",
        Accelerator.B200,
        TextGen(),
        solution=BISV2(
            trt_config=llamalike_config_v2(
                repoid="Qwen/Qwen3-32B",
                quant=TrussTRTLLMQuantizationType.FP4_KV,
            )
        ),
    ),
    Deployment(
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-4B",
        Accelerator.H100,
        TextGen(),
        solution=BISV2(
            trt_config=llamalike_config_v2(
                repoid="Qwen/Qwen3-4B",
                quant=TrussTRTLLMQuantizationType.FP8_KV,
            )
        ),
    ),
    Deployment(
        "nvidia/Qwen3-8B-FP4",
        "nvidia/Qwen3-8B-FP4",
        Accelerator.B200,
        TextGen(),
        solution=BISV2(
            trt_config=llamalike_config_v2(
                repoid="nvidia/Qwen3-8B-FP4",
                quant=TrussTRTLLMQuantizationType.NO_QUANT,
            )
        ),
    ),
    Deployment(
        "nvidia/Qwen3-30B-A3B-FP4",
        "nvidia/Qwen3-30B-A3B-FP4",
        Accelerator.B200,
        TextGen(),
        solution=BISV2(
            trt_config=llamalike_config_v2(
                repoid="nvidia/Qwen3-30B-A3B-FP4",
                quant=TrussTRTLLMQuantizationType.NO_QUANT,
            )
        ),
    ),
    Deployment(
        "nvidia/Llama-3.1-8B-Instruct-FP4",
        "nvidia/Llama-3.1-8B-Instruct-FP4",
        Accelerator.B200,
        TextGen(),
        solution=BISV2(
            trt_config=llamalike_config_v2(
                repoid="nvidia/Llama-3.1-8B-Instruct-FP4",
                quant=TrussTRTLLMQuantizationType.NO_QUANT,
            )
        ),
    ),
    # Deployment(
    #     "Qwen/Qwen3-30B-A3B-Instruct-2507",
    #     "Qwen/Qwen3-30B-A3B-Instruct-2507",
    #     Accelerator.H100,
    #     TextGen(),
    #     solution=BISV2(
    #         trt_config=llamalike_config_v2(
    #             repoid="Qwen/Qwen3-30B-A3B-Instruct-2507",
    #             quant=TrussTRTLLMQuantizationType.FP8,
    #         )
    #     ),
    # ),
    # Deployment(
    #     "Qwen/Qwen3-30B-A3B-Instruct-2507",
    #     "Qwen/Qwen3-30B-A3B-Instruct-2507",
    #     Accelerator.B200,
    #     TextGen(),
    #     solution=Briton(
    #         trt_config=llamalike_config(
    #             repoid="Qwen/Qwen3-30B-A3B-Instruct-2507",
    #             quant=TrussTRTLLMQuantizationType.FP4,
    #             calib_dataset="baseten/quant_calibration_dataset_v1",
    #         )
    #     ),
    # ),
    Deployment(
        "meta-llama/Llama-3.1-405B",
        "meta-llama/Llama-3.1-405B",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(repoid="meta-llama/Llama-3.1-405B", tp=8)
        ),
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
    Deployment(
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        Accelerator.B200,
        TextGen(),
        solution=BISV2(
            trt_config=llamalike_config_v2(
                repoid="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                quant=TrussTRTLLMQuantizationType.FP4_KV,
            )
        ),
    ),
    # Qwen/Qwen2.5-72B-Instruct
    Deployment(
        "Qwen/Qwen2.5-72B-Instruct-tp2",
        "Qwen/Qwen2.5-72B-Instruct",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="Qwen/Qwen2.5-72B-Instruct",
                tp=2,
                quant=TrussTRTLLMQuantizationType.FP8,
            )
        ),
    ),
    Deployment(
        "Qwen/QwQ-32B-reasoning",
        "Qwen/QwQ-32B",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="Qwen/QwQ-32B",
                tp=1,
                quant=TrussTRTLLMQuantizationType.FP8,
            )
        ),
    ),
    Deployment(
        "Qwen/QwQ-32B-reasoning-with-speculative",
        "Qwen/QwQ-32B",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_lookahead(
                repoid="Qwen/QwQ-32B",
                tp=1,
                quant=TrussTRTLLMQuantizationType.FP8,
            )
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
    ),  # unsloth/phi-4
    Deployment(
        "microsoft/phi-4",
        "unsloth/phi-4",
        Accelerator.L4,
        TextGen(),
        solution=Briton(trt_config=llamalike_config(repoid="unsloth/phi-4", tp=2)),
    ),
    Deployment(
        "tiiuae/Falcon3-10B-Instruct",
        "tiiuae/Falcon3-10B-Instruct",
        Accelerator.L4,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(repoid="tiiuae/Falcon3-10B-Instruct", tp=2)
        ),
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
    Deployment(
        "Qwen/Qwen2.5-7B-Instruct-with-speculative-lookahead-decoding",
        "Qwen/Qwen2.5-7B-Instruct",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_lookahead(
                repoid="Qwen/Qwen2.5-7B-Instruct", quant=TrussTRTLLMQuantizationType.FP8
            )
        ),
    ),
    Deployment(
        "meta-llama/Llama-3.1-8B-Instruct-with-speculative-lookahead-decoding",
        "meta-llama/Llama-3.1-8B-Instruct",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_lookahead(repoid="meta-llama/Llama-3.1-8B-Instruct")
        ),
    ),
    Deployment(
        "Qwen/Qwen2.5-Coder-7B-Instruct-min-latency",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_lookahead(
                repoid="Qwen/Qwen2.5-Coder-7B-Instruct",
                quant=TrussTRTLLMQuantizationType.FP8,
                use_dynamic_lengths=True,
                base_model=TrussTRTLLMModel.DECODER,
            )
        ),
    ),
    Deployment(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        Accelerator.B200,
        TextGen(),
        solution=BISV2(
            trt_config=llamalike_config_v2(
                repoid="Qwen/Qwen2.5-Coder-7B-Instruct",
                quant=TrussTRTLLMQuantizationType.FP4,
            )
        ),
    ),
    Deployment(
        "Qwen/Qwen2.5-Coder-7B-Instruct-calib-dataset",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        Accelerator.B200,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_config(
                repoid="Qwen/Qwen2.5-Coder-7B-Instruct",
                quant=TrussTRTLLMQuantizationType.FP4_MLP_ONLY,
                calib_dataset="baseten/quant_calibration_dataset_v1",
            )
        ),
    ),
    Deployment(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        Accelerator.B200,
        TextGen(),
        solution=BISV2(
            trt_config=llamalike_config_v2(
                repoid="Qwen/Qwen2.5-Coder-7B-Instruct",
                quant=TrussTRTLLMQuantizationType.NO_QUANT,
            )
        ),
    ),
    Deployment(
        "Qwen/Qwen3-8B-min-latency",
        "Qwen/Qwen3-8B",
        Accelerator.H100,
        TextGen(),
        solution=Briton(
            trt_config=llamalike_lookahead(
                repoid="Qwen/Qwen3-8B",
                base_model=TrussTRTLLMModel.DECODER,
                use_dynamic_lengths=True,
            )
        ),
    ),
]


ALL_DEPLOYMENTS = DEPLOYMENTS_BEI + DEPLOYMENTS_BRITON + DEPLOYMENTS_BEI_BERT
ALL_DEPLOYMENTS_WITH_TEI = ALL_DEPLOYMENTS + DEPLOYMENTS_HFTEI

if __name__ == "__main__":
    for dp in ALL_DEPLOYMENTS_WITH_TEI:
        generate_deployment(dp)

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
- *speculative decoding* using lookahead decoding
- *fp8 quantization* on new GPUS such as H100, H200 and L4 GPUs

Examples:
{generation_names_fmt}
"""
    (Path(__file__).parent.parent / "README.md").write_text(readme)
    print(readme)
