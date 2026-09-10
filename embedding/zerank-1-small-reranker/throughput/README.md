# Zerank 1 Small Reranker on BEI

Serve [ZeroEntropy's Zerank 1 Small Reranker](https://huggingface.co/zeroentropy/zerank-1-small-reranker), an Apache-2.0, English, 1.7B-parameter text reranker, with Baseten Embeddings Inference (BEI). This preset uses one L4 (`L4:4x16`) and FP8 quantization, following the [BEI sequence-classification examples](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt). The engine is built with a 32,768-token sequence limit and batch token budget. The unquantized weights occupy approximately 3.44 GB; an L4 provides 24 GB of GPU memory for weights, activations, and runtime buffers.

## Checkpoint and scoring

The original checkpoint declares `Qwen3ForCausalLM` and implements reranking through custom Python. BEI needs a sequence-classification head, so this preset pins the existing [fred-baseten/zerank-1-small-seq conversion](https://huggingface.co/fred-baseten/zerank-1-small-seq/tree/65292825034a60e01fb33bf3c437ababc5ca5c26) at `65292825034a60e01fb33bf3c437ababc5ca5c26`. It declares `Qwen3ForSequenceClassification`, with one label, `Yes`, and FP16 weights before BEI quantization. BEI manages checkpoint caching and engine construction; no custom Python or remote-code execution is needed in the serving path. BEI runtime and builder versions are managed by Baseten.

The [upstream implementation](https://huggingface.co/zeroentropy/zerank-1-small-reranker/blob/a65fd51c450e9b47fdddab98e31166ecad21af8d/modeling_zeranker.py) scores the final prompt token as `sigmoid(Yes_logit / 5)`. The converted `score.weight` was checked against the upstream tied output-embedding row for `Yes` (token 9454): all 2,048 elements match within FP16 rounding (maximum absolute difference `2.98e-8`). **The conversion does not include division by five.** Request `raw_scores: true` and apply `sigmoid(raw_score / 5)` on the client. Sort scores descending to rank documents. BEI's default sigmoid would use a different calibration.

This head check does not establish full checkpoint or end-to-end numerical equivalence. FP16 conversion and FP8 engine quantization can change scores; evaluate ranking quality on your own data before production use.

## Call the model

POST to `https://model-MODEL_ID.api.baseten.co/environments/production/sync/predict`. The request accepts `inputs` containing fully formatted prompt strings; the response is a list of predictions per input, each with `label` and `score`. This preset exposes classification through `/predict`, so the client constructs each query/document prompt and sorts the results.

Use the pinned tokenizer's chat template with the query as the system message, the document as the user message, and `add_generation_prompt=True`. Preserve its exact escaping; do not replace it with another Qwen reranker template or add an instruction. Upstream clips queries to 2,000 characters and documents to 10,000 characters before stripping whitespace; the example reproduces that behavior. Reject prompts exceeding 32,768 tokens instead of truncating away the assistant suffix.

```bash
pip install requests transformers
```

```python
import math
import os

import requests
from transformers import AutoTokenizer

TOKENIZER_REPO = "fred-baseten/zerank-1-small-seq"
REVISION = "65292825034a60e01fb33bf3c437ababc5ca5c26"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, revision=REVISION)

query = "What is 2+2?"
documents = ["4", "The answer is definitely 1 million"]
inputs = []
for document in documents:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": query[:2000].strip()},
            {"role": "user", "content": document[:10000].strip()},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if len(tokenizer.encode(prompt, add_special_tokens=False)) > 32768:
        raise ValueError("Formatted query/document exceeds the 32,768-token limit")
    inputs.append(prompt)

response = requests.post(
    f"https://model-{os.environ['MODEL_ID']}.api.baseten.co/environments/production/sync/predict",
    headers={"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"},
    json={"inputs": inputs, "raw_scores": True, "truncate": False},
    timeout=120,
)
response.raise_for_status()


def sigmoid(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


scores = [sigmoid(predictions[0]["score"] / 5) for predictions in response.json()]
ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
print(ranked)  # (original document index, relevance score)
```

The config's playground example contains the exact pinned template output for `What is 2+2?` / `4` and returns the unscaled raw logit.

## Validation

Registry PR CI deploys this configuration, calls its playground example, runs the `/predict` classification smoke benchmark, and deactivates the temporary deployment. The benchmark measures serving performance with synthetic inputs; it does not measure retrieval quality or verify the upstream score calibration.