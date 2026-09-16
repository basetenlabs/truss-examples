# Zerank 2 Reranker on BEI

Serve [ZeroEntropy's zerank-2-reranker](https://huggingface.co/zeroentropy/zerank-2-reranker), an Apache-2.0, English, 4B-parameter text reranker fine-tuned from Qwen3-4B, with Baseten Embeddings Inference (BEI). This config is the registry copy of the one serving the ZeroEntropy lab listing `zeroentropy/zerank-2-reranker`: one `H100`, FP8 quantization with the deployed calibration settings, a 40,960-token batch budget, batch size 256, and the deployed runtime and plugin settings, following the [BEI sequence-classification examples](https://github.com/basetenlabs/truss-examples/tree/main/11-embeddings-reranker-classification-tensorrt). ZeroEntropy documents a 32k trained context; `max_seq_len` is left to the checkpoint's 40,960 limit as in the deployed config.

## Checkpoint and scoring

The original checkpoint declares `Qwen3ForCausalLM` and scores a (query, document) pair through the next-token logit of the relevance token `Yes` (id 9454, from its `1_LogitScore` sentence-transformers head). BEI needs a sequence-classification head, so this preset pins the public [baseten-admin/zerank-2-reranker-seq conversion](https://huggingface.co/baseten-admin/zerank-2-reranker-seq/tree/ac4b5f3c452cf22a917aa6b93b0a92c750db284a) at `ac4b5f3c452cf22a917aa6b93b0a92c750db284a`. It declares `Qwen3ForSequenceClassification` with one label, `Yes`, and a `score` head that copies the tied embedding row for that token, so its logit equals the original relevance score by construction (see the conversion's README). The repo is public and ungated, so no `hf_access_token` is declared. BEI manages checkpoint caching and engine construction; no custom Python or remote-code execution is needed in the serving path. BEI runtime and builder versions are managed by Baseten.

Upstream returns raw `Yes` logits and documents `(scores / 5).sigmoid()` as the calibrated 0–1 score. **The conversion does not include division by five.** Request `raw_scores: true` and apply `sigmoid(raw_score / 5)` on the client. Sort scores descending to rank documents. BEI's default sigmoid would use a different calibration.

This preset relies on the conversion's stated construction; the element-wise head comparison done for Zerank 1 Small (#463) was not repeated here. FP8 engine quantization can change scores; evaluate ranking quality on your own data before production use.

## Call the model

POST to `https://model-MODEL_ID.api.baseten.co/environments/production/sync/predict`. The request accepts `inputs` containing fully formatted prompt strings; the response is a list of predictions per input, each with `label` and `score`. This preset exposes classification through `/predict`, so the client constructs each query/document prompt and sorts the results.

The playground example matches the deployed listing: plain strings with `truncate: true` and `truncation_direction: Right`. For best fidelity to how the model was trained, format each input with the pinned tokenizer's chat template instead: the query in a `query` message and the document in a `document` message, with `add_generation_prompt=True`. The template renders the query as the `system` turn and the document as the `user` turn. Do not substitute another Qwen reranker template or add an instruction. When you format prompts yourself, send `truncate: false` and reject prompts exceeding 40,960 tokens instead of letting truncation remove the assistant suffix.

```bash
pip install requests transformers
```

```python
import math
import os

import requests
from transformers import AutoTokenizer

TOKENIZER_REPO = "baseten-admin/zerank-2-reranker-seq"
REVISION = "ac4b5f3c452cf22a917aa6b93b0a92c750db284a"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_REPO, revision=REVISION)

query = "What is 2+2?"
documents = ["4", "The answer is definitely 1 million"]
inputs = []
for document in documents:
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "query", "content": query},
            {"role": "document", "content": document},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if len(tokenizer.encode(prompt, add_special_tokens=False)) > 40960:
        raise ValueError("Formatted query/document exceeds the 40,960-token limit")
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

The config's playground example sends two untemplated strings, as the deployed listing does, and returns the unscaled raw logit for each.

## Relationship to the ZeroEntropy lab listing

This preset is the registry copy of the config that serves the ZeroEntropy lab listing `zeroentropy/zerank-2-reranker`. It is deploy-only and does not publish a listing of its own: the README declares no `library_id`, so registry CI does not create a separate `baseten/zerank-2-reranker-throughput` listing beside the lab-owned one.

## Validation

Registry PR CI deploys this configuration, calls its playground example, runs the `/predict` classification smoke benchmark, and deactivates the temporary deployment. The benchmark measures serving performance with synthetic inputs; it does not measure retrieval quality or verify the upstream score calibration.