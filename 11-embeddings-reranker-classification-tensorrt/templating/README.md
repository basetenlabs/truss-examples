# Generating examples

Edit model entries and shared templates in `generate_templates.py`, then regenerate the examples. Do not edit generated configs or READMEs directly.

From a checkout named `truss-examples`, with `truss`, `transformers`, and `requests` installed:

```bash
python 11-embeddings-reranker-classification-tensorrt/templating/generate_templates.py \
  --model BEI-Bert-michaelfeil-laya-typed-decisions \
  --model BEI-Bert-michaelfeil-rune-26b-a4b \
  --model BEI-Bert-qwen-qwen3-embedding-0.6b \
  --model BEI-Bert-perplexity-ai-pplx-decider-v1-27b \
  --model BEI-Bert-cloudflare-clef
```

Repeat `--model` to select examples, or omit it to regenerate all examples. The category index is regenerated in either case.
