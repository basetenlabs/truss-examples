"""
Validate BEI-Bert NER deployment against HuggingFace Transformers ground truth.

Tests:
  1. Token-level predictions match transformers (100% label agreement expected)
  2. Aggregated entity spans with aggregation_strategy="max"
  3. Latency benchmark (mean + p99 over 30 timed calls after 10-call warmup)
  4. Throughput benchmark (256 sentences in 32 concurrent requests)

Usage:
  pip install torch transformers baseten-performance-client
  python test_ner_integration.py --base-url https://model-xxxxxx.api.baseten.co/environments/production/sync
"""

import argparse
import statistics
import time

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from baseten_performance_client import PerformanceClient

MODEL_PATH = "dslim/bert-base-NER-uncased"

TEST_TEXTS = [
    "Apple is looking at buying U.K. startup for $1 billion",
    "John works at Google in Mountain View, California",
    "The Eiffel Tower is in Paris, France",
]

AGGREGATION_TEXTS = [
    "Apple is looking at buying U.K. startup for $1 billion",
    "John works at Google in Mountain View, California",
    "j.d. vance is living in New York City",
]

# Expected aggregated entities for smoke-test validation
EXPECTED_AGGREGATED = {
    "Apple is looking at buying U.K. startup for $1 billion": [
        ("Apple", "ORG"),
        ("U.K.", "LOC"),
    ],
    "John works at Google in Mountain View, California": [
        ("John", "PER"),
        ("Google", "ORG"),
        ("Mountain View", "LOC"),
        ("California", "LOC"),
    ],
    "j.d. vance is living in New York City": [
        ("j.d. vance", "PER"),
        ("New York City", "LOC"),
    ],
}


def load_hf_model(model_path: str):
    print(f"Loading HuggingFace model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model, model.config.id2label


def hf_token_predictions(text: str, tokenizer, model, label_list):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    labels = [label_list[p.item()] for p in predictions[0]]
    return tokens, labels


def test_token_accuracy(client: PerformanceClient, tokenizer, model, label_list):
    print("\n" + "=" * 72)
    print("TEST 1: Token-level prediction accuracy vs HuggingFace Transformers")
    print("=" * 72)

    total_tokens = 0
    total_matches = 0

    for text in TEST_TEXTS:
        print(f"\nText: '{text}'")

        bei_response = client.batch_post(
            "/predict_tokens",
            payloads=[{"inputs": [[text]], "raw_scores": False}],
        )
        bei_data = bei_response.data[0][0]

        hf_tokens, hf_labels = hf_token_predictions(text, tokenizer, model, label_list)

        bei_tokens = [t["token"] for t in bei_data]
        bei_labels = [
            max(t["results"].items(), key=lambda x: x[1])[0] for t in bei_data
        ]

        assert len(bei_tokens) == len(hf_tokens), (
            f"Token count mismatch: BEI={len(bei_tokens)} HF={len(hf_tokens)}"
        )

        fmt = f"  {'Token':<20} {'BEI':<12} {'HF':<12} {'Match'}"
        print(fmt)
        print("  " + "-" * 55)

        for bei_tok, bei_lbl, hf_tok, hf_lbl in zip(
            bei_tokens, bei_labels, hf_tokens, hf_labels
        ):
            match = "✓" if bei_lbl == hf_lbl else "✗"
            print(f"  {bei_tok:<20} {bei_lbl:<12} {hf_lbl:<12} {match}")
            total_tokens += 1
            if bei_lbl == hf_lbl:
                total_matches += 1

    accuracy = total_matches / total_tokens if total_tokens > 0 else 0
    print(f"\nOverall: {total_matches}/{total_tokens} tokens match ({accuracy:.1%})")
    assert accuracy == 1.0, f"Expected 100% accuracy, got {accuracy:.1%}"
    print("PASS ✓")


def test_aggregation_strategy(client: PerformanceClient):
    print("\n" + "=" * 72)
    print("TEST 2: Aggregated entity spans (aggregation_strategy='max')")
    print("=" * 72)

    bei_response = client.batch_post(
        "/predict_tokens",
        payloads=[
            {
                "inputs": [[t] for t in AGGREGATION_TEXTS],
                "raw_scores": False,
                "aggregation_strategy": "max",
            }
        ],
    )

    for text, entities in zip(AGGREGATION_TEXTS, bei_response.data[0]):
        expected = EXPECTED_AGGREGATED[text]
        print(f"\nText: '{text}'")
        print(f"  {'Span':<25} {'Label':<10} {'Score':<8} {'Expected?'}")
        print("  " + "-" * 55)

        actual = [(e["token"], list(e["results"].keys())[0]) for e in entities]

        for entity in entities:
            span = entity["token"]
            label = list(entity["results"].keys())[0]
            score = list(entity["results"].values())[0]
            in_expected = (span, label) in expected
            print(
                f"  {span:<25} {label:<10} {score:<8.4f} {'✓' if in_expected else '?'}"
            )

        # Check all expected entities are present
        for exp_span, exp_label in expected:
            assert (exp_span, exp_label) in actual, (
                f"Expected entity ({exp_span!r}, {exp_label!r}) not found in: {actual}"
            )

    print("\nPASS ✓")


def benchmark_latency(
    client: PerformanceClient, text: str, warmup: int = 10, timed: int = 30
):
    print("\n" + "=" * 72)
    print("TEST 3: Latency benchmark (single sentence)")
    print("=" * 72)

    times = []
    for i in range(warmup + timed):
        resp = client.batch_post(
            "/predict_tokens",
            payloads=[
                {"inputs": [[text]], "raw_scores": False, "aggregation_strategy": "max"}
            ],
        )
        if i >= warmup:
            times.append(resp.total_time)

    mean_ms = statistics.mean(times) * 1000
    p99_ms = sorted(times)[int(len(times) * 0.99)] * 1000

    print(f"  Warmup: {warmup} calls | Timed: {timed} calls")
    print(f"  Mean latency: {mean_ms:.1f} ms  (incl network)")
    print(f"  P99 latency:  {p99_ms:.1f} ms  (incl network)")
    print("  Target: mean < 10ms on same-region client")


def benchmark_throughput(client: PerformanceClient, text: str):
    print("\n" + "=" * 72)
    print("TEST 4: Throughput benchmark (256 sentences in 32 concurrent requests)")
    print("=" * 72)

    payloads = [
        {"inputs": [[text]] * 8, "raw_scores": False, "aggregation_strategy": "max"}
    ] * 32

    start = time.time()
    resp = client.batch_post("/predict_tokens", payloads=payloads)
    elapsed = time.time() - start

    total_sentences = 32 * 8
    qps = total_sentences / elapsed
    print(f"  {total_sentences} sentences in {elapsed:.2f}s → {qps:.0f} sentences/sec")


def main():
    parser = argparse.ArgumentParser(description="Validate BEI-Bert NER deployment")
    parser.add_argument(
        "--base-url",
        required=True,
        help="BEI sync endpoint, e.g. https://model-xxxxxx.api.baseten.co/environments/production/sync",
    )
    parser.add_argument(
        "--api-key-env",
        default="BASETEN_API_KEY",
        help="Environment variable name for the Baseten API key (default: BASETEN_API_KEY)",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip HuggingFace accuracy comparison (faster, no torch/transformers needed)",
    )
    args = parser.parse_args()

    import os

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise EnvironmentError(f"Set {args.api_key_env} environment variable")

    client = PerformanceClient(api_key=api_key, base_url=args.base_url)

    if not args.skip_hf:
        tokenizer, model, label_list = load_hf_model(MODEL_PATH)
        test_token_accuracy(client, tokenizer, model, label_list)
    else:
        print("Skipping HuggingFace accuracy comparison (--skip-hf)")

    test_aggregation_strategy(client)
    benchmark_latency(client, TEST_TEXTS[0])
    benchmark_throughput(client, TEST_TEXTS[0])

    print("\n" + "=" * 72)
    print("All tests passed.")
    print("=" * 72)


if __name__ == "__main__":
    main()
