# Decision 3.0 D3 27B

Deploy [vLLM Semantic Router's d3](https://huggingface.co/vllm-sr/d3)
as a SystemOne HTTP server on one NVIDIA H100.
The model evaluates text or JSON with optional images and videos, returning
Choice, Noul (yes/no) and Score probabilities without generating text.

## Deploy

From the repository root, authenticate with the intended Baseten workspace:

```bash
uvx truss login --remote <remote-name>
uvx truss push classifiers/d3/h100x1-bf16/default --remote <remote-name>
```

The checkpoint is public and requires no Hugging Face access-token secret.
Wait until the deployment is active before calling it.

## Call

Set `BASETEN_API_KEY` and `MODEL_ID` for the deployment:

```bash
curl --fail-with-body -sS \
  "https://model-${MODEL_ID}.api.baseten.co/environments/production/predict" \
  -H "Authorization: Api-Key ${BASETEN_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "state": "The parcel arrived damaged. Please send a replacement today.",
    "questions": {
      "route": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {"delivery": "Damaged parcels", "billing": "Payments and invoices"}
      },
      "urgent": {"type": "noul", "instructions": "Does the customer request action today?"},
      "priority": {"type": "score", "instructions": "How urgent is this?",
                   "criteria": ["Routine", "Soon", "Today"]}
    }
  }'
```

The response contains `model`, `answers` keyed by question ID, and `usage`.
Choice answers include `choice`, `confidence` and candidate `probabilities`;
Noul answers include `noul`; Score answers include `score`, `confidence`,
`legend` and level `probabilities`. Token counts and probabilities vary by input.
See the [upstream contract](https://huggingface.co/vllm-sr/d3#quickstart).

For `POST /v1/systemone`, use the same payload plus `"model": "d3"`
at `https://model-${MODEL_ID}.api.baseten.co/environments/production/sync/v1/systemone`.
The repository alias `vllm-sr/d3` is also accepted. `GET /v1/models`
is available at the same `/sync` prefix. Keep the Baseten authentication header.

### Images and videos

Add `images` and/or `videos` lists to either endpoint. Each item must be an
uploaded base64 data URL; every question sees all attached media. For example,
set `images` to `["data:image/png;base64,<base64-encoded-image>"]` or `videos`
to `["data:video/mp4;base64,<base64-encoded-video>"]`. Replace the placeholder
with the actual file encoded as base64. `state` may be omitted for media-only requests.
File paths and remote URLs are rejected by the HTTP adapter.

Upstream's strict decoders accept PNG, JPEG and WebP images (at most 8 MB and
16 million source pixels each), and MP4, WebM, QuickTime and Matroska videos
(at most 32 MB, 300 seconds and 8,294,400 source pixels per frame).
Images are resized to at most 1,638,400 pixels. Videos are sampled at 2 fps,
with at most 32 frames per video, 200,704 pixels per sampled frame, and
16,384 video tokens across the request.

## Runtime and limits

- Weights, processors, tokenizer and inference code are pinned together to
  Hugging Face revision `093cffbaa7be6f5dfe14252ae7e80c396c06dbd9`.
  Every manifest-listed file is mounted; upstream verifies all SHA-256 hashes
  with `verify="full"`. D3 accepts mount symlinks and extra mount metadata,
  so it loads directly from `/models/d3`.
- `AutoModel.from_pretrained()` uses `trust_remote_code=True` and
  `local_files_only=True`. The pinned checkpoint's Python code executes at
  startup. Dependencies include Transformers 5.17.0 and the matching
  Torch 2.8.0 / Torchvision 0.23.0 pair.
- Inference preserves the upstream BF16 backbone and FP32 readout. Optional
  flash-linear-attention kernels are omitted; the runtime can use its PyTorch
  reference path. Upstream AMD timings are not performance measurements for
  this NVIDIA deployment.
- The adapter serializes HTTP inference and sets `batch_size=1` to limit
  simultaneous question memory. Choice supports 2–255 candidates; Score
  supports 2–10 ordered levels; at most 1,024 questions are accepted per request.
- The checkpoint declares no fixed input-token cap. Available GPU memory
  limits request size; the Decision 2 limit of 8,192 tokens does not apply.
  Upstream input and media errors return HTTP 422; invalid model outputs return
  HTTP 500.
- Startup verifies that both image and video processors are available and
  warms up a text question before `/health` becomes ready. `/metrics` exposes
  Python process metrics.

## Validation

CPU adapter and loader tests (no checkpoint weights required):

```bash
PYTHONPATH=classifiers/d3/h100x1-bf16/default uv run --python 3.11 \
  --with-requirements classifiers/d3/h100x1-bf16/default/tests/requirements.txt \
  pytest classifiers/d3/h100x1-bf16/default/tests
```

Registry CI runs the enrolled [classifier JevBench profiles](https://github.com/basetenlabs/model-registry/blob/main/benchmarking/classifiers/README.md).
These measure text decision quality on the pinned 72-case corpus; they do not
measure the full [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index),
vision/video quality, or throughput. GPU deployment and text/image/video
prediction validation are pending; CPU tests use a fake model.