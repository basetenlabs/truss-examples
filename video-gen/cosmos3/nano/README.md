# NVIDIA Cosmos 3 Nano (8B)

NVIDIA Cosmos 3 is a world foundation model (WFM) that unifies understanding and generation under a single Mixture-of-Transformer (MoT) architecture: a vision-language Reasoner tower and a world-simulator Generator tower share latent representations, so structured perception grounds realistic, temporally consistent simulation.

This deployment is the **Nano (8B-param)** variant. Supports text2image, text2video, image2video, plus action-conditioned modalities (forward dynamics, inverse dynamics, policy). ~32 GB VRAM, single H100.

- **Upstream:** <https://github.com/nvidia-cosmos/cosmos3>
- **Weights:** `nvidia-cosmos-ea/Cosmos3-Nano` (gated NDA, NVIDIA Software and Model Evaluation License until GA on 2026-05-31)
- **Linear:** [LABS-95](https://linear.app/baseten/issue/LABS-95/nvidia-cosmos-3)

## Base image

`baseten/cosmos3-nano:v1` is a private Docker Hub image built from `nvidia-cosmos/cosmos3-ea-external/Dockerfile` at upstream SHA `61fb84f`. It bakes in:

- Python 3.13 + uv-installed `cu130` group (torch 2.10+cu130, flash-attn-3-nv, natten 0.21.6.dev6 with libnatten, ...)
- The gated `cosmos3` Python package source (NDA-licensed; image must stay private until the 2026-05-31 GA)

After GA, rebuild the image off the public `nvidia-cosmos/cosmos3` repo and flip it public.

## Modalities

Cosmos3-Nano supports all six omni modes. The truss accepts a `model_mode` field on the request:

| `model_mode` | Required fields | Output |
|---|---|---|
| `text2image` | `prompt`, `num_frames: 1` | `vision.jpg` (base64) |
| `text2video` (default) | `prompt` | `vision.mp4` (base64) |
| `image2video` | `prompt`, `vision_path` (URL or container path) | `vision.mp4` |
| `forward_dynamics` | `prompt`, `vision_path`, `action_path`, `action_chunk_size`, `domain_name`, `image_size` | `vision.mp4` (predicted rollout) |
| `inverse_dynamics` | `prompt`, `vision_path`, `raw_action_dim`, `image_size` | `sample_outputs.action` |
| `policy` | `prompt`, `vision_path`, `raw_action_dim`, `image_size`, `domain_name` | `sample_outputs.action` (+ optional `vision.*`) |

`vision_path` and `action_path` are paths the container can read (URL or absolute path on disk). `action_path` points at a JSON file holding the action sequence; format matches NVIDIA's example inputs at `cosmos3/inputs/omni/action_*.json`.

## Sample requests

Each example is built from `nvidia-cosmos/cosmos3-ea-external/inputs/omni/*.json` (the upstream reference inputs). Action-mode examples reproduce NVIDIA's payloads verbatim; vision-mode prompts are shortened for readability — substitute your own at runtime. `vision_path` / `action_path` can be either a URL the container can fetch or an absolute path inside the container.

### `text2image`

```json
{
  "model_mode": "text2image",
  "prompt": "A medium shot of a modern robotics research laboratory with white walls and a gray floor.",
  "resolution": "720",
  "aspect_ratio": "16,9",
  "num_frames": 1,
  "seed": 0
}
```

### `text2video`

```json
{
  "model_mode": "text2video",
  "prompt": "A bustling city street at night, neon signs reflecting on wet pavement, light rain falling.",
  "resolution": "720",
  "aspect_ratio": "16,9",
  "num_frames": 189,
  "fps": 24,
  "seed": 0
}
```

### `image2video`

```json
{
  "model_mode": "image2video",
  "vision_path": "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/vision/robot_153.jpg",
  "prompt": "A robotic arm picks up the red spherical object and places it on a lower shelf, completing a smooth deliberate manipulation.",
  "resolution": "720",
  "aspect_ratio": "16,9",
  "num_frames": 189,
  "fps": 24,
  "seed": 0
}
```

### `forward_dynamics` (action → predicted future video)

```json
{
  "model_mode": "forward_dynamics",
  "vision_path": "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_0.mp4",
  "action_path": "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_0.json",
  "prompt": "Put the pot to the left of the purple item. This video is captured from a first-person perspective looking at the scene.",
  "image_size": 480,
  "fps": 5,
  "num_steps": 30,
  "guidance": 1.0,
  "shift": 5.0,
  "seed": 0,
  "action_chunk_size": 16,
  "domain_name": "bridge_orig_lerobot"
}
```

### `inverse_dynamics` (video → predicted actions)

```json
{
  "model_mode": "inverse_dynamics",
  "vision_path": "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/av_vision_25_73d01c91-51f0-46cf-9b76-5682a76fb349.mp4",
  "prompt": "You are an autonomous vehicle planning system. This video is captured from a first-person perspective looking at the scene.",
  "image_size": 480,
  "fps": 10,
  "num_steps": 30,
  "guidance": 1.0,
  "shift": 5.0,
  "seed": 0,
  "raw_action_dim": 9,
  "action_chunk_size": 60,
  "domain_name": "av"
}
```

### `policy` (observation + goal → predicted actions)

```json
{
  "model_mode": "policy",
  "vision_path": "https://github.com/nvidia-cosmos/cosmos-dependencies/raw/refs/heads/assets/cosmos3/inputs/action/bridge_0.mp4",
  "prompt": "Put the pot to the left of the purple item. This video is captured from a first-person perspective looking at the scene.",
  "image_size": 480,
  "fps": 5,
  "num_steps": 30,
  "guidance": 1.0,
  "shift": 5.0,
  "seed": 0,
  "action_chunk_size": 16,
  "raw_action_dim": 10,
  "domain_name": "bridge_orig_lerobot"
}
```

## Response shape

Vision-emitting modes (`text2image`, `text2video`, `image2video`, `forward_dynamics`):

```json
{
  "name": "<sample-name>",
  "model_mode": "text2video",
  "format": "mp4",
  "data": "<base64>"
}
```

Action-emitting modes (`inverse_dynamics`, `policy`) — `data` is the input video echoed back from `download()`; the generated tensor lives under `sample_outputs.outputs[0].content.action`:

```json
{
  "name": "<sample-name>",
  "model_mode": "policy",
  "format": "mp4",
  "data": "<base64 of input video>",
  "sample_outputs": {
    "status": "success",
    "outputs": [
      {
        "content": {"action": [[...], [...]]},
        "files": [...]
      }
    ]
  }
}
```

## Validation (NVIDIA golden tests)

Cosmos3 ships golden reference outputs for `inverse_dynamics` and `policy`. MSE against the golden tensors on this deployment (computed against `golden_action_path` from each example's `extra` block):

| Mode | Our MSE | Threshold | Verdict |
|---|---|---|---|
| `inverse_dynamics` | 6.2e-5 | ≤ 0.05 | PASS |
| `policy` | 0.0132 | ≤ 0.05 | PASS |

`forward_dynamics` ships `golden_psnr_min: 14.0` but no published golden video, so PSNR verification is visual only.