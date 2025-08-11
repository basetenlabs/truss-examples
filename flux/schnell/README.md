# b10_tcache example

https://docs.baseten.co/development/model/b10-tcache

b10‑tcache + torch.compile example (Flux image generation)

Speed up cold starts for a diffusers Flux pipeline on Baseten by caching PyTorch compilation artifacts with b10‑tcache. This example shows how to:

Compile the heavy parts of the pipeline with torch.compile

Warm up the model across supported resolutions

Load a previously saved compile cache on startup and save it after first warmup

Return a base64‑encoded JPEG image from predict

This repo is an illustrative example you can adapt to your own diffusion / vision models.

How it works

Load (and later save) compile cache via b10_tcache:

from b10_tcache import load_compile_cache, save_compile_cache

cache_loaded = load_compile_cache()
# ... compile & warm up ...
if not cache_loaded:
    save_compile_cache()

Compile the hotspots of the pipeline with Torch Dynamo/TorchInductor:

self.pipe.transformer = torch.compile(
    self.pipe.transformer, mode="max-autotune-no-cudagraphs", dynamic=False
)
self.pipe.vae.decode = torch.compile(
    self.pipe.vae.decode, mode="max-autotune-no-cudagraphs", dynamic=False
)

Warm up with dummy prompts across every resolution you intend to serve (so later requests hit already‑compiled kernels):

for width, height in [(1024, 1024), (1216, 832), (896, 1152)]:
    self.pipe(prompt="dummy prompt", prompt_2=None, guidance_scale=0.0,
              max_sequence_length=256, num_inference_steps=4,
              width=width, height=height, output_type="pil",
              generator=generator)
    # run twice to exercise more code paths

On subsequent cold starts, load_compile_cache() restores previously compiled artifacts and dramatically reduces compile latency.

Project structure

.
├── model.py        # Model server entrypoint (Baseten)
├── config.yaml     # Truss/Baseten config (add b10-tcache to requirements)
└── README.md       # This file

Prerequisites

Baseten account with access to b10cache/compile caching features

GPU deployment (e.g., L4)

Hugging Face access token for gated repos (saved as hf_access_token secret)

Python deps include torch, diffusers, transformers, Pillow, b10-tcache

config.yaml snippets

Add the cache helper to your requirements:

requirements:
  - b10-tcache
  - torch
  - diffusers
  - transformers
  - pillow

If you are pulling weights from Hugging Face, include the secret:

secrets:
  hf_access_token: null

Optionally, use model weight caching (recommended for large models) with model_cache:; see Baseten docs for full examples.

The Model class

__init__(**kwargs)

Reads secrets and config.model_metadata.repo_id (your HF repo id)

Creates placeholders for the pipeline

load()

Initialize your pipeline (example):

from diffusers import FluxPipeline
self.pipe = FluxPipeline.from_pretrained(
    self.model_name, torch_dtype=torch.bfloat16, token=self.hf_access_token
).to("cuda")

Try to load a previously saved compile cache.

Compile the transformer and VAE decode.

Warm up across the resolutions you plan to support.

Save the compile cache (first run only).

predict(model_input)

Inputs (JSON):

prompt (string, required)

prompt2 (string, optional)

width (int, default 1024)

height (int, default 1024)

seed (int, optional; random if omitted)

guidance_scale (float, default 0.0) — Flux.1‑schnell requires 0.0

num_inference_steps (int, default 4) — schnell is timestep‑distilled

max_sequence_length (int, default 256) — overly long prompts are truncated

Output:

{ "data": "<base64‑encoded JPEG>" }

Notes

If guidance_scale != 0.0, it is reset to 0.0 with a warning.

Long prompts are truncated to max_sequence_length tokens.

We include a simple nvidia-smi call in load() for debugging GPU visibility in logs.

Example requests

Minimal

{
  "prompt": "a cozy reading nook with warm sunlight"
}

With size and seed

{
  "prompt": "a watercolor painting of a fox in a misty forest",
  "width": 1216,
  "height": 832,
  "seed": 42
}

Dual‑prompt

{
  "prompt": "a modern living room, scandinavian style",
  "prompt2": "with large windows and houseplants"
}

Tips & customization

Resolutions: Add every (width, height) pair you plan to support to the warmup loop so those kernels are compiled ahead of time.

Compile mode: max-autotune-no-cudagraphs works well for diffusion models; you can experiment with other modes if you profile your workload.

Cache hit logic: We save the cache only if load_compile_cache() returned False. If you prefer, you can always call save_compile_cache() at the end of warmup.

Model weights: Pair this with model_cache to prefetch HF weights and minimize time‑to‑first‑token.

Troubleshooting

Cache not loading: Ensure b10-tcache is listed in requirements and that your account has b10cache enabled. Check deployment logs for confirmation.

Still slow on first pod: The very first run must compile & warm up; later pods benefit from the saved cache.

HF auth: For gated repos, confirm hf_access_token secret is set and the token has read access.

Guidance scale warnings: FLUX.1‑schnell only supports guidance_scale=0.0.

Acknowledgements

Built for Baseten’s Torch Compile Cache API and b10cache.

Uses Hugging Face Diffusers FluxPipeline and PyTorch torch.compile.

License

MIT
