# b10‑transfer + torch.compile example (Flux image generation)

Speed up cold starts for a diffusers Flux pipeline on Baseten by caching PyTorch compilation artifacts with b10‑tcache. This example shows how to:

- Compile the heavy parts of the pipeline with torch.compile

- Warm up the model across supported resolutions

- Load a previously saved compile cache on startup and save it after first warmup

In this case, we brought the cold start time from 900s to 70s.

See the full docs on [b10_tcache](https://docs.baseten.co/development/model/b10-tcache).

## Prerequisites

Add the cache helper to your requirements in `config.yaml`:

```yaml
requirements:
  - b10-transfer
```

## How it works

Load (and later save) compile cache via b10_tcache:

```python
from b10_transfer import load_compile_cache, save_compile_cache, OperationStatus

cache_loaded = load_compile_cache()

if cache_loaded == OperationStatus.ERROR:
    logging.info("Run in eager mode, skipping torch compile")
else:
    self.compile()

if cache_loaded == OperationStatus.DOES_NOT_EXIST:
    save_compile_cache()
```

Compile the hotspots of the pipeline with Torch Dynamo/TorchInductor:

```python
self.pipe.transformer = torch.compile(
    self.pipe.transformer, mode="max-autotune-no-cudagraphs", dynamic=False
)
self.pipe.vae.decode = torch.compile(
    self.pipe.vae.decode, mode="max-autotune-no-cudagraphs", dynamic=False
)
```

Warm up with dummy prompts across every resolution you intend to serve (so later requests hit already‑compiled kernels):

```python
for width, height in [(1024, 1024), (1216, 832), (896, 1152)]:
    self.pipe(
        prompt="dummy prompt",
        prompt_2=None,
        guidance_scale=0.0,
        max_sequence_length=256,
        num_inference_steps=4,
        width=width,
        height=height,
        output_type="pil",
        generator=generator,
    )
    self.pipe(
        prompt="extra dummy prompt",
        prompt_2=None,
        guidance_scale=0.0,
        max_sequence_length=256,
        num_inference_steps=4,
        width=width,
        height=height,
        output_type="pil",
        generator=generator,
    )
```

On subsequent cold starts, load_compile_cache() restores previously compiled artifacts and dramatically reduces compile latency.
