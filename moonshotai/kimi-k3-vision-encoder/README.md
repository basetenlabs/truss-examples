# moonshotai/Kimi-K3 vision encoder

Deploy the Rust HTTP server with a C++ LibTorch backend using a prebuilt image.
This serves visual embeddings for a downstream model; it does not run the language
model or generate text.

## Deploy

Add a Baseten secret named `hf_access_token` with access to
[moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3), then run from this repository:

```sh
truss push moonshotai/kimi-k3-vision-encoder
```

The configuration uses **RTX_PRO_6000:2**, eight CPUs, and 16 GiB of host memory.
Two visible GPUs are detected automatically and each gets a persistent model worker,
with shared preprocessing and output workers. This is replication, not tensor parallelism.
The pinned image is used without building a custom image (`no_build: true`).
Baseten mounts the selected checkpoint files at `/models/kimi-k3`; the server
does not download weights itself.

The launcher selects the host CUDA driver or bundled compatibility libraries at
startup. No cluster-specific `LD_LIBRARY_PATH` override is needed. The GPU and
host driver must still support the selected CUDA runtime.

## Predict

Send the JSON example in `config.yaml` to `/generate` using POST or PUT.
Successful responses are **MessagePack**, not JSON. With
`allow_bytes_without_b64: true`, embedding payloads contain binary bytes; decode
with a MessagePack client rather than `response.json()`. The server listens on
port 8000 and exposes `/health` for readiness and liveness.

Only JSON configuration files and the specified encoder weight shards are mounted.
The shard patterns are specific to this checkpoint layout; review them if changing
the checkpoint or revision.
