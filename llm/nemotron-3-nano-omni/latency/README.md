# Nemotron 3 Nano Omni — latency preset

NVIDIA Nemotron 3 Nano Omni Reasoning 30B A3B served via vLLM (from the NGC container
`nvcr.io/0767305323357365/n3-nano-omni/nemotron-3-nano-omni-reasoning-30b-a3b`) on a single
H100 80GB. Supports text, image, video (with optional in-video audio), and audio inputs via
the OpenAI-compatible chat completions endpoint.

## Access

- **Container image**: private on NGC (`nvcr.io`). Requires the `DOCKER_REGISTRY_nvcr.io`
  secret on the deploying Baseten account (base64-encoded `$oauthtoken:<NGC_API_KEY>`).
- **Weights**: mirrored to the private HF repo `baseten-admin/nemotron-3-nano-omni-ga`.
  Requires the `hf_access_token` secret on the deploying Baseten account.

## Reasoning mode

Reasoning (chain-of-thought) is on by default for text and image inputs. For video and audio
inputs pass `chat_template_kwargs: {enable_thinking: false}` in the request — reasoning is
not supported with video/audio in the EA release.

## Recommended generation settings

- Non-thinking: `temperature=0.2`
- Thinking: `temperature=0.6`, `top_p=0.95`