# Voxtral-Mini-4B-Realtime-2602

Deploy Voxtral-Mini-4B-Realtime-2602 on Baseten using a vLLM engine.

| Property | Value |
|----------|-------|
| Model | [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) |
| Task | Audio |
| Engine | vLLM |
| GPU | H100_40GB:1 |

## Deploy

> **Note:** This model requires a HuggingFace access token. Set `hf_access_token` in your Baseten secrets before deploying.

```sh
truss push
```

## Invoke

This model uses a WebSocket endpoint for realtime audio streaming. Use the included `streaming.py` client:

```sh
python streaming.py
```

Or connect directly via WebSocket:

```python
import websockets

async with websockets.connect(
    "wss://model-<model_id>.api.baseten.co/environments/production/websocket",
    extra_headers={"Authorization": "Api-Key YOUR_BASETEN_API_KEY"}
) as ws:
    await ws.send('{"type": "session.update", "model": "mistralai/Voxtral-Mini-4B-Realtime-2602"}')
    await ws.send('{"type": "input_audio_buffer.append", "audio": "<base64_encoded_pcm16_audio>"}')
    await ws.send('{"type": "input_audio_buffer.commit"}')
```

## Configuration highlights

- Base image: `vllm/vllm-openai:nightly-d88a1df699f68e5284fe3a3170f8ae292a3e9c3f`
- System packages: `python3.10-venv, ffmpeg, openmpi-bin, libopenmpi-dev`
- Environment variables: `VLLM_DISABLE_COMPILE_CACHE`
