# Orpheus 3B WebSocket TTS

Real-time text-to-speech over WebSocket using [Orpheus 3B](https://huggingface.co/baseten/orpheus-3b-0.1-ft) with TensorRT-LLM and SNAC audio decoding.

The model accepts text over a persistent WebSocket connection and streams back raw PCM audio bytes (16-bit, 24kHz, mono). It uses sentence-aware buffering to produce natural-sounding speech with low latency.

## Deploy

Install the Truss CLI and push:

```bash
pip install truss
truss push
```

This builds a TensorRT-LLM engine with FP8 quantization on an H100 GPU. The engine build takes roughly 15 minutes on first deploy.

## Call the model

The WebSocket endpoint expects two phases:

1. **Metadata (JSON):** Send connection parameters like voice, token limits, and buffer size.
2. **Text (strings):** Send words one at a time. Send `__END__` to signal the end of input.

The server streams back binary PCM audio frames as each sentence completes.

### Python client

Install dependencies:

```bash
pip install aiohttp pyaudio python-dotenv
```

Set your API key and model ID in `call.py`, then run:

```bash
python call.py
```

The client connects, sends text word-by-word, and plays audio through your speakers as it arrives.

## Configuration

Key settings in `config.yaml`:

- `runtime.transport.kind: websocket`: Enables WebSocket transport instead of HTTP.
- `resources.accelerator: H100_40GB`: Required for TensorRT-LLM engine build and inference.
- `trt_llm.build.quantization_type: fp8_kv`: FP8 quantization for lower memory usage and faster inference.

## Connection parameters

The first message sent over the WebSocket must be a JSON object:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voice` | string | `"tara"` | Speaker voice for synthesis. |
| `max_tokens` | int | `6144` | Maximum tokens per generated audio segment. |
| `temperature` | float | `0.6` | Sampling temperature. |
| `top_p` | float | `0.8` | Nucleus sampling threshold. |
| `repetition_penalty` | float | `1.3` | Penalizes repeated tokens. |
| `buffer_size` | int | `10` | Words buffered before forced flush. |
