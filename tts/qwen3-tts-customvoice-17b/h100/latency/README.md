# Qwen3 TTS 12Hz CustomVoice Streaming 1.7B

Qwen3-TTS 1.7B CustomVoice from the Qwen team: streaming text-to-speech with nine
built-in speakers and natural-language style control (`instructions`), in Chinese,
English, Japanese, Korean, German, French, Russian, Portuguese, Spanish and Italian.
This preset is tuned for time to first audio on one H100.

| Speaker | Description | Native language |
|---|---|---|
| `vivian` | Bright, slightly edgy young female voice | Chinese |
| `serena` | Warm, gentle young female voice | Chinese |
| `uncle_fu` | Seasoned male voice with a low, mellow timbre | Chinese |
| `dylan` | Youthful Beijing male voice | Chinese (Beijing dialect) |
| `eric` | Lively Chengdu male voice | Chinese (Sichuan dialect) |
| `ryan` | Dynamic male voice with strong rhythmic drive | English |
| `aiden` | Sunny American male voice with a clear midrange | English |
| `ono_anna` | Playful Japanese female voice | Japanese |
| `sohee` | Warm Korean female voice | Korean |

Every speaker can speak every supported language; each sounds best in its native one.

## Example usage

The deployment is WebSocket-only. Open one connection per conversation, send
`session.config` once, then send `input.text` and `input.done` for each utterance.
`input.done` flushes the utterance and keeps the connection open, so later
utterances skip the handshake.

```python
#!/usr/bin/env python3
"""Stream speech from a Qwen3-TTS CustomVoice deployment on Baseten.

Set BASETEN_API_KEY in your environment before running.

    python call.py --text "Hello! How are you today?" --voice ryan
    python call.py --text "I can't believe it!" --voice aiden --instructions "Very excited"

Requirements:
    pip install websockets
"""

import argparse
import asyncio
import json
import os
import time
import wave

import websockets

WS_URL = "wss://model-<model_id>.api.baseten.co/environments/production/websocket"
SAMPLE_RATE = 24000


async def synthesize(text: str, voice: str, instructions: str | None, language: str | None,
                     output: str) -> None:
    headers = {"Authorization": f"Api-Key {os.environ['BASETEN_API_KEY']}"}
    async with websockets.connect(WS_URL, additional_headers=headers, max_size=None) as ws:
        config = {
            "type": "session.config",
            "task_type": "CustomVoice",
            "voice": voice,
            # Raw 16-bit mono PCM at 24 kHz, sent as soon as each chunk is decoded.
            "response_format": "pcm",
            "stream_audio": True,
        }
        if instructions:
            config["instructions"] = instructions
        if language:
            config["language"] = language
        await ws.send(json.dumps(config))

        t0 = time.perf_counter()
        await ws.send(json.dumps({"type": "input.text", "text": text}))
        await ws.send(json.dumps({"type": "input.done"}))

        pcm = bytearray()
        ttfa = None
        while True:
            msg = await ws.recv()
            if isinstance(msg, bytes):
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                pcm += msg
                continue
            event = json.loads(msg)
            if event["type"] == "session.done":
                break
            if event["type"] == "error":
                raise RuntimeError(event["message"])

    with wave.open(output, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(bytes(pcm))
    print(f"TTFA {ttfa * 1000:.0f} ms, {len(pcm) / 2 / SAMPLE_RATE:.2f} s of audio -> {output}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--voice", default="vivian")
    p.add_argument("--instructions", default=None, help="Style control, e.g. 'Speak slowly and calmly'")
    p.add_argument("--language", default=None, help="e.g. English, Chinese; omit to auto-detect")
    p.add_argument("--output", "-o", default="output.wav")
    args = p.parse_args()
    asyncio.run(synthesize(args.text, args.voice, args.instructions, args.language, args.output))


if __name__ == "__main__":
    main()
```

### Session options

| Field | Default | Notes |
|---|---|---|
| `voice` | `vivian` | One of the speakers above (case-insensitive). |
| `instructions` | none | Free-form style control, such as "Speak in a calm, reassuring tone". |
| `language` | auto | `English`, `Chinese`, `Japanese`, ... Setting it skips language detection. |
| `stream_audio` + `response_format: pcm` | off, `wav` | Turn both on for the lowest TTFA; WAV is sent per sentence instead of per chunk. |
| `split_granularity` | `none` | `sentence` or `clause` starts synthesis at each boundary while text is still arriving, for LLM token streams. |
| `speed` | `1.0` | Playback speed multiplier. |

Send another `session.config` between utterances to switch speaker or style. Send
`{"type": "session.close"}` to end the connection.

## Deploy your own CustomVoice checkpoint

This preset can serve a fine-tuned Qwen3-TTS CustomVoice checkpoint (for example, one with
your own speakers) in place of the stock weights. Everything else, including the latency
tuning, stays the same.

1. Clone this directory and install the [Baseten CLI](https://docs.baseten.co/quickstart).
2. If the Hugging Face repo is private or gated, create a Hugging Face read token and save it
   in your Baseten workspace as a secret named `hf_access_token`
   ([Secrets](https://app.baseten.co/settings/secrets)).
3. In `config.yaml`, point the `weights` block at your repo:

   ```yaml
   weights:
     - source: "hf://<your-org>/<your-checkpoint>@<commit-sha>"
       # Keep this path: data/do.sh and docker_server.start_command load the model from it.
       mount_location: "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"
       # Only needed for a private or gated repo.
       auth:
         auth_method: CUSTOM_SECRET
         auth_secret_name: "hf_access_token"
   ```

   Pin a commit SHA (or omit `@<commit-sha>` to track the default branch while iterating).
   Keep the `secrets: hf_access_token` entry in `config.yaml`.
4. If your checkpoint doesn't include the `vivian` speaker, set the warm-up speaker to one
   it does have, or the start-up warm-up requests fail (the server still starts):

   ```yaml
   environment_variables:
     WARMUP_VOICE: <one-of-your-speakers>
   ```

   Also update `model_metadata.example_model_input.voice`.
5. Run `baseten model push` from inside the directory. It creates a deployment in your
   workspace; push again after each config change to get a new deployment.

The checkpoint must keep the stock CustomVoice layout so vLLM-Omni can load it:

- `config.json` with `"tts_model_type": "custom_voice"` and the `Qwen3TTSForConditionalGeneration`
  architecture.
- The `speech_tokenizer/` directory (codec config and weights) next to `model.safetensors`.
- The tokenizer files (`vocab.json`, `merges.txt`, `tokenizer_config.json`).

Speaker names come from `talker_config.spk_id` in `config.json`. Clients pass one of those
names as `voice` (case-insensitive); any other name is rejected with the list of valid
speakers. A Base (voice-clone) checkpoint won't work here: vLLM-Omni reads the variant from
`tts_model_type` and rejects CustomVoice requests on a Base checkpoint. Use the Qwen3-TTS Base
presets for those.

## Concurrency

`predict_concurrency: 32`. Measured on one H100 through the Baseten endpoint with
`qwen_tts_bench.py` (closed loop: each agent holds one WebSocket and synthesizes short
voice-agent replies back to back for 60 s, `vivian`, PCM streaming):

| Agents | TTFA p50 | TTFA p95 | RTF p50 | RTF p95 | Stalled | Audio × real time |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.10 s | 0.10 s | 0.10 | 0.13 | 0% | 10× |
| 4 | 0.11 s | 0.13 s | 0.12 | 0.15 | 0% | 34× |
| 8 | 0.12 s | 0.15 s | 0.15 | 0.17 | 0% | 55× |
| 16 | 0.16 s | 0.21 s | 0.20 | 0.25 | 0% | 78× |
| 24 | 0.20 s | 0.27 s | 0.26 | 0.31 | 0% | 93× |
| **32** | **0.24 s** | **0.33 s** | **0.31** | **0.38** | **0%** | **104×** |
| 48 | 0.47 s | 0.60 s | 0.43 | 0.62 | 39% | 110× |
| 64 | 0.75 s | 0.90 s | 0.55 | 1.15 | 99% | 109× |

"Stalled" is the share of utterances with a playback gap over 100 ms if playback starts
at the first chunk. The GPU saturates around 110× real time, so past 32 agents latency
grows without adding throughput. Agents never pause between replies, so a real voice
agent (which only needs TTS while the bot is talking) supports more calls per replica
than the agent count. Set the autoscaling concurrency target around 24 to keep headroom.

Reproduce:

```bash
python3 qwen_tts_bench.py --url wss://model-<model_id>.api.baseten.co/environments/production/websocket \
    --api-key $BASETEN_API_KEY --concurrencies 1,4,8,16,24,32,48,64
```

## Notes

- Serving stack: vLLM-Omni v0.30.0 (same image as the Qwen3-TTS Base presets), with
  the talker and code2wav stages on one GPU and async chunk streaming.
- Latency tuning in `data/qwen3_tts_deploy.yaml`: a codec chunk ramp of 2, 2, 4, 8,
  16 and then 25 frames, so the first audio leaves after two 80 ms frames and later
  chunks arrive before playback runs dry; 512-token prefill steps; CUDA graphs on
  both stages; and prefix caching for the shared speaker prompt. Silence banning is
  off, because it targets voice clones.
- `do.sh` restores the shared torch.compile cache (`b10-compile-cache`) before
  start-up and sends two warm-up syntheses once the server is healthy.
- Bench: registry CI's `qwen3_tts` protocol drives a voice-clone session, which a
  CustomVoice checkpoint rejects, so the CI smoke bench can't measure this preset.
  The numbers above come from `qwen_tts_bench.py` in this directory.