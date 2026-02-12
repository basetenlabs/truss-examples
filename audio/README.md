# Audio Models

Truss configurations for speech-to-text, text-to-speech, music generation, and multimodal audio models. Includes both batch and streaming deployments.

| Directory | Variants | Description |
|-----------|----------|-------------|
| [whisper](whisper/) | 8 | OpenAI Whisper speech-to-text models including Faster Whisper, WhisperX, and streaming variants |
| [kokoro](kokoro/) | 1 | Kokoro text-to-speech model |
| [chatterbox-tts](chatterbox-tts/) | 1 | Chatterbox text-to-speech model |
| [piper-tts](piper-tts/) | 1 | Piper lightweight text-to-speech engine |
| [xtts-v2](xtts-v2/) | 1 | Coqui XTTS v2 multilingual text-to-speech |
| [xtts-streaming](xtts-streaming/) | 1 | Coqui XTTS with streaming audio output |
| [orpheus-3b-websockets](orpheus-3b-websockets/) | 1 | Orpheus 3B TTS with WebSocket streaming |
| [orpheus-best-performance](orpheus-best-performance/) | 1 | Orpheus TTS optimized for lowest latency |
| [sesame-csm-1b](sesame-csm-1b/) | 1 | Sesame CSM 1B conversational speech model |
| [metavoice-1b](metavoice-1b/) | 1 | MetaVoice 1B text-to-speech model |
| [ultravox](ultravox/) | 1 | Ultravox multimodal audio-language model |
| [audiogen-medium](audiogen-medium/) | 1 | Meta AudioGen medium audio generation from text |
| [musicgen-large](musicgen-large/) | 1 | Meta MusicGen large music generation |
| [musicgen-melody](musicgen-melody/) | 1 | Meta MusicGen melody-conditioned music generation |
| [nvidia-parakeet](nvidia-parakeet/) | 1 | NVIDIA Parakeet automatic speech recognition |
| [qwen-asr](qwen-asr/) | 1 | Qwen automatic speech recognition model |
| [qwen-omni](qwen-omni/) | 1 | Qwen Omni multimodal audio model |
| [qwen-omni-thinker](qwen-omni-thinker/) | 1 | Qwen Omni Thinker reasoning audio model |
| [voxtral-streaming-4b](voxtral-streaming-4b/) | 1 | Mistral Voxtral 4B streaming speech model |

## Deploying

Each audio model can be deployed to Baseten with:

```bash
truss push
```
