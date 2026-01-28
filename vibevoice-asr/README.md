# VibeVoice-ASR Truss

This repository packages [VibeVoice-ASR](https://github.com/microsoft/VibeVoice) as a [Truss](https://truss.baseten.co/) using vLLM for inference.

VibeVoice-ASR is a unified speech-to-text model designed to handle **60-minute long-form audio** in a single pass, generating structured transcriptions containing **Who (Speaker), When (Timestamps), and What (Content)**, with support for **Customized Hotwords**.

## Key Features

- **🕒 60-minute Single-Pass Processing**: Accepts up to 60 minutes of continuous audio input within 64K token length, ensuring consistent speaker tracking and semantic coherence across the entire hour
- **👤 Customized Hotwords**: Users can provide customized hotwords (e.g., specific names, technical terms, or background info) to guide the recognition process
- **📝 Rich Transcription (Who, When, What)**: Jointly performs ASR, diarization, and timestamping, producing structured output that indicates who said what and when
- **🌐 Multilingual Support**: Natively supports over 50 languages
- **⚡ vLLM Inference**: Uses vLLM for faster inference with optimized batching and memory management

## Deployment

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`
3. Set up your Hugging Face access token as a secret in Baseten (required for model access)

With `vibevoice-asr` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## Model Inputs

The model uses the OpenAI-compatible chat completions API format. It accepts:

- **messages**: A list of message objects with:
  - `role`: "system" or "user"
  - `content`: Can include:
    - Audio URL: `{"type": "audio_url", "audio_url": {"url": "https://..."}}`
    - Text: `{"type": "text", "text": "Your instructions here"}`
- **stream**: Boolean (default: false) - Whether to stream the response
- **max_tokens**: Maximum tokens to generate (default: 8192)
- **temperature**: Sampling temperature (default: 0.0)
- **top_p**: Nucleus sampling parameter (default: 1.0)
- **repetition_penalty**: Penalty for repetition (default: 1.0)

## Invoking the Model

### Example: Basic Transcription

```python
import requests

response = requests.post(
    "https://model-<model-id>.api.baseten.co/development/predict",
    headers={"Authorization": "Api-Key YOUR_API_KEY"},
    json={
        "model": "microsoft/VibeVoice-ASR",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that transcribes audio input into text output in JSON format."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": "https://cdn.baseten.co/docs/production/Gettysburg.mp3"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Please transcribe it with these keys: Start time, End time, Speaker ID, Content"
                    }
                ]
            }
        ],
        "stream": False,
        "max_tokens": 8192,
        "temperature": 0.0
    }
)

print(response.json())
```

### Example: With Customized Hotwords

You can provide customized hotwords in the system message to improve accuracy for domain-specific content:

```python
response = requests.post(
    "https://model-<model-id>.api.baseten.co/development/predict",
    headers={"Authorization": "Api-Key YOUR_API_KEY"},
    json={
        "model": "microsoft/VibeVoice-ASR",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that transcribes audio input into text output in JSON format. Important terms: Baseten, Truss, vLLM, Hugging Face"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": "https://your-audio-url.com/recording.mp3"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Please transcribe this audio with speaker diarization and timestamps."
                    }
                ]
            }
        ],
        "max_tokens": 8192,
        "temperature": 0.0
    }
)
```

### Example: Using Truss CLI

```sh
truss predict -d '{
  "model": "microsoft/VibeVoice-ASR",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant that transcribes audio input into text output in JSON format."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "audio_url",
          "audio_url": {
            "url": "https://cdn.baseten.co/docs/production/Gettysburg.mp3"
          }
        },
        {
          "type": "text",
          "text": "Please transcribe it with these keys: Start time, End time, Speaker ID, Content"
        }
      ]
    }
  ],
  "stream": false,
  "max_tokens": 8192,
  "temperature": 0.0
}'
```

## Output Format

The model returns structured transcriptions in JSON format with:
- **Start time**: Beginning timestamp of the segment
- **End time**: Ending timestamp of the segment
- **Speaker ID**: Identifier for the speaker
- **Content**: Transcribed text content

## Configuration

This Truss is configured with:
- **Base Image**: `vllm/vllm-openai:v0.14.0`
- **Model**: `microsoft/VibeVoice-ASR`
- **Max Model Length**: 65536 tokens (supports up to 60 minutes of audio)
- **Max Batched Tokens**: 32768
- **GPU Memory Utilization**: 0.8
- **Accelerator**: H100 (recommended)
- **Predict Concurrency**: 64

## Requirements

- Hugging Face access token (set as `hf_access_token` secret in Baseten)
- H100 GPU recommended for optimal performance
- Audio files must be accessible via URL (supported formats: MP3, WAV, etc.)

## Resources

- [VibeVoice GitHub Repository](https://github.com/microsoft/VibeVoice)
- [VibeVoice-ASR Documentation](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-asr.md)
- [Hugging Face Model](https://huggingface.co/microsoft/VibeVoice-ASR)
- [Truss Documentation](https://truss.baseten.co)

## License

This project is licensed under the MIT License, as per the [VibeVoice repository](https://github.com/microsoft/VibeVoice).
