# Qwen3 ASR 1.7B

This example shows how to call a Baseten deployment using the OpenAI Python SDK to run **Qwen/Qwen3-ASR-1.7B** on an audio URL.

## Prerequisites

- Python 3.9+
- OpenAI Python SDK installed:

```bash
pip install openai
```

## Example: Transcribe an audio URL

```python
from openai import OpenAI

model_id = ""  # place model ID here

client = OpenAI(
    api_key="BASETEN-API-KEY",
    base_url=f"https://model-{model_id}.api.baseten.co/environments/production/sync/v1"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-ASR-1.7B",
    stream=False,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url":
                        {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav"}

                }
            ]
        }
    ],
)

print(response.choices[0].message.content)
```

## Sample Output
```txt
Response: language English<asr_text>Uh huh. Oh yeah, yeah. He wasn't even that big when I started listening to him, but and his solo music didn't do overly well, but he did very well when he started writing for other people.

```

