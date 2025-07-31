# GLM 4.5 Air FP8 (SGLang)

This example shows how to deploy [GLM 4.5 Air in FP8](https://huggingface.co/zai-org/GLM-4.5-Air-FP8) using SGLang on Baseten. This model requires 4xH100 to access the full context window.

## Deploying the model

This model can be deployed to Baseten using Truss:

```
pip install --upgrade truss
truss push --publish z-ai/glm-4-5-air-fp8
```

## Calling the model

This model is OpenAI compatible and can be called using the OpenAI client.

```python
import os
from openai import OpenAI

# https://model-XXXXXXX.api.baseten.co/environments/production/sync/v1
model_url = ""

client = OpenAI(
    base_url=model_url,
    api_key=os.environ.get("BASETEN_API_KEY"),
)

stream = client.chat.completions.create(
    model="baseten",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write FizzBuzz."}
    ],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```
