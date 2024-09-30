# Whisper V3 Turbo

Whisper V3 Turbo is a variant of Whisper Large with:

- 8x faster relative speed vs Whisper Large
- 4x faster than Medium
- 2x faster than Small
- 809M parameters
- Full multilingual support
- Minimal degradation in accuracy

This Whisper Large implementation uses TensorRT-LLM via our [TensorRT-LLM Engine Builder](https://docs.baseten.co/performance/examples/whisper-trt).

To deploy it to Baseten:

```
pip install --upgrade truss
truss push --publish
```

For inference, run:

```python
import requests
import os

# Model ID for production deployment
model_id = ""
# Read secrets from environment variables
baseten_api_key = os.environ["BASETEN_API_KEY"]

# Call model endpoint
resp = requests.post(
    f"https://model-{model_id}.api.baseten.co/production/predict",
    headers={"Authorization": f"Api-Key {baseten_api_key}"},
    json={
      "url": "https://www2.cs.uic.edu/~i101/SoundFiles/gettysburg10.wav",
    }
)

print(resp.content.decode("utf-8"))
```
