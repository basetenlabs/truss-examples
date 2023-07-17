# MusicGen Truss

This repository packages [MusicGen](https://github.com/facebookresearch/audiocraft/) as a [Truss](https://truss.baseten.co).

MusicGen is a simple and controllable model for music generation developed by Facebook AI Research.

Utilizing this model for inference can be challenging given the hardware requirements. With Baseten and Truss, inference is dead simple.

## Deploying MusicGen

We found this model runs reasonably fast on A10Gs; you can configure the hardware you'd like in the config.yaml.

```yaml
---
resources:
  cpu: "3"
  memory: 14Gi
  use_gpu: true
  accelerator: A10G
```

Before deployment:

1. Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup).
2. Install Truss and the Baseten Python client: `pip install --upgrade baseten truss`
3. Authenticate your development environment with `baseten login`

Deploying the Truss is easy; simply load it and push from a Python script:

```python
import baseten
import truss

musicgen_truss = truss.load('.')
baseten.deploy(musicgen_truss)
```

## Invoking MusicGen

MusicGen takes a list of prompts and a duration in seconds. It will generate one clip per prompt and return each clip as a base64 encoded WAV file.

```python
import baseten

model = baseten.deployed_model_id("YOUR MODEL ID")
model_output = model.predict({"prompts": ["happy rock" "energetic EDM", "sad jazz"], "duration": 8})

import base64

for idx, clip in enumerate(model_output["data"]):
  with open(f"clip_{idx}.wav", "wb") as f:
    f.write(base64.b64decode(clip))
```

You can also invoke your model via a REST API

```
curl -X POST " https://app.baseten.co/models/YOUR_MODEL_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompts": ["happy rock" "energetic EDM", "sad jazz"], "duration": 8
         }'
```

## Model sizes

MusicGen supports four model sizes:

- `small`: 300M model, text to music only
- `medium`: 1.5B model, text to music only
- `melody`: 1.5B model, text to music and text+melody to music
- `large`: 3.3B model, text to music only

This truss can been configured to run the large size but you can easily select other versions by changing the `MODEL_SIZE` constant in `model/model.py`.
