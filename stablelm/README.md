[![Deploy to Baseten](https://user-images.githubusercontent.com/2389286/236301770-16f46d4f-4e23-4db5-9462-f578ec31e751.svg)](https://app.baseten.co/explore/stablelm)

# StableLM Truss

This repository packages [StableLM](https://github.com/Stability-AI/StableLM) as a [Truss](https://truss.baseten.co).

## Deploying StableLM

Stability AI recently announced the ongoing development of the StableLM series of language models, and simultaneously released a number of checkpoints for this model.

Utilizing these models for inference can be challenging given the hardware requirements. With Baseten and Truss, this can be dead simple. You can see the full code repository here.

There are four models that were released:
* "stabilityai/stablelm-base-alpha-7b"
* "stabilityai/stablelm-tuned-alpha-7b"
* "stabilityai/stablelm-base-alpha-3b"
* "stabilityai/stablelm-tuned-alpha-3b"

You can modify the `load` method in `model.py` to select the version you'd like to deploy.

``` python
model_name = "stabilityai/stablelm-tuned-alpha-7b" #@param ["stabilityai/stablelm-base-alpha-7b", "stabilityai/stablelm-tuned-alpha-7b", "stabilityai/stablelm-base-alpha-3b", "stabilityai/stablelm-tuned-alpha-3b"]
```

We found this model runs reasonably fast on A10Gs; you can configure the hardware you'd like in the `config.yaml`.

```yaml
...
resources:
  cpu: "3"
  memory: 14Gi
  use_gpu: true
  accelerator: A10G
...
```
The usual GPT-style parameters will pass right through to the inference point:

* max_new_tokens (_default_: 64)
* temperature (_default_: 0.5)
* top_p (_default_: 0.9)
* top_k (_default_: 0)
* num_beams (_default_: 4)

If the tuned versions are needed for use in Chatbots; prepend the input message with the system prompt as described in the StableLM Readme:

```python
system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

prompt = f"{system_prompt}<|USER|>What's your mood today?<|ASSISTANT|>"
```

Deploying the Truss is easy; simply load it and push.

```python
import baseten
import truss

stablelm_truss = truss.load('.')
baseten.deploy(stablelm_truss)
```
