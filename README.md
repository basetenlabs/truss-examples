# Truss Examples

Truss is the simplest way to serve AI/ML models in production. 

To get you started with Truss, this repository has dozens of example models, each ready to deploy as-is or adapt to your needs. Check out models like [Falcon-40B](falcon-40b-truss), [Starcoder](starcoder-truss), and [Stable Diffusion](stable-diffusion-truss) for inspiration.

## Installation

Get the repository with:

```
git clone https://github.com/basetenlabs/truss-examples
```

Install Truss with:

```
pip install --upgrade truss baseten
```

## Deployment

Log in with a [Baseten API key](https://app.baseten.co/settings/account/api_keys):

```
baseten login
```

Pick a model to deploy by passing a path to that model. 

```python
import truss
import baseten

# Use these variables to pick and name a model
model_path = "wizardlm-truss"
model_name = "WizardLM"

model = truss.load(model_path)
baseten.deploy(model, model_name=model_name)
```

## Invocation

Invocation depends on the model's input and output specifications. See individual model READMEs for invocation details.
