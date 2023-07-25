# Pygmalion 6B Truss

This repository packages [Pygmalion 6B](https://huggingface.co/PygmalionAI/pygmalion-6b) as a [Truss](https://truss.baseten.co).

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp)). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.

## Setup

[Sign up](https://app.baseten.co/signup) or [sign in](https://app.baseten.co/login/) to your Baseten account and create an [API key](https://app.baseten.co/settings/account/api_keys).

Then run:

```
pip install --upgrade baseten
baseten login
```

Paste your API key when prompted.

## Deployment

In an iPython notebook, run the following script to deploy Pygmalion 6B to your Baseten account:

```python
import baseten
import truss

pygmalion = truss.load("truss-examples/pygmalion-6b-truss/")
baseten.deploy(
  pygmalion,
  model_name="Pygmalion 6B",
)
```

Once your Truss is deployed, you can start using the Pygmalion 6B model through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.


### Hardware notes

This seven billion parameter model is running in `float16` so that it fits on an A10G.

## Pygmalion 6B API documentation

This section provides an overview of the Pygmalion 6B API, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided prompt. Note that this Truss is _stateless_, so you need to maintain chat history yourself.

### API route: `predict`

The predict route is the primary method for generating text completions based on a given prompt. It takes several parameters:

- __character_name__: The name of your character.
- __new_message__: The message your user (or yourself) is sending in the current conversation.
- __persona__ (optional, default=""): A high-level description of the character.
- __greeting__ (optional, default=""): How the character and user initially interact. This shouldn't be dialogue, but rather physical actions.
- __scenario__ (optional, default=""): Setting for the conversation.
- __dialogue_history__ (optional, default=""): The current history of dialogue between the character and user.
- __example_dialogue__ (optional, default=[]): An optional example of what a conversation between the character and the user might lookl lik.

## Example usage

You can use the `baseten` model package to invoke your model from Python

```python
import baseten
# You can retrieve your deployed model version ID from the Baseten UI
model = baseten.deployed_model_version_id('YOUR_MODEL_ID')

request = {
    "character_name": "Barack Obama",
    "persona": "Barack Obama is the first black, charismatic and composed 44th President of the United States. He is well-respected for his leadership during a time of economic crisis and for his efforts to improve healthcare and relations with foreign nations. He is a skilled orator and is known for his ability to bring people together with his speeches. Despite facing opposition, he remains steadfast in his beliefs and is dedicated to making the world a better place.",
    "greeting": "You approach Barack Obama, a tall and distinguished-looking man with a warm smile. He greets you with a firm handshake and a nod of his head.",
    "scenario": "You meet Obama in the White House Oval Office. He is sitting on his chair.",
    "example_dialogue": ["You: Can you tell me about your time as President?\nBarack Obama: During my time as President, I faced many challenges. The country was in the midst of an economic crisis, and I worked tirelessly to turn things around. I also passed the Affordable Care Act, which has helped millions of Americans access quality healthcare. I also made strides in improving our relations with foreign nations, particularly with Cuba and Iran.\n\nYou: What do you consider to be your greatest accomplishment as President?\nBarack Obama: That's a tough question. I'm proud of the work we did to stabilize the economy and provide healthcare to so many people who needed it. But I think what I'm most proud of is the way that we were able to bring people together and have conversations about difficult issues. It wasn't always easy, but I believe that we made progress towards a more united and just society.\n\nYou: What do you think about the current state of politics in the US?\nBarack Obama: Well, politics can be divisive and messy at times. But I have faith in the American people and in our democratic system. We've been through tough times before, and I believe that we'll get through this as well. What's important is that we continue to have honest and respectful conversations, and that we work together to find solutions to the challenges we face.\n\nYou: What do you think is the most pressing issue facing the world today?\nBarack Obama: There are many pressing issues, but if I had to choose one, I would say it's climate change. The science is clear, and the evidence is overwhelming. We have a limited window of time to take meaningful action, and it's up to all of us to do our part. Whether it's reducing our carbon footprint or supporting policies that will address this issue, we all have a role to play.\n"],
    "new_message": "Hi Mr. President, how is it going?"
}

response = model.predict(request)
```

You can also invoke your model via a REST API using cURL.