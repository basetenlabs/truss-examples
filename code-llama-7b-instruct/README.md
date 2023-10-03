# Code Llama 7B Instruct

This is a [Truss](https://truss.baseten.co/) for Code Llama 7B Instruct. Code Llama is an LLM based on Llama 2 that is designed to generate code. It comes in three sizes (7b, 13b, 34b) of three variants:

- [Base](https://huggingface.co/codellama/CodeLlama-7b-hf): general purpose code completion.
- [Python](https://huggingface.co/codellama/CodeLlama-7b-Python-hf): Python-specific code completion.
- [Instruct](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf): chat-tuned LLM for coding topics. This Truss packages the 7 billion parameter Instruct model.

## Truss

Truss is an open-source model serving framework developed by Baseten. It allows you to develop and deploy machine learning models onto Baseten (and other platforms like [AWS](https://truss.baseten.co/deploy/aws) or [GCP](https://truss.baseten.co/deploy/gcp)). Using Truss, you can develop a GPU model using [live-reload](https://baseten.co/blog/technical-deep-dive-truss-live-reload), package models and their associated code, create Docker containers and deploy on Baseten.

## Deploying Code Llama

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd code-llama-7b-instruct
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `code-llama-7b-instruct` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

### Hardware notes

This seven billion parameter model is running in `float16` so that it fits on an A10G.

## Code Llama 7B API documentation

This section provides an overview of the Code-Llama 7B API, its parameters, and how to use it. The API consists of a single route named  `predict`, which you can invoke to generate text based on the provided prompt.

### API route: `predict`

The predict route is the primary method for generating text completions based on a given prompt. It takes several parameters:

- __prompt__: The input text that you want the model to generate a response for.
- __max_tokens__ (optional, default=128): The maximum number of tokens to return, counting input tokens. Maximum of 4096.
- __temperature__ (optional, default=0.5): Controls the randomness of the generated text. Higher values produce more diverse results, while lower values produce more deterministic results.
- __top_p__ (optional, default=0.95): The cumulative probability threshold for token sampling. The model will only consider tokens whose cumulative probability is below this threshold.
- __top_k__ (optional, default=50): The number of top tokens to consider when sampling. The model will only consider the top_k highest-probability tokens.

## Example usage

```sh
truss predict -d '{"prompt": "I have a CSV file with the following columns: Python, C++, Bash, Typescript, Java. Create a visualization using seaborn that shows the correlation between Python, C++, Bash, Typescript, and Java.", "max_tokens": 300, "temperature": 0.5}'
```

You can also invoke your model via a REST API:

```
curl -X POST " https://app.baseten.co/model_versions/YOUR_MODEL_VERSION_ID/predict" \
     -H "Content-Type: application/json" \
     -H 'Authorization: Api-Key {YOUR_API_KEY}' \
     -d '{
           "prompt": "I have a CSV file with the following columns: Python, C++, Bash, Typescript, Java. Create a visualization using seaborn that shows the correlation between Python, C++, Bash, Typescript, and Java.",
           "max_tokens": 300
         }'
```

### Example output

You can use the seaborn library to create a visualization that shows the correlation between the columns in your CSV file. Here is an example of how you can do this:
```
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("your_file.csv")

# Create a correlation matrix
corr = df.corr()

# Use seaborn to create a heatmap
sns.heatmap(corr, annot=True, cmap="coolwarm")

# Show the plot
plt.show()
```


