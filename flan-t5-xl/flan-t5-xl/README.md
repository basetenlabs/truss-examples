# FLAN-T5 XL Truss

[Flan-T5 XL](https://huggingface.co/google/flan-t5-xl?text=Q%3A+%28+False+or+not+False+or+False+%29+is%3F+A%3A+Let%27s+think+step+by+step) is an open-source large language model developed by Google.

Flan-T5 XL has a number of use cases such as:

- Sentiment analysis
- Paraphrasing/sentence similarity
- Natural language inference
- Sentence completion
- Question answering

Flan-T5 XL is similar to T5 except it is "instruction tuned". In practice, this means that the model is comparable to GPT-3 in multitask benchmarks because it is fine-tuned to follow human inputs / instructions.

## Deploying FLAN-T5 XL

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd flan-t5-xl-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `flan-t5-xl-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

## FLAN-T5 XL API documentation

### Input

The input should be a list of dictionaries and may contain the following key:

- `prompt` - the prompt for text generation
- `bad_words` - an optional list of strings to avoid in the generated output

The [official documentation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate) has information on additional parameters.

```
{
    "prompt": "What is 1+1? Explain your reasoning",
    "bad_words" : ["bad", "word"]
}
```

### Output

The result will be a dictionary containing:

- `status` - either `success` or `failed`
- `data` - the output text
- `message` - will contain details in the case of errors

```
{
    "status" : "success",
    "data" : ["the models response to your prompt"],
    "message" : None

}
```

## Example usage

```sh
truss predict -d '{"prompt": "Answer the question: What is 1+1"}'
```

You can also invoke this model on Baseten with the following cURL command (just fill in the model version ID and API Key):

```bash
 curl -X POST https://app.baseten.co/models/{MODEL_VERSION_ID}/predict \
  -H 'Authorization: Api-Key {YOUR_API_KEY}' \
  -d '{"prompt": "Answer the question: What is 1+1"}'
```
