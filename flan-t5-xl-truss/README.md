[![Deploy to Baseten](https://user-images.githubusercontent.com/2389286/236301770-16f46d4f-4e23-4db5-9462-f578ec31e751.svg)](https://app.baseten.co/explore/flan_t5)

# FLAN-T5 XL Truss

[Flan-T5 XL](https://huggingface.co/google/flan-t5-xl?text=Q%3A+%28+False+or+not+False+or+False+%29+is%3F+A%3A+Let%27s+think+step+by+step) is an open-source large language model developed by Google.

Flan-T5 XL has a number of use cases such as:

* Sentiment analysis
* Paraphrasing/sentence similarity
* Natural language inference
* Sentence completion
* Question answering

Flan-T5 XL is similar to T5 except it is "instruction tuned". In practice, this means that the model is comparable to GPT-3 in multitask benchmarks because it is fine-tuned to follow human inputs / instructions.

## Deploying FLAN-T5 XL

To deploy the FLAN-T5 XL Truss, you'll need to follow these steps:

1. __Prerequisites__: Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup).

2. __Install Truss and the Baseten Python client__: If you haven't already, install the Baseten Python client and Truss in your development environment using:
```
pip install --upgrade baseten truss
```

3. __Load the FLAN-T5 XL Truss__: Assuming you've cloned this repo, spin up an IPython shell and load the Truss into memory:
```
import truss

flan_t5_truss = truss.load("path/to/flan_t5_truss")
```

4. __Log in to Baseten__: Log in to your Baseten account using your API key (key found [here](https://app.baseten.co/settings/account/api_keys)):
```
import baseten

baseten.login("PASTE_API_KEY_HERE")
```

5. __Deploy the FLAN-T5 XL Truss__: Deploy the FLAN-T5 XL Truss to Baseten with the following command:
```
baseten.deploy(flan_t5_truss)
```

Once your Truss is deployed, you can start using the FLAN-T5 XL model through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

## FLAN-T5 XL API documentation

### Input

The input should be a list of dictionaries and may contain the following key:

* `prompt` - the prompt for text generation
* `bad_words` - an optional list of strings to avoid in the generated output

The [official documentation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate) has information on additional parameters.

```
{
    "prompt": "What is 1+1? Explain your reasoning",
    "bad_words" : ["bad", "word"]
}
```

### Output

The result will be a dictionary containing:

* `status` - either `success` or `failed`
* `data` - the output text
* `message` - will contain details in the case of errors

```
{
    "status" : "success",
    "data" : ["the models response to your prompt"],
    "message" : None

}
```

## Example usage

You can invoke this model on Baseten with the following cURL command (just fill in the model version ID and API Key):

```bash
 curl -X POST https://app.baseten.co/models/{MODEL_VERSION_ID}/predict \
  -H 'Authorization: Api-Key {YOUR_API_KEY}' \
  -d '{"prompt": "Answer the question: What is 1+1"}'
```
