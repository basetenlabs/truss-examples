[![Deploy to Baseten](https://user-images.githubusercontent.com/2389286/236301770-16f46d4f-4e23-4db5-9462-f578ec31e751.svg)](https://app.baseten.co/explore/gpt_j)

# GPT-J Truss

This is an implementation of EleutherAI
[GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6B). The model consists of 28 layers with a model dimension of 4096,
and a feedforward dimension of 16384. The model dimension is split into 16 heads, each with a dimension of 256.
Rotary Position Embedding (RoPE) is applied to 64 dimensions of each head. The model is trained with a tokenization
vocabulary of 50257, using the same set of BPEs as GPT-2/GPT-3.

## Deploying GPT-J

To deploy the GPT-J Truss, you'll need to follow these steps:

1. __Prerequisites__: Make sure you have a Baseten account and API key. You can sign up for a Baseten account [here](https://app.baseten.co/signup).

2. __Install Truss and the Baseten Python client__: If you haven't already, install the Baseten Python client and Truss in your development environment using:
```
pip install --upgrade baseten truss
```

3. __Load the GPT-J Truss__: Assuming you've cloned this repo, spin up an IPython shell and load the Truss into memory:
```
import truss

gpt_j_truss = truss.load("path/to/gpt_j_truss")
```

4. __Log in to Baseten__: Log in to your Baseten account using your API key (key found [here](https://app.baseten.co/settings/account/api_keys)):
```
import baseten

baseten.login("PASTE_API_KEY_HERE")
```

5. __Deploy the GPT-J Truss__: Deploy the GPT-J Truss to Baseten with the following command:
```
baseten.deploy(gpt_j_truss)
```

Once your Truss is deployed, you can start using the GPT-J model through the Baseten platform! Navigate to the Baseten UI to watch the model build and deploy and invoke it via the REST API.

## GPT-J API documentation

### Input

The input should be a list of dictionaries and must contain the following key:

* `prompt` - the prompt for text generation

Additionally; the following optional parameters are supported as pass thru to the `generate` method. For more details, see the [official documentation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate)

* `max_length` - int - limited to  512
* `min_length` - int - limited to 64
* `do_sample` - bool
* `early_stopping` - bool
* `num_beams` - int
* `temperature`  - float
* `top_k` - int
* `top_p` - float
* `repetition_penalty` - float
* `length_penalty` - float
* `encoder_no_repeat_ngram_size` - int
* `num_return_sequences` - int
* `max_time` - float
* `num_beam_groups` - int
* `diversity_penalty` - float
* `remove_invalid_values` - bool

Here's an example input:

```json
{
    "prompt": "If I was a billionaire, I would",
    "max_length": 50
}
```

### Output

The result will be a dictionary containing:

* `status` - either `success` or `failed`
* `data` - the output text
* `message` - will contain details in the case of errors

```json
{"status": "success", "data": "If I was a billionaire, I would buy a plane.", "message": null}
```

## Example usage

You can invoke this model on Baseten with the following cURL command (just fill in the model version ID and API Key):

```bash
 curl -X POST https://app.baseten.co/models/{MODEL_VERSION_ID}/predict \
  -H 'Authorization: Api-Key {YOUR_API_KEY}' \
  -d '{
    "prompt": "If I was a billionaire, I would",
    "max_length": 50
}'
```
