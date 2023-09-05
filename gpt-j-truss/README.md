[![Deploy to Baseten](https://user-images.githubusercontent.com/2389286/236301770-16f46d4f-4e23-4db5-9462-f578ec31e751.svg)](https://app.baseten.co/explore/gpt_j)

# GPT-J Truss

This is an implementation of EleutherAI
[GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6B). The model consists of 28 layers with a model dimension of 4096,
and a feedforward dimension of 16384. The model dimension is split into 16 heads, each with a dimension of 256.
Rotary Position Embedding (RoPE) is applied to 64 dimensions of each head. The model is trained with a tokenization
vocabulary of 50257, using the same set of BPEs as GPT-2/GPT-3.

## Deploying GPT-J

First, clone this repository:

```sh
git clone https://github.com/basetenlabs/truss-examples/
cd stable-diffusion-truss
```

Before deployment:

1. Make sure you have a [Baseten account](https://app.baseten.co/signup) and [API key](https://app.baseten.co/settings/account/api_keys).
2. Install the latest version of Truss: `pip install --upgrade truss`

With `stable-diffusion-truss` as your working directory, you can deploy the model with:

```sh
truss push
```

Paste your Baseten API key if prompted.

For more information, see [Truss documentation](https://truss.baseten.co).

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

```sh
truss predict -d '{"prompt": "If I was a billionaire, I would"}'
```

You can also invoke this model on Baseten with the following cURL command (just fill in the model version ID and API Key):

```bash
 curl -X POST https://app.baseten.co/models/{MODEL_VERSION_ID}/predict \
  -H 'Authorization: Api-Key {YOUR_API_KEY}' \
  -d '{
    "prompt": "If I was a billionaire, I would",
    "max_length": 50
}'
```
