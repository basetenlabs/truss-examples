# Text Embeddings Inference Truss (A100)
This is an example of a Truss model that uses the Text Embeddings Inference API.

## How to Deploy
In the `config.yaml` file, you can specify the model to use, as well as other arguments per the [Text Embeddings Inference API](https://huggingface.co/docs/text-embeddings-inference) documentation.
Note that not all models are supported by TEI.

To run the model, you can use the following command:
```bash
truss push
```

## How to Generate Embeddings
The truss expects:
- "texts" parameter with either a single string or an array of strings.
- "stream" parameter with a boolean value (default is false).

To generate embeddings, you can use the following command:
```bash
truss predict --d '{"texts": "This is a test"}'
```

# Notes
- The base image is created by installing python on one of the images provided here: https://github.com/huggingface/text-embeddings-inference?tab=readme-ov-file. The current example was built for Ampere 80 architecture, which includes the A100.
- Multi-GPU appears to have no impact on performance
- Be aware of the token limit for each embedding model. It is currently up to the caller to ensure that the texts do not exceed the token limit.

# Improvements
- It may be possible to create a universal base image using the `-all` dockerfile to support a GPU-agnostic implementation
- handle truncation / chunking with averaging (or other technique) when tokens > supported
- investigate impact of dtype on performance
- Add prompt support to embed with prompt
