# Llama.cpp Baseten model server

Deploying llama.cpp requires a llama.cpp image with python installed and a config.yaml to deploy it. A sample config.yaml is provided in this repository. This sample deploys llama.cpp with Gemma 3 27B Instruct int4 QAT with the 1B model as a draft model.

The following are instructions in case you need to build a image from source. We enable some non-default flags to optimize for performance, so please use the provided Dockerfile.

### Building the docker image from source

#### Prerequisites

- Docker
- NVIDIA Docker runtime
- CUDA-capable GPU

#### Building the docker image

To build the docker image, use the following command:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cp cuda.Dockerfile llama.cpp/.devops/cuda.Dockerfile
cd llama.cpp
docker build -t local/llama.cpp:server-cuda --target server -f .devops/cuda.Dockerfile .
```

You can then push this image to a container registry of your choice and then replace the base_image in the config.yaml