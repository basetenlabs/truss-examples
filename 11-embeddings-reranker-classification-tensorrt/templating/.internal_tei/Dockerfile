ARG TAG=1.7.1
# this image builds a truss-compatible image with the text-embeddings-inference image as base
# it mainly requires python3
# optional, git and git-lfs are installed to allow for easy cloning of the huggingface model repos.
FROM ghcr.io/huggingface/text-embeddings-inference:${TAG}
RUN apt-get update && apt-get install -y python3 python3-pip git git-lfs
RUN git lfs install
ENTRYPOINT ["text-embeddings-router"]
CMD ["--json-output"]
