# default to 1.5 as tag
ARG TAG=1.5
FROM ghcr.io/huggingface/text-embeddings-inference:${TAG}
RUN apt-get update && apt-get install -y python3 python3-pip git git-lfs
RUN git lfs install
ENTRYPOINT ["text-embeddings-router"]
CMD ["--json-output"]
s
