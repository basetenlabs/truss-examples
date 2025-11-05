FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --use-pep517 --no-build-isolation flash-attn==2.6.3
