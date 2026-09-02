FROM nvidia/cuda:12.2.2-devel-ubuntu20.04

ENV CUDNN_VERSION=8.9.5.29
ENV CUDA=12.2
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cuda-command-line-tools-12-2 \
        libcublas-12-2 \
        libcublas-dev-12-2 \
        libcufft-12-2 \
        libcurand-12-2 \
        libcusolver-12-2 \
        libcusparse-12-2 \
        libcudnn8=${CUDNN_VERSION}-1+cuda${CUDA} \
        libgomp1 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update -y && \
    apt install -y python3.11 && \
    apt install -y python3.11-distutils && \
    apt install -y python3.11-venv && \
    apt install -y python3.11-dev && \
    rm -rf /var/lib/apt/lists

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py

RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN apt-get update && \
    apt-get install --yes --no-install-recommends python3-packaging && \
    rm -rf /var/lib/apt/lists
RUN pip install --no-cache-dir torch>=2.0.1
RUN pip install --no-cache-dir flash-attn>=2.0.0.post1