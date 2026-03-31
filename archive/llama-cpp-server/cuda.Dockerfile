ARG BASE_CUDA_DEV_CONTAINER=nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ARG BASE_CUDA_RUNTIME_CONTAINER=nvcr.io/nvidia/cuda:12.8.1-runtime-ubuntu22.04
FROM ${BASE_CUDA_DEV_CONTAINER} AS build

# CUDA architecture to build for (defaults to all supported archs)
ARG CUDA_DOCKER_ARCH=default

RUN apt-get update && \
    apt-get install -y build-essential cmake python3 python3-pip git libcurl4-openssl-dev libgomp1

WORKDIR /app

COPY . .

RUN if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
    export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B build -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_BUILD_SERVER=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="86;89;90" -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=OFF ${CMAKE_ARGS} -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
    cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so" -exec cp {} /app/lib \;

RUN mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

## Base image
FROM ${BASE_CUDA_RUNTIME_CONTAINER} AS base

RUN apt-get update \
    && apt-get install -y libgomp1 curl\
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

COPY --from=build /app/lib/ /app
COPY --from=build /app/requirements /app/requirements
COPY --from=build /app/requirements.txt /app/requirements.txt

### Server, Server only
FROM base AS server
COPY --from=build /app/full /app
COPY --from=build /app/requirements /app/requirements
COPY --from=build /app/requirements.txt /app/requirements.txt
ENV LLAMA_ARG_HOST=0.0.0.0
WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]
COPY requirements.txt /app/
RUN apt-get update \
    && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python-is-python3 \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

ENTRYPOINT [ "/app/llama-server" ]
