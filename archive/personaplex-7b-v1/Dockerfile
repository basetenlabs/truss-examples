FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libopus-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest (Ubuntu's pip is too old for modern pyproject.toml)
RUN python3 -m pip install --upgrade pip

# Clone personaplex
RUN git clone https://github.com/basetenlabs/personaplex-baseten /opt/personaplex


# Now install moshi-personaplex (version can be read now)
RUN cd /opt/personaplex/moshi && python3 -m pip install --no-cache-dir . \
    && rm -rf /opt/personaplex
