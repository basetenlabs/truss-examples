FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED True
ENV NVIDIA_DISABLE_REQUIRE=0

RUN apt-get update && \
    apt-get install --no-install-recommends -y sox libsox-fmt-all curl wget gcc git git-lfs build-essential libaio-dev libsndfile1 ssh ffmpeg && \
    apt-get clean && apt-get -y autoremove

COPY requirements.txt .
RUN python -m pip install --use-deprecated=legacy-resolver -r requirements.txt \
    && python -m pip cache purge
