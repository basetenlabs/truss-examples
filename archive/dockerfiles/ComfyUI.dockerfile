FROM baseten/truss-server-base:3.11-gpu-v0.7.17

ARG COMMIT_HASH 6a7bc35db845179a26e62534f3d4b789151e52fe

RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/ComfyUI

RUN cd /app/ComfyUI; git checkout $COMMIT_HASH; pip install -r requirements.txt
