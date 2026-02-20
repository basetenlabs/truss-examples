# Qwen3Guard-Stream-0.6B on SGLang (support_qwen3_guard branch)
#
# SGLang's launch_server does not support Qwen3ForGuardModel — we use
# the Engine class wrapped in a thin FastAPI server (server.py).

FROM lmsysorg/sglang:dev-x86

# MUST use the support_qwen3_guard branch — main does NOT have
# Qwen3ForGuardModel support. This branch adds qwen3_guard.py (the model
# class with classification heads) plus patches to the scheduler, model
# runner, and tokenizer manager to handle non-generative guard output.
# Without it, launch_server crashes (sgl-project/sglang#15339).
RUN git clone -b support_qwen3_guard https://github.com/sgl-project/sglang.git /tmp/sglang && \
    cd /tmp/sglang && \
    pip install --upgrade pip && \
    pip install -e "python" && \
    rm -rf /tmp/sglang/.git

# Pre-download the model weights so startup is fast
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3Guard-Stream-0.6B')"

COPY server.py /app/server.py

EXPOSE 8000

ENTRYPOINT ["python3", "/app/server.py"]
CMD ["--model-path", "Qwen/Qwen3Guard-Stream-0.6B", "--host", "0.0.0.0", "--port", "8000"]
