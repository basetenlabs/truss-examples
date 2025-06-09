FROM baseten/truss-server-base:3.12-gpu-v0.9.0

# Fix the urllib3/six dependency issue first
RUN pip install --upgrade --force-reinstall urllib3 --no-cache-dir

RUN pip uninstall numpy -y || true
RUN pip install numpy==1.26.0 --no-cache-dir