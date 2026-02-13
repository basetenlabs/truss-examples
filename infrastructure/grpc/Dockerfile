FROM debian:latest

RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py ./model.py
COPY example_pb2.py example_pb2_grpc.py ./

EXPOSE 8080

CMD ["python", "model.py"]
