FROM python:3.13-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py ./model.py
COPY example_pb2.py example_pb2_grpc.py ./

EXPOSE 8080

CMD ["python", "model.py"]
