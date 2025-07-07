FROM golang:1.24.0 AS builder

WORKDIR /

COPY go.mod go.sum ./

RUN go mod download

COPY example /example
COPY server /server

RUN go build -o grpcserver server/main.go

FROM debian:latest

WORKDIR /

# Install Python, pip, and venv
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

COPY --from=builder /grpcserver ./
