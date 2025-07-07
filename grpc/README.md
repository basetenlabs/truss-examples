# gRPC Truss Example

This project demonstrates a gRPC service using Truss for model deployment.

## Prerequisites

- Docker CLI installed
- Go installed
- Access to a container registry (Docker Hub, Google Container Registry, etc.)

## Setup

Get the repository with:
```bash
git clone https://github.com/basetenlabs/truss-grpc-example.git
```

Install Truss with:
```bash
pip install --upgrade truss
```

## Deployment

### 0. Generate Protobuf files
```bash
protoc --go_out=. --go-grpc_out=. example.proto
```
For more information about Protobuf, refer to the [Protobuf documentation](https://protobuf.dev/reference/go/go-generated/).

### 1. Build and Push Docker Image

First, build the Docker image using the provided Dockerfile:

```bash
docker build -t your-registry/truss-grpc-demo:latest . --platform linux/amd64
docker push your-registry/truss-grpc-demo:latest
```

### 2. Update Configuration

Update the `config.yaml` file to use your newly built image:

```yaml
base_image:
  image: your-registry/truss-grpc-demo:latest  # Replace with your image
[...]
```

### 3. Push Model with Truss

Deploy your model using the Truss CLI:

```bash
truss push . --promote
```

For more detailed information about Truss deployment, refer to the [truss push documentation](https://docs.baseten.co/reference/cli/truss/push).

### 4. Call the Model
Copy the model ID from the [Baseten Models page](https://app.baseten.co/models) into `client/main.go` as the `modelID` variable.

Copy the API key from the [Baseten API keys page](https://app.baseten.co/settings/account/api_keys) into `client/main.go` as the `basetenApiKey` variable.

Test the model by running the client:

```bash
go run client/main.go
```
