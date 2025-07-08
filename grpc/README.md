# gRPC on Baseten

This example demonstrates how to deploy a gRPC model on Baseten using Truss.

# Prerequisites

1. **Install Truss:**
   ```bash
   pip install --upgrade truss
   ```

2. **Install Protocol Buffer compiler:**
   ```bash
   # On macOS
   brew install protobuf

   # On Ubuntu/Debian
   sudo apt-get install protobuf-compiler

   # On other systems, see: https://protobuf.dev/getting-started/
   ```

3. **Install gRPC tools:**
   ```bash
   pip install grpcio-tools
   ```

# Steps to Deploy

### Step 1: Generate Protocol Buffer Code

Generate the Python code from your `.proto` file:

```bash
python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path . example.proto
```

### Step 2: Build and Push Docker Image

Build and push your Docker image to a container registry:

```bash
docker build -t your-registry/truss-grpc-demo:latest . --platform linux/amd64
docker push your-registry/truss-grpc-demo:latest
```

### Step 3: Configure your Truss

Update the `config.yaml` file with your model name and Docker image:

```yaml
model_name: "gRPC Model Example"
base_image:
    image: your-registry/truss-grpc-demo:latest
```

### Step 4: Deploy with Truss

Deploy your model using the Truss CLI:

```bash
truss push --promote
```

### Step 5: Invoke the Model

Update the code in `client.py` to connect to your deployed model. Replace `{MODEL_ID}` with your actual model ID,
and the API_KEY with your Baseten API key:

Run your client to test the deployed model:

```bash
python client.py
```
