# gRPC Model Example

Deploy a model using gRPC transport on Baseten. This example demonstrates how to serve a gRPC service using a custom Docker server with Truss.

| Property | Value |
|----------|-------|
| Task | Infrastructure / gRPC transport |
| Engine | Docker Server |
| GPU | A10G |

## Deploy

```sh
truss push
```

## Invoke

This model uses **gRPC transport**, not HTTP/REST. Use the included `client.py` as a reference:

```python
import grpc
import example_pb2
import example_pb2_grpc

channel = grpc.insecure_channel(
    "model-<MODEL_ID>.grpc.api.baseten.co:80",
)

stub = example_pb2_grpc.GreeterStub(channel)

request = example_pb2.HelloRequest(name="World")

metadata = [
    ("baseten-authorization", "Api-Key YOUR_BASETEN_API_KEY"),
    ("baseten-model-id", "<MODEL_ID>"),
]

response = stub.SayHello(request, metadata=metadata)
print(response.message)
```

The proto definition (`example.proto`) defines a simple `Greeter` service with a `SayHello` RPC method.

## Configuration highlights

- Transport: **gRPC** (not HTTP/REST -- curl will not work)
- Base image: `your/repository:tag`
- Docker server with custom `model.py` entrypoint
