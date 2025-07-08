import grpc
import example_pb2
import example_pb2_grpc


def run():
    channel = grpc.insecure_channel(
        "model-{MODEL_ID}.grpc.api.baseten.co:80",
    )

    stub = example_pb2_grpc.GreeterStub(channel)

    request = example_pb2.HelloRequest(name="World")

    metadata = [
        ("baseten-authorization", "Api-Key {API_KEY}"),
        ("baseten-model-id", "{MODEL_ID}"),
    ]

    response = stub.SayHello(request, metadata=metadata)
    print(response.message)


if __name__ == "__main__":
    run()
