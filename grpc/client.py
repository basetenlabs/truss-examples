import grpc
import example_pb2
import example_pb2_grpc


def run():
    channel = grpc.secure_channel(
        "model-{MODEL_ID}.api.baseten.co:443", grpc.ssl_channel_credentials()
    )

    stub = example_pb2_grpc.GreeterStub(channel)

    request = example_pb2.HelloRequest(name="World")

    metadata = [
        ("authorization", "Api-Key YOUR_API_KEY"),
    ]

    response = stub.SayHello(request, metadata=metadata)
    print(response.message)


if __name__ == "__main__":
    run()
