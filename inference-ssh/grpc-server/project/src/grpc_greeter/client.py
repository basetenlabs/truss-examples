"""Call a deployed gRPC greeter model on Baseten.

Run with ``uv run grpc-greeter-client``. Set ``MY_MODEL_ID`` and ``MY_API_KEY``
in the environment, and optionally pass ``--name`` to change who is greeted.
"""

import argparse
import os

import grpc

from grpc_greeter import greeter_pb2, greeter_pb2_grpc


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the deployed gRPC greeter model.")
    parser.add_argument("--name", default="World", help="Name to greet.")
    args = parser.parse_args()

    model_id = os.environ.get("MY_MODEL_ID")
    api_key = os.environ.get("MY_API_KEY")
    if not model_id or not api_key:
        parser.error("set MY_MODEL_ID and MY_API_KEY in the environment")

    channel = grpc.secure_channel(
        f"model-{model_id}.grpc.api.baseten.co:443",
        grpc.ssl_channel_credentials(),
    )
    stub = greeter_pb2_grpc.GreeterStub(channel)
    metadata = [
        ("baseten-authorization", f"Api-Key {api_key}"),
        ("baseten-model-id", f"model-{model_id}"),
    ]
    response = stub.SayHello(greeter_pb2.HelloRequest(name=args.name), metadata=metadata)
    print(response.message)


if __name__ == "__main__":
    main()
