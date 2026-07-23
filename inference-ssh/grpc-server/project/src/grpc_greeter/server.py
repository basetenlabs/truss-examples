"""gRPC greeter server.

In the container this runs as ``python -m grpc_greeter.server`` under ``entr``,
which relaunches it whenever a file in the package changes. Edit ``GREETING``
over SSH and save to see the next call return the new greeting.
"""

from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from grpc_greeter import greeter_pb2, greeter_pb2_grpc

# Edit this over SSH and save. entr restarts the server and the next call
# returns the new greeting.
GREETING = "Hello"

# 50051 is the only server port Baseten's gRPC transport supports.
PORT = 50051


class GreeterServicer(greeter_pb2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        return greeter_pb2.HelloReply(message=f"{GREETING}, {request.name}!")


def main() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    greeter_pb2_grpc.add_GreeterServicer_to_server(GreeterServicer(), server)

    # Baseten decides the server is healthy through the standard gRPC health
    # check service, so it must be registered and marked SERVING. The empty
    # service name is the overall server status that probes query by default;
    # the named service is registered too.
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("greeter.Greeter", health_pb2.HealthCheckResponse.SERVING)

    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"gRPC greeter server started on port {PORT}")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
