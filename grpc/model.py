import grpc
from concurrent import futures
import time
import example_pb2
import example_pb2_grpc

from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc
from grpc_health.v1.health import HealthServicer


class GreeterServicer(example_pb2_grpc.GreeterServicer):
    def SayHello(self, request, context):
        response = example_pb2.HelloReply()
        response.message = f"Hello, {request.name}!"
        return response


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    example_pb2_grpc.add_GreeterServicer_to_server(GreeterServicer(), server)

    health_servicer = HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    health_servicer.set(
        "example.GreeterService", health_pb2.HealthCheckResponse.SERVING
    )

    server.add_insecure_port("[::]:50051")

    server.start()
    print("gRPC server started on port 50051")

    # Keep the server running
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        print("Shutting down server...")
        server.stop(0)


if __name__ == "__main__":
    serve()
