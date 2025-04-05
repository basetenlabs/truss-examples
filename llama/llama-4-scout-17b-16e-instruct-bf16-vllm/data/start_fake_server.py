from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ["/health", "/health_generate"]:
            logger.info(
                f"Received health probe request at {self.path}, responding with 200 OK"
            )
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            logger.warning(
                f"Received unexpected GET request at {self.path}, responding with 404"
            )
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            logger.info(
                f"Received model request at {self.path}, responding with 503 Service Unavailable"
            )
            self.send_response(503)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(
                b'{"error": "still downloading model weights, model server is not up yet"}'
            )
        else:
            logger.warning(
                f"Received unexpected POST request at {self.path}, responding with 404"
            )
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "endpoint not found"}')

    def log_message(self, format, *args):
        # Only log errors, not regular requests
        if (
            args and len(args) > 2 and not str(args[1]).startswith("2")
        ):  # Log non-200 responses
            logger.info(f"HTTP {args[1]}: {args[0]}")


def main():
    server = None
    server_thread = None

    # Start HTTP server to respond to Kubernetes probes
    logger.info("Starting fake server on 0.0.0.0:8000")
    logger.info(
        "This server will respond to Kubernetes health probes but reject actual model requests"
    )

    server = HTTPServer(("0.0.0.0", 8000), HealthHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    logger.info("Fake server started successfully")
    logger.info("Server will respond with 200 OK to GET /health and /health_generate")
    logger.info("Server will respond with 503 to POST /v1/chat/completions")

    # Sleep indefinitely to keep the process running
    logger.info(
        "Entering sleep mode. You can now SSH into the pod and run your commands."
    )
    try:
        while True:
            time.sleep(3600)  # Sleep for 1 hour at a time
    except KeyboardInterrupt:
        logger.info("Shutting down fake server")
        server.shutdown()


if __name__ == "__main__":
    main()
