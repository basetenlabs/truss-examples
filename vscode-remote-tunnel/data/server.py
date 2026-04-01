"""Minimal FastAPI server for /predict and /health."""

import logging

from fastapi import FastAPI, Request

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    return {"message": "hello from vscode-tunnel truss", "input": body}


# Suppress uvicorn access logs for /health so liveness/readiness probe
# traffic doesn't drown out useful log output.
class _HealthFilter(logging.Filter):
    def filter(self, record):
        return "/health" not in record.getMessage()


logging.getLogger("uvicorn.access").addFilter(_HealthFilter())
