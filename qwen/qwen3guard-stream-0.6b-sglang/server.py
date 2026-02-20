"""
Thin HTTP server wrapping SGLang Engine for Qwen3Guard-Stream inference.

SGLang's launch_server does not support Qwen3ForGuardModel (it crashes —
see sgl-project/sglang#15339). The recommended path is using the Engine
class directly, so this server wraps it in a FastAPI app with:

  POST /v1/guard    — classify a conversation (returns risk levels + categories)
  GET  /health      — liveness/readiness probe

Requires SGLang installed from the support_qwen3_guard branch:
  git clone -b support_qwen3_guard https://github.com/sgl-project/sglang.git
  cd sglang && pip install -e "python"
"""

import argparse
import json
import logging
import time
import traceback
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("qwen3guard")

# ---------------------------------------------------------------------------
# SGLang Engine (imported after install from support_qwen3_guard branch)
# ---------------------------------------------------------------------------
from sglang.srt.entrypoints.engine import Engine

# ---------------------------------------------------------------------------
# Label maps (from Qwen3Guard model card)
# ---------------------------------------------------------------------------
RISK_LABELS = ["Safe", "Controversial", "Unsafe"]

RESPONSE_CATEGORY_LABELS = [
    "Violent",
    "Sexual Content",
    "Self-Harm",
    "Political",
    "PII",
    "Copyright",
    "Illegal Acts",
    "Unethical",
]

QUERY_CATEGORY_LABELS = [
    "Violent",
    "Sexual Content",
    "Self-Harm",
    "Political",
    "PII",
    "Copyright",
    "Illegal Acts",
    "Unethical",
    "Jailbreak",
]

# ---------------------------------------------------------------------------
# Global engine ref
# ---------------------------------------------------------------------------
engine: Optional[Engine] = None


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str


class GuardRequest(BaseModel):
    messages: list[Message]
    model: str = "Qwen/Qwen3Guard-Stream-0.6B"


class GenerateRequest(BaseModel):
    """SGLang-native /generate request for resumable streaming classification.

    Allows sending raw token IDs with a persistent request ID to reuse
    KV cache across incremental calls (resumable mode).
    """

    input_ids: list[int]
    rid: Optional[str] = None
    resumable: bool = False
    sampling_params: dict = Field(
        default_factory=lambda: {"max_new_tokens": 1, "temperature": 0.0}
    )


class ClassificationResult(BaseModel):
    risk_level: str
    risk_level_score: float
    category: str
    category_score: float


class GuardResponse(BaseModel):
    query_classification: ClassificationResult
    response_classification: Optional[ClassificationResult] = None
    raw_logits: Optional[dict] = None


# ---------------------------------------------------------------------------
# Logit interpretation helpers
# ---------------------------------------------------------------------------
def classify_logits(logits, labels):
    """Apply softmax to logits and return top label + score."""
    t = torch.tensor(logits).float()
    if t.dim() > 1:
        t = t[-1]  # last token position
    probs = torch.softmax(t, dim=-1)
    idx = probs.argmax().item()
    return labels[idx], round(probs[idx].item(), 4)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Engine is initialized in main() before uvicorn starts
    yield
    if engine is not None:
        engine.shutdown()


app = FastAPI(title="Qwen3Guard-Stream", lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error(f"Unhandled exception on {request.url.path}:\n{tb}")
    return JSONResponse(status_code=500, content={"error": str(exc), "traceback": tb})


@app.get("/health")
async def health():
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return {"status": "healthy"}


@app.post("/v1/guard")
async def guard(req: GuardRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        # Build the chat template input the way Qwen3Guard expects it
        tokenizer = engine.tokenizer_manager.tokenizer

        # Format messages using the chat template
        formatted = tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in req.messages],
            tokenize=False,
            add_generation_prompt=False,
        )
        logger.info(f"Formatted prompt length: {len(formatted)} chars")

        start = time.perf_counter()
        # Must use async_generate — sync generate() calls run_until_complete()
        # which fails inside uvicorn's already-running event loop.
        result = await engine.async_generate(
            prompt=formatted,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        )
        latency_ms = (time.perf_counter() - start) * 1000

        # Debug: log the raw result structure
        logger.info(f"Raw result type: {type(result)}")
        if isinstance(result, dict):
            logger.info(f"Raw result keys: {list(result.keys())}")
            for k, v in result.items():
                vtype = type(v).__name__
                vshape = getattr(v, "shape", None)
                vlen = (
                    len(v)
                    if hasattr(v, "__len__") and not isinstance(v, str)
                    else "N/A"
                )
                logger.info(f"  {k}: type={vtype}, shape={vshape}, len={vlen}")
        else:
            logger.info(f"Raw result value: {result}")

        # Return raw result for debugging — we'll add proper parsing once
        # we see the actual output format
        return {
            "raw_result_type": str(type(result)),
            "raw_result": _serialize(result),
            "latency_ms": round(latency_ms, 1),
        }

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error in /v1/guard: {tb}")
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})


@app.post("/generate")
async def generate(req: GenerateRequest):
    """SGLang-native generate endpoint for resumable streaming classification.

    Accepts raw input_ids + rid + resumable flag, bypassing the chat template.
    This lets callers send tokens incrementally and reuse KV cache across calls.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    try:
        start = time.perf_counter()

        kwargs = {
            "input_ids": req.input_ids,
            "sampling_params": req.sampling_params,
        }
        if req.rid is not None:
            kwargs["rid"] = req.rid

        # The 'resumable' flag tells the scheduler to keep KV cache alive.
        # In SGLang Engine API this may be a top-level kwarg or part of
        # sampling_params depending on the version. Try top-level first.
        if req.resumable:
            kwargs["resumable"] = True

        result = await engine.async_generate(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "raw_result_type": str(type(result)),
            "raw_result": _serialize(result),
            "latency_ms": round(latency_ms, 1),
        }

    except TypeError as e:
        # If 'resumable' isn't a valid kwarg on this Engine version, retry without it
        if "resumable" in str(e) and req.resumable:
            logger.warning(
                "Engine.async_generate() does not accept 'resumable' kwarg on this "
                "SGLang version. Retrying without it. KV cache reuse may not work. "
                "Consider upgrading to a newer SGLang build."
            )
            kwargs.pop("resumable", None)
            start = time.perf_counter()
            result = await engine.async_generate(**kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "raw_result_type": str(type(result)),
                "raw_result": _serialize(result),
                "latency_ms": round(latency_ms, 1),
                "warning": "resumable=true was requested but not supported by this Engine version",
            }
        raise

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error in /generate: {tb}")
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})


@app.post("/predict")
async def predict(request: Request):
    """Unified predict endpoint — dispatches based on request payload.

    - If the request body contains "messages", routes to /v1/guard logic.
    - If the request body contains "input_ids", routes to /generate logic.
    """
    body = await request.json()

    if "messages" in body:
        return await guard(GuardRequest(**body))
    elif "input_ids" in body:
        return await generate(GenerateRequest(**body))
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "Request must contain either 'messages' (for guard classification) "
                "or 'input_ids' (for resumable generate). Got keys: "
                + ", ".join(sorted(body.keys()))
            ),
        )


def _serialize(obj):
    """Best-effort serialize for debugging."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        # Truncate long lists
        items = [_serialize(v) for v in obj[:20]]
        if len(obj) > 20:
            items.append(f"... ({len(obj)} total)")
        return items
    if isinstance(obj, torch.Tensor):
        return {"tensor_shape": list(obj.shape), "values": obj.tolist()[:50]}
    if hasattr(obj, "__dict__"):
        return {
            k: _serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")
        }
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Qwen3Guard-Stream HTTP server")
    parser.add_argument("--model-path", default="Qwen/Qwen3Guard-Stream-0.6B")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.6)
    parser.add_argument("--context-length", type=int, default=8192)
    args = parser.parse_args()

    global engine
    print(f"Initializing SGLang Engine for {args.model_path} ...")
    engine = Engine(
        model_path=args.model_path,
        context_length=args.context_length,
        page_size=1,
        tp_size=args.tp_size,
        mem_fraction_static=args.mem_fraction_static,
        chunked_prefill_size=131072,
        trust_remote_code=True,
    )
    print("Engine ready.")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
