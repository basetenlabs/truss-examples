from contextlib import asynccontextmanager
from threading import Lock
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from model.model import MODEL_NAME, REPO_NAME, Model
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

Content = str | dict[str, Any] | list[Any]


class Question(BaseModel):
    instructions: Content | None = None


class NoulCriteria(BaseModel):
    true: Content | None = None
    false: Content | None = None


class Noul(Question):
    type: Literal["noul"]
    criteria: NoulCriteria | None = None


class Choice(Question):
    type: Literal["choice"]
    criteria: dict[str, Content | None] = Field(min_length=2, max_length=255)


class Score(Question):
    type: Literal["score"]
    criteria: list[Content] = Field(min_length=2, max_length=10)


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal[MODEL_NAME, REPO_NAME] = MODEL_NAME
    state: Content
    questions: dict[
        Annotated[str, Field(pattern=r"\S")],
        Annotated[Noul | Choice | Score, Field(discriminator="type")],
    ] = Field(min_length=1, max_length=1024)


class SystemOneRequest(PredictRequest):
    model: Literal[MODEL_NAME, REPO_NAME]


def create_app(model=None):
    model = model if model is not None else Model()
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        await run_in_threadpool(model.load)
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/models")
    def models():
        return {
            "models": [
                {
                    "name": MODEL_NAME,
                    "description": "Text typed decisions",
                    "modalities": ["text"],
                    "release_date": "2026-10-09",
                }
            ]
        }

    def infer(request):
        payload = request.model_dump(exclude_unset=True)
        payload.setdefault("model", MODEL_NAME)
        # Normalize absent or null SystemOne instructions.
        for question in payload["questions"].values():
            if question.get("instructions") is None:
                question["instructions"] = ""
        # Serialize access to the upstream runtime and its cached inference state.
        with lock:
            try:
                result = model.predict(payload)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        errors = [
            {
                "loc": ["body", "questions", name],
                "msg": answer["error"],
                "type": answer["error"],
            }
            for name, answer in result["answers"].items()
            if "error" in answer
        ]
        if errors:
            status = (
                500 if any(e["type"] == "invalid_model_output" for e in errors) else 422
            )
            raise HTTPException(status_code=status, detail=errors)
        return result

    @app.post("/v1/systemone")
    def system_one(request: SystemOneRequest):
        return infer(request)

    @app.post("/predict")
    def predict(request: PredictRequest):
        return infer(request)

    return app


app = create_app()
