import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import anyio
import httpx
import httpx2
import pytest
from fastapi.testclient import TestClient
from model.server import MODEL_NAME, create_app
from typesafe_sdk import Choice, Noul, Score, TypeSafeAPIError, TypeSafeClient


class FakeModel:
    def __init__(self):
        self.payloads = []
        self.error = None
        self.running = False

    def load(self):
        pass

    def predict(self, payload):
        assert not self.running, "Inference must be serialized"
        self.running = True
        time.sleep(0.01)
        self.payloads.append(payload)
        answers = {}
        for name, question in payload["questions"].items():
            kind = question["type"]
            if self.error:
                answers[name] = {"type": kind, "error": self.error}
            elif kind == "noul":
                answers[name] = {"type": kind, "noul": 0.9}
            elif kind == "choice":
                keys = list(question["criteria"])
                answers[name] = {
                    "type": kind,
                    "choice": keys[0],
                    "confidence": 1.0,
                    "probabilities": dict.fromkeys(keys, 1 / len(keys)),
                }
            else:
                legend = {str(i): value for i, value in enumerate(question["criteria"])}
                answers[name] = {
                    "type": kind,
                    "score": 0.5,
                    "confidence": 0.5,
                    "legend": legend,
                    "probabilities": dict.fromkeys(legend, 1 / len(legend)),
                }
        self.running = False
        return {
            "model": MODEL_NAME,
            "answers": answers,
            "usage": {"input_tokens": 12, "output_tokens": 0},
        }


@pytest.fixture
def server():
    model = FakeModel()
    with TestClient(create_app(model)) as client:
        yield model, client


def payload():
    return {
        "model": MODEL_NAME,
        "state": {"text": "Help today"},
        "questions": {"urgent": {"type": "noul"}},
    }


def test_sdk_all_types_and_models(server):
    model, http = server

    def send(request):
        assert request.headers["authorization"] == "Api-Key test-key"
        assert request.url.path.startswith("/sync/v1/")
        response = http.request(
            request.method,
            request.url.path.removeprefix("/sync"),
            content=request.content,
            headers=dict(request.headers),
        )
        return httpx2.Response(response.status_code, json=response.json())

    def baseten_auth(request):
        request.headers["Authorization"] = "Api-Key test-key"

    with TypeSafeClient(
        api_key="test-key",
        base_url="https://example.test/sync",
        model=MODEL_NAME,
        http_client=httpx2.Client(
            transport=httpx2.MockTransport(send),
            event_hooks={"request": [baseten_auth]},
        ),
    ) as client:
        assert client.models.list().models[0].name == MODEL_NAME
        result = client.system_one(
            state={"text": "Help today"},
            questions={
                "urgent": Noul(instructions=None),
                "route": Choice(criteria={"delivery": None, "billing": "Payments"}),
                "priority": Score(criteria=["Low", "High"]),
            },
        )
        assert result.nouls["urgent"].noul == 0.9
        assert result.choices["route"].choice == "delivery"
        assert result.scores["priority"].score == 0.5
        assert all(
            q["instructions"] == "" for q in model.payloads[-1]["questions"].values()
        )
        model.error = "max_length_exceeded"
        with pytest.raises(TypeSafeAPIError) as error:
            client.system_one("long", {"urgent": Noul()})
        assert error.value.status == 422


def test_predict_accepts_original_payload(server):
    _, http = server
    body = payload()
    del body["model"]
    assert http.post("/predict", json=body).status_code == 200
    assert http.post("/v1/systemone", json=body).status_code == 422


@pytest.mark.parametrize(
    "change",
    [
        {"model": "unknown"},
        {"state": 42},
        {"questions": {}},
        {"questions": {"": {"type": "noul"}}},
        {"questions": {"a": {"type": "chat"}}},
        {"questions": {"a": {"type": "score", "criteria": ["one"]}}},
    ],
)
def test_invalid_requests_do_not_reach_model(server, change):
    model, http = server
    response = http.post("/v1/systemone", json=payload() | change)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][0] == "body"
    assert not model.payloads


@pytest.mark.parametrize(
    "error,status",
    [
        ("invalid_question", 422),
        ("max_length_exceeded", 422),
        ("invalid_model_output", 500),
    ],
)
def test_upstream_errors(server, error, status):
    model, http = server
    model.error = error
    response = http.post("/v1/systemone", json=payload())
    assert response.status_code == status
    assert response.json()["detail"][0]["type"] == error


def test_concurrent_requests_are_serialized(server):
    model, http = server
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(
            pool.map(lambda _: http.post("/v1/systemone", json=payload()), range(4))
        )
    assert all(r.status_code == 200 for r in responses)
    assert len(model.payloads) == 4


@pytest.mark.parametrize("endpoint", ["/predict", "/v1/systemone"])
def test_health_responds_when_inference_exhausts_worker_pool(endpoint):
    release = Event()

    class BlockingModel(FakeModel):
        def predict(self, body):
            assert release.wait(timeout=5), "Inference was not released"
            return super().predict(body)

    async def exercise():
        limiter = anyio.to_thread.current_default_thread_limiter()
        original_tokens = limiter.total_tokens
        # One worker runs inference; the other waits for the serialization lock.
        limiter.total_tokens = 2
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app(BlockingModel())),
                base_url="http://testserver",
            ) as client:
                async def infer():
                    response = await client.post(endpoint, json=payload())
                    assert response.status_code == 200

                async with anyio.create_task_group() as tasks:
                    try:
                        tasks.start_soon(infer)
                        tasks.start_soon(infer)
                        with anyio.fail_after(2):
                            while limiter.borrowed_tokens < 2:
                                await anyio.sleep(0)
                        with anyio.fail_after(1):
                            response = await client.get("/health")
                        assert response.status_code == 200
                        assert response.json() == {"status": "ok"}
                    finally:
                        release.set()
        finally:
            limiter.total_tokens = original_tokens

    anyio.run(exercise)


def test_health_and_metrics(server):
    _, client = server
    assert client.get('/health').status_code == 200
    response = client.get('/metrics')
    assert response.status_code == 200
    assert 'text/plain' in response.headers['content-type']
    assert 'python_info' in response.text


def test_repository_alias_and_modalities(server):
    model, http = server
    response = http.post("/v1/systemone", json=payload() | {"model": f"vllm-sr/{MODEL_NAME}"})
    assert response.status_code == 200
    entry = http.get("/v1/models").json()["models"][0]
    assert entry["name"] == MODEL_NAME
    assert entry["modalities"] == ["text", "image", "video"]


def test_media_only_and_mixed_requests_preserve_media(server):
    model, http = server
    body = payload()
    del body["state"]
    body["images"] = ["data:image/png;base64,aW1hZ2U="]
    body["videos"] = ["data:video/mp4;base64,dmlkZW8="]
    assert http.post("/v1/systemone", json=body).status_code == 200
    assert model.payloads[-1]["images"] == body["images"]
    assert model.payloads[-1]["videos"] == body["videos"]
    assert http.post("/predict", json=body | {"state": "What is shown?"}).status_code == 200
    assert model.payloads[-1]["state"] == "What is shown?"


@pytest.mark.parametrize("change", [
    {"images": ["/etc/passwd"]},
    {"images": ["https://example.test/image.png"]},
    {"videos": ["file:///tmp/video.mp4"]},
    {"videos": ["https://example.test/video.mp4"]},
    {"images": ["data:image/svg+xml;base64,abc"]},
    {"images": "data:image/png;base64,abc"},
    {"videos": ["data:image/png;base64,abc"]},
    {"image": ["data:image/png;base64,abc"]},
])
def test_unsupported_media_and_unknown_fields_never_reach_inference(server, change):
    model, http = server
    assert http.post("/predict", json=payload() | change).status_code == 422
    assert not model.payloads


def test_decoder_errors_return_422(server):
    model, http = server
    def invalid_media(payload):
        raise ValueError("images[0]: invalid base64")
    model.predict = invalid_media
    response = http.post("/predict", json=payload() | {"images": ["data:image/png;base64,!"]})
    assert response.status_code == 422
    assert response.json()["detail"] == "images[0]: invalid base64"
