import time
from concurrent.futures import ThreadPoolExecutor

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


def test_health_and_metrics(server):
    _, client = server
    assert client.get('/health').status_code == 200
    response = client.get('/metrics')
    assert response.status_code == 200
    assert 'text/plain' in response.headers['content-type']
    assert 'python_info' in response.text
