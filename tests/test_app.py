import base64
import json
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    original_config_file = app.CONFIG_FILE
    original_config = app.config.copy()
    original_proxy_cache = app._proxy_cache.copy()

    monkeypatch.setattr(app, "CONFIG_FILE", str(tmp_path / "config.json"))
    app.config = {**app.DEFAULT_CONFIG, "output_dir": str(tmp_path / "gallery")}
    app._proxy_cache = {"value": None, "checked_at": 0.0}

    yield TestClient(app.app)

    app.CONFIG_FILE = original_config_file
    app.config = original_config
    app._proxy_cache = original_proxy_cache


def test_config_ignores_runtime_and_unknown_fields(client, tmp_path):
    response = client.post(
        "/api/config",
        json={
            "text_model": "test-model",
            "detected_proxy": "http://should-not-persist",
            "unexpected": "ignored",
        },
    )

    assert response.status_code == 200
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["text_model"] == "test-model"
    assert "detected_proxy" not in saved
    assert "unexpected" not in saved


def test_config_rejects_invalid_types(client):
    response = client.post("/api/config", json={"dual_api": "not-a-boolean"})

    assert response.status_code == 422


def test_delete_image_rejects_path_traversal(client, tmp_path):
    canary = tmp_path / "canary.txt"
    canary.write_text("must survive", encoding="utf-8")

    response = client.post("/api/delete-image", json={"filename": "../canary.txt"})

    assert response.status_code == 400
    assert canary.exists()


def test_gallery_returns_image_urls_not_base64_payloads(client, tmp_path):
    output_dir = tmp_path / "gallery"
    output_dir.mkdir()
    image_path = output_dir / "pig_test.png"
    Image.new("RGB", (1, 1), "white").save(image_path)

    response = client.get("/api/gallery")

    assert response.status_code == 200
    item = response.json()["images"][0]
    assert item["filename"] == "pig_test.png"
    assert item["url"] == "/api/gallery/image/pig_test.png"
    assert "data" not in item
    assert client.get(item["url"]).content == image_path.read_bytes()


def test_proxy_detection_uses_cached_result(monkeypatch, client):
    calls = []

    class ClosedSocket:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def settimeout(self, _):
            pass

        def connect_ex(self, address):
            calls.append(address)
            return 1

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(app.socket, "socket", lambda *_: ClosedSocket())

    assert app.auto_detect_proxy() == ""
    first_call_count = len(calls)
    assert app.auto_detect_proxy() == ""

    assert first_call_count == 5
    assert len(calls) == first_call_count


def test_render_error_includes_the_provider_failure(monkeypatch, client):
    class FailingCompletions:
        def create(self, **_):
            raise RuntimeError("provider returned 429")

    class FailingChat:
        completions = FailingCompletions()

    class FailingClient:
        chat = FailingChat()
        base_url = ""

    monkeypatch.setattr(app, "get_image_client", lambda: FailingClient())

    response = client.post("/api/render-image", json={"theme": "测试", "features": "草帽"})

    assert response.status_code == 500
    assert "provider returned 429" in response.json()["message"]


def test_generate_plan_clamps_count_and_rejects_invalid_input(monkeypatch, client):
    captured_messages = []

    class Completion:
        content = '[{"theme":"测试","features":"草帽","slug":"test"}]'

    class Choice:
        message = Completion()

    class Response:
        choices = [Choice()]

    class Completions:
        def create(self, **kwargs):
            captured_messages.extend(kwargs["messages"])
            return Response()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    monkeypatch.setattr(app, "get_text_client", lambda: Client())

    response = client.post("/api/generate-plan", json={"count": 999})

    assert response.status_code == 200
    assert "请设计 20 个" in captured_messages[-1]["content"]
    assert client.post("/api/generate-plan", json={"count": "many"}).status_code == 400


def test_cors_does_not_allow_untrusted_origins(client):
    untrusted = client.get("/api/config", headers={"Origin": "https://attacker.example"})
    trusted = client.get("/api/config", headers={"Origin": "http://127.0.0.1:8000"})

    assert "access-control-allow-origin" not in untrusted.headers
    assert trusted.headers["access-control-allow-origin"] == "http://127.0.0.1:8000"
