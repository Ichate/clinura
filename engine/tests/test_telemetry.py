from __future__ import annotations

import time

import httpx
import pytest
from fastapi.testclient import TestClient

from clinura.assistant import Assistant, AssistantConfig, AssistantError, local_handover
from clinura.server import Runtime, create_app
from clinura.telemetry import EventDetector, TelemetryStore, build_context


def _sample(ts: float, bpm: float | None, stability: float = 90.0, eyes: float = 0.0) -> dict:
    return {
        "ts": ts,
        "bpm": bpm,
        "provisional_bpm": None,
        "rmssd_ms": 40.0,
        "sdnn_ms": 50.0,
        "respiration_bpm": 14.0,
        "snr_db": 4.0,
        "confidence": 0.8,
        "status": "good" if bpm is not None else "low_signal",
        "stability": stability,
        "motion_px_s": 2.0,
        "face_detected": 1,
        "eyes_closed": eyes,
    }


def _feed(store: TelemetryStore, start: float) -> None:
    detector = EventDetector(store)
    t = start
    for _ in range(120):
        sample = _sample(t, 72.0)
        store.record(sample)
        detector.update(sample)
        t += 1
    for _ in range(15):
        sample = _sample(t, 118.0, stability=30.0)
        store.record(sample)
        detector.update(sample)
        t += 1
    for _ in range(30):
        sample = _sample(t, 73.0)
        store.record(sample)
        detector.update(sample)
        t += 1
    store.flush()


def test_detector_logs_tachycardia_with_motion_context() -> None:
    store = TelemetryStore(None)
    start = time.time() - 600
    _feed(store, start)
    events = {event.kind: event for event in store.events(start - 1)}
    assert "tachycardia" in events
    tachy = events["tachycardia"]
    assert tachy.end_ts is not None
    assert tachy.value == 118.0
    assert "artifact" in (tachy.detail or "")
    assert "movement" in events


def test_context_summarizes_window() -> None:
    store = TelemetryStore(None)
    _feed(store, time.time() - 600)
    context = build_context(store, hours=1.0)
    assert context["window"]["seconds_recorded"] == 165
    assert context["heart_rate_peak"]["bpm"] == 118.0
    assert context["heart_rate_bpm"]["max"] == 118.0
    assert context["series"]
    assert any(e["kind"] == "tachycardia" for e in context["events"])
    text = local_handover(context, "04", None)
    assert "Bed 04" in text and "Heart rate above 100 BPM" in text


class StubAssistant(Assistant):
    def __init__(self) -> None:
        super().__init__(AssistantConfig(api_key="test"))
        self.prompts: list[str] = []

    async def _chat(self, messages, max_tokens):
        self.prompts.append(messages[-1]["content"])
        return "## Summary\nStable."


def test_ai_endpoints_use_telemetry_context() -> None:
    store = TelemetryStore(None)
    _feed(store, time.time() - 600)
    runtime = Runtime(telemetry=store)
    runtime.start = lambda: None
    runtime.stop = lambda: None
    stub = StubAssistant()
    with TestClient(
        create_app(runtime, ui_dir=None, assistant=stub), base_url="http://127.0.0.1:3000"
    ) as client:
        response = client.post("/api/handover", json={"hours": 1, "bed": "04"})
        assert response.status_code == 200
        assert response.json()["text"].startswith("## Summary")
        assert "tachycardia" in stub.prompts[-1]
        answer = client.post(
            "/api/ask",
            json={"question": "Any spikes?", "history": [{"role": "user", "content": "hi"}]},
        )
        assert answer.status_code == 200
        assert client.post("/api/events", json={"title": "Paracetamol 1 g"}).json()["ok"]
        notes = [e for e in client.get("/api/events").json() if e["source"] == "manual"]
        assert notes[0]["title"] == "Paracetamol 1 g"
        assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_unconfigured_assistant_falls_back_and_reports() -> None:
    runtime = Runtime(telemetry=TelemetryStore(None))
    runtime.start = lambda: None
    runtime.stop = lambda: None
    assistant = Assistant(AssistantConfig(api_key=None))
    with TestClient(
        create_app(runtime, ui_dir=None, assistant=assistant), base_url="http://127.0.0.1:3000"
    ) as client:
        handover = client.post("/api/handover", json={"hours": 1})
        assert handover.status_code == 200
        assert handover.json()["generated_by"] == "local template"
        ask = client.post("/api/ask", json={"question": "Any spikes?"})
        assert ask.status_code == 503
        assert "DEEPSEEK_API_KEY" in ask.json()["detail"]


@pytest.mark.anyio
async def test_deepseek_error_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/completions"
        return httpx.Response(401, json={"error": "bad key"})

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    assistant = Assistant(AssistantConfig(api_key="nope"))
    with pytest.raises(AssistantError, match="rejected the API key"):
        await assistant.ask("q", [], {})


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
