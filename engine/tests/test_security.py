"""Attack scenarios that must fail, and encryption at rest."""

from __future__ import annotations

import sqlite3

import pytest
from cryptography.exceptions import InvalidTag
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from clinura.athletics import AthleticsStore
from clinura.config import ServerConfig
from clinura.server import Runtime, create_app
from clinura.telemetry import TelemetryStore

BASE = "http://127.0.0.1:3000"


def _client(token: str | None = None) -> TestClient:
    runtime = Runtime(telemetry=TelemetryStore(None), server=ServerConfig(access_token=token))
    runtime.start = lambda: None  # type: ignore[method-assign]
    runtime.stop = lambda: None  # type: ignore[method-assign]
    return TestClient(create_app(runtime, ui_dir=None), base_url=BASE)


def test_dns_rebinding_host_is_rejected() -> None:
    with _client() as client:
        r = client.get("/api/health", headers={"Host": "evil.example:3000"})
        assert r.status_code == 400


def test_cross_site_post_is_blocked() -> None:
    with _client() as client:
        r = client.post(
            "/api/athletics/programs",
            json={"title": "x"},
            headers={"Origin": "https://evil.example"},
        )
        assert r.status_code == 403
        r = client.post(
            "/api/events",
            content='{"title":"x"}',
            headers={"Content-Type": "text/plain", "Origin": BASE},
        )
        assert r.status_code == 415
        ok = client.post("/api/events", json={"title": "ok"}, headers={"Origin": BASE})
        assert ok.status_code == 200


def test_cross_site_websocket_hijack_is_blocked() -> None:
    with _client() as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "ws://127.0.0.1:3000/ws", headers={"Origin": "https://evil.example"}
            ) as ws:
                ws.receive_text()


def test_security_headers_and_no_api_docs() -> None:
    with _client() as client:
        r = client.get("/api/health")
        assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["cache-control"] == "no-store"
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


def test_access_token_required_when_configured() -> None:
    with _client(token="s3cret-token-value") as client:
        assert client.get("/api/health").status_code == 401
        r = client.get("/api/health", headers={"Authorization": "Bearer s3cret-token-value"})
        assert r.status_code == 200
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("ws://127.0.0.1:3000/ws") as ws:
                ws.receive_text()


def test_pin_brute_force_lockout_and_wipe_needs_pin() -> None:
    with _client() as client:
        client.post("/api/athletics/pin", json={"pin": "4821"})
        for _ in range(5):
            assert (
                client.post("/api/athletics/pin/verify", json={"pin": "0000"}).json()["ok"] is False
            )
        assert client.post("/api/athletics/pin/verify", json={"pin": "4821"}).status_code == 429
        r = client.post("/api/data/wipe", json={"confirm": "WIPE"})
        assert r.status_code in (403, 429)


def test_wipe_deletes_everything() -> None:
    with _client() as client:
        client.post("/api/events", json={"title": "note"})
        client.post("/api/athletics/programs", json={"title": "Plan"})
        assert client.post("/api/data/wipe", json={"confirm": "nope"}).status_code == 400
        assert client.post("/api/data/wipe", json={"confirm": "WIPE"}).json()["ok"]
        assert client.get("/api/events").json() == []
        assert client.get("/api/athletics/programs").json() == []


def test_databases_are_encrypted_on_disk(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLINURA_DATA_KEY", "test passphrase")
    path = tmp_path / "athletics.db"
    store = AthleticsStore(path)
    store.create_program({"title": "Knee rehab", "patient": "Jane Example"})
    store.close()
    blob = path.read_bytes()
    assert b"Jane Example" not in blob and not blob.startswith(b"SQLite format 3")
    reopened = AthleticsStore(path)
    assert reopened.programs()[0]["patient"] == "Jane Example"
    reopened.close()
    monkeypatch.setenv("CLINURA_DATA_KEY", "wrong passphrase")
    with pytest.raises(InvalidTag):
        AthleticsStore(path)


def test_plaintext_database_is_migrated(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CLINURA_DATA_KEY", "test passphrase")
    path = tmp_path / "telemetry.db"
    legacy = sqlite3.connect(path)
    legacy.execute("PRAGMA journal_mode=WAL")
    legacy.execute("CREATE TABLE samples (ts REAL PRIMARY KEY, bpm REAL)")
    legacy.execute("INSERT INTO samples VALUES (?, 71.0)", (__import__("time").time(),))
    legacy.commit()
    legacy.close()
    store = TelemetryStore(path)
    store.close()
    assert not path.read_bytes().startswith(b"SQLite format 3")
    again = TelemetryStore(path)  # reopening the migrated, encrypted file must work
    assert again._db.execute("SELECT count(*) FROM samples").fetchone()[0] == 1
    again.close()
