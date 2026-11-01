from __future__ import annotations

import json
import queue
import time

import numpy as np
from fastapi.testclient import TestClient

from clinura.capture import BlackFrameError, CaptureError
from clinura.config import CaptureConfig, FaceConfig, ServerConfig
from clinura.server import CAMERA_ERROR, RUNNING, Runtime, _clean, create_app
from clinura.telemetry import TelemetryStore
from clinura.types import FaceObservation


class FakeCamera:
    fail_opens = 0
    black_indices: set[int] = set()

    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        self._t = 0.0
        self._stopped = False
        self.failed = False

    def start(self) -> None:
        if FakeCamera.fail_opens > 0:
            FakeCamera.fail_opens -= 1
            raise CaptureError("Camera busy")
        if self.config.camera_index in FakeCamera.black_indices:
            raise BlackFrameError(
                f"Camera {self.config.camera_index} is only sending black frames."
            )

    def read(self, timeout: float = 2.0) -> tuple[float, np.ndarray]:
        if self._stopped or self.failed:
            raise queue.Empty
        time.sleep(0.005)
        self._t += 1 / 30
        return self._t, np.full((48, 64, 3), 120, dtype=np.uint8)

    def latest(self) -> tuple[float, np.ndarray] | None:
        return self._t, np.full((48, 64, 3), 120, dtype=np.uint8)

    @property
    def measured_fps(self) -> float:
        return 30.0

    @property
    def size(self) -> tuple[int, int]:
        return (64, 48)

    def stop(self) -> None:
        self._stopped = True


class FakeTracker:
    def __init__(self, config: FaceConfig) -> None:
        self.closed = False

    def detect(self, frame: np.ndarray, timestamp: float) -> FaceObservation:
        return FaceObservation(timestamp=timestamp, detected=False, reason="No face detected")

    def close(self) -> None:
        self.closed = True


def _runtime() -> Runtime:
    return Runtime(
        server=ServerConfig(camera_retry_s=0.05, preview_fps=30.0),
        capture=CaptureConfig(stall_timeout_s=0.3),
        camera_factory=FakeCamera,
        tracker_factory=FakeTracker,
        telemetry=TelemetryStore(None),
    )


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_clean_replaces_non_finite_values() -> None:
    cleaned = _clean({"a": float("nan"), "b": [np.float64(1.5), float("inf")], "c": np.int64(3)})
    assert cleaned == {"a": None, "b": [1.5, None], "c": 3}
    json.dumps(cleaned, allow_nan=False)


def test_runtime_retries_camera_until_available() -> None:
    FakeCamera.fail_opens = 2
    runtime = _runtime()
    runtime.start()
    try:
        assert _wait_for(lambda: runtime.state == RUNNING)
        assert runtime.error is None
        assert _wait_for(lambda: runtime.metrics_payload()["status"] == "no_face")
        payload = runtime.metrics_payload()
        assert payload["engine_state"] == RUNNING
        json.dumps(payload, allow_nan=False)
    finally:
        runtime.stop()
        FakeCamera.fail_opens = 0


def test_runtime_reports_and_recovers_from_camera_loss() -> None:
    runtime = _runtime()
    runtime.start()
    try:
        assert _wait_for(lambda: runtime.state == RUNNING)
        camera = runtime._camera
        assert camera is not None
        camera.failed = True
        assert _wait_for(lambda: runtime.state == CAMERA_ERROR)
        assert runtime.metrics_payload()["status"] == "error" or runtime.state == RUNNING
        assert _wait_for(lambda: runtime.state == RUNNING)
    finally:
        runtime.stop()


def test_mode_commands_are_applied_on_processing_thread() -> None:
    runtime = _runtime()
    runtime.start()
    try:
        assert _wait_for(lambda: runtime.state == RUNNING)
        runtime.set_mode("tongue")
        runtime.set_mode("not-a-mode")
        assert _wait_for(lambda: runtime.snapshot().mode == "tongue")
        assert _wait_for(lambda: runtime.metrics_payload()["experimental"] is not None)
        runtime.reset()
        assert _wait_for(lambda: runtime.snapshot().mode == "tongue")
    finally:
        runtime.stop()


def test_websocket_protocol_tolerates_bad_messages() -> None:
    runtime = _runtime()
    app = create_app(runtime, ui_dir=None)
    with TestClient(app, base_url="http://127.0.0.1:3000") as client:
        assert _wait_for(lambda: runtime.state == RUNNING)
        health = client.get("/api/health").json()
        assert health["state"] == RUNNING
        with client.websocket_connect("ws://127.0.0.1:3000/ws") as ws:
            hello = json.loads(ws.receive_text())
            assert hello["type"] == "hello"
            ws.send_text("not json")
            ws.send_text(json.dumps(["unexpected"]))
            ws.send_text(json.dumps({"type": "set_mode", "mode": "palm"}))
            kinds = set()
            for _ in range(20):
                message = ws.receive()
                if message.get("text"):
                    kinds.add(json.loads(message["text"])["type"])
                elif message.get("bytes"):
                    kinds.add("frame")
                if {"metrics", "frame"} <= kinds:
                    break
            assert {"metrics", "frame"} <= kinds
        assert client.post("/api/reset").json() == {"ok": True}


def test_runtime_skips_cameras_that_send_black_frames() -> None:
    FakeCamera.black_indices = {0}
    runtime = _runtime()
    runtime.start()
    try:
        assert _wait_for(lambda: runtime.state == RUNNING)
        assert runtime.health_payload()["camera"]["index"] == 1
    finally:
        runtime.stop()
        FakeCamera.black_indices = set()


def test_runtime_explains_when_every_camera_is_black() -> None:
    FakeCamera.black_indices = {0, 1, 2, 3}
    runtime = _runtime()
    runtime.start()
    try:
        assert _wait_for(lambda: runtime.state == CAMERA_ERROR)
        assert "Every camera is sending black frames" in (runtime.error or "")
    finally:
        runtime.stop()
        FakeCamera.black_indices = set()
