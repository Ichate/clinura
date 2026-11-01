from __future__ import annotations

import math
from datetime import date

import numpy as np
from fastapi.testclient import TestClient

from clinura.athletics import AthleticsStore
from clinura.exercise import L_ANKLE, L_HIP, L_KNEE, LIBRARY, R_ANKLE, R_HIP, R_KNEE, RepCounter
from clinura.server import Runtime, create_app
from clinura.telemetry import TelemetryStore


def _squat_pose(knee_deg: float) -> tuple[np.ndarray, np.ndarray]:
    world = np.zeros((33, 3))
    image = np.zeros((33, 3))
    image[:, 2] = 0.95
    world[[11, 12], 1] = -0.5  # shoulders above hips (y down)
    for hip, knee, ankle, x in ((L_HIP, L_KNEE, L_ANKLE, 0.1), (R_HIP, R_KNEE, R_ANKLE, -0.1)):
        world[knee] = [x, 0.45, 0.0]
        world[ankle] = [x, 0.9, 0.0]
        bend = math.radians(180.0 - knee_deg)
        world[hip] = [x, 0.45 - 0.45 * math.cos(bend), -0.45 * math.sin(bend)]
        world[[11, 12][hip == R_HIP]] = world[hip] + [0.0, -0.5, 0.0]
    return world, image


def test_rep_counter_counts_full_squats_and_flags_partials() -> None:
    counter = RepCounter("squat")
    t = 0.0
    profile = []
    for depth in (90, 90, 130, 90):  # third movement is too shallow
        down = np.linspace(170, depth, 30)
        profile.extend([*down, *down[::-1], *[170] * 10])
    for angle in profile:
        world, image = _squat_pose(float(angle))
        counter.update(world, image, t)
        t += 1 / 30
    summary = counter.summary()
    assert summary["reps"] == 3
    assert summary["partials"] == 1
    assert summary["last_rep"]["rom_min"] < 100


def test_hold_accumulates_only_in_position() -> None:
    counter = RepCounter("wall_sit")
    t = 0.0
    for angle in [170] * 30 + [95] * 90:
        world, image = _squat_pose(float(angle))
        counter.update(world, image, t)
        t += 1 / 30
    assert 2.5 <= counter.summary()["hold_s"] <= 3.1


def test_store_schedule_and_adherence() -> None:
    store = AthleticsStore(None)
    program = store.create_program({"title": "Knee rehab", "prescriber": "Dr A"})
    item = store.add_item(program, {"exercise": "squat", "sets": 3, "reps": 10, "times": ["09:00"]})
    slots = store.schedule(date.today())
    assert len(slots) == 1 and slots[0]["item"]["id"] == item
    store.add_session(
        {
            "program_id": program,
            "item_id": item,
            "exercise": "squat",
            "slot_date": date.today().isoformat(),
            "slot_time": "09:00",
            "reps_done": 30,
        }
    )
    assert store.schedule(date.today())[0]["status"] == "done"
    assert store.adherence(1)[0]["done"] == 1


def test_athletics_api_and_pin() -> None:
    runtime = Runtime(telemetry=TelemetryStore(None))
    runtime.start = lambda: None  # type: ignore[method-assign]
    runtime.stop = lambda: None  # type: ignore[method-assign]
    with TestClient(create_app(runtime, ui_dir=None)) as client:
        assert len(client.get("/api/athletics/library").json()) == len(LIBRARY)
        program = client.post("/api/athletics/programs", json={"title": "Plan"}).json()
        r = client.post(
            f"/api/athletics/programs/{program['id']}/items",
            json={"exercise": "pushup", "sets": 2, "reps": 8, "days": [0, 1, 2, 3, 4, 5, 6]},
        )
        assert r.status_code == 200 and r.json()["items"][0]["reps"] == 8
        assert client.post("/api/athletics/pin", json={"pin": "1234"}).json()["set"]
        assert client.post("/api/athletics/programs", json={"title": "X"}).status_code == 403
        ok = client.post(
            "/api/athletics/programs", json={"title": "X"}, headers={"X-Clinician-Pin": "1234"}
        )
        assert ok.status_code == 200
        saved = client.post(
            "/api/athletics/sessions", json={"exercise": "pushup", "reps_done": 16, "sets_done": 2}
        )
        assert saved.json()["ok"]
        assert client.get("/api/athletics/sessions").json()[0]["reps_done"] == 16


def test_neck_rotation_counts_both_sides_from_neutral() -> None:
    counter = RepCounter("neck_rotation")
    t = 0.0
    sweep = [*np.linspace(0, 45, 20), *np.linspace(45, 0, 20)]
    angles = [0.0] * 15 + sweep + [-a for a in sweep] + sweep
    for yaw in angles:
        counter.update_head((5.0 + float(yaw), 2.0, 1.0), t)  # neutral offset is removed
        t += 1 / 30
    assert counter.summary()["reps"] == 3


def test_neck_nod_counts_and_hold_tracks_level() -> None:
    counter = RepCounter("neck_nod")
    t = 0.0
    sweep = [*np.linspace(0, 30, 20), *np.linspace(30, 0, 20)]
    for pitch in [0.0] * 15 + sweep * 2:
        counter.update_head((0.0, float(pitch), 0.0), t)
        t += 1 / 30
    assert counter.summary()["reps"] == 2
    hold = RepCounter("chin_tuck_hold")
    t = 0.0
    for _ in range(15 + 60):
        hold.update_head((1.0, 1.0, 1.0), t)
        t += 1 / 30
    assert 1.7 <= hold.summary()["hold_s"] <= 2.1
