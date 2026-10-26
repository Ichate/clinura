from __future__ import annotations

from clinura.behavior import EyeActivityTracker, MotionTracker


def test_eye_activity_counts_blinks() -> None:
    tracker = EyeActivityTracker()
    for index in range(20):
        tracker.update(0.0, 0.0, float(index))
    tracker.update(0.9, 0.9, 20.5)
    tracker.update(0.0, 0.0, 21.0)
    result = tracker.update(0.0, 0.0, 22.0)
    assert result["blinks"] == 1
    assert result["blink_rate"] is not None
    assert result["blink_rate"] > 0
    assert result["openness"] == 1.0


def test_motion_tracker_prefers_stable_head() -> None:
    stable = MotionTracker()
    moving = MotionTracker()
    stable_result = None
    moving_result = None
    for index in range(12):
        t = index * 0.1
        stable_result = stable.update(t, 0.01, 0.0, (100 + index, 100, 200, 200))
        moving_result = moving.update(t, 0.22, 12.0, (100 + index * 25, 100 + index * 12, 200, 200))
    assert stable_result is not None and moving_result is not None
    assert float(stable_result["score"]) > float(moving_result["score"])
    assert float(stable_result["score"]) > 80
