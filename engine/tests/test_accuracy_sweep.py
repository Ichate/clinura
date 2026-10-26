from __future__ import annotations

from clinura.config import SignalConfig
from clinura.pipeline import SignalPipeline
from clinura.synthetic import generate_synthetic_signal


def _run(bpm: float, seed: int) -> float:
    synthetic = generate_synthetic_signal(
        duration_s=35,
        fs=30,
        bpm=bpm,
        rmssd_ms=32,
        noise=0.0003,
        jitter_s=0.001,
        seed=seed,
    )
    pipeline = SignalPipeline(SignalConfig(hrv_window_s=60, min_hrv_window_s=30))
    for timestamp, rgb in zip(synthetic.timestamps, synthetic.rgb, strict=True):
        pipeline.set_face_status(True, 0.9, None, float(timestamp))
        pipeline.add_sample(float(timestamp), tuple(rgb), 0.9, 180.0, 800)
    result = pipeline.analyze(float(synthetic.timestamps[-1]), force=True)
    assert result.bpm is not None
    return result.bpm


def test_synthetic_accuracy_sweep() -> None:
    errors = []
    for index, bpm in enumerate((55.0, 65.0, 75.0, 85.0)):
        estimate = _run(bpm, seed=100 + index)
        errors.append(abs(estimate - bpm))
    assert max(errors) < 3.0
