from __future__ import annotations

from clinura.config import SignalConfig
from clinura.pipeline import SignalPipeline
from clinura.synthetic import generate_synthetic_signal


def _feed(pipeline: SignalPipeline, synthetic: object, quality: float = 0.85) -> None:
    for timestamp, rgb in zip(synthetic.timestamps, synthetic.rgb, strict=True):
        pipeline.set_face_status(True, quality, None, float(timestamp))
        pipeline.add_sample(float(timestamp), tuple(rgb), quality, 180.0, 800)
        pipeline.analyze(float(timestamp))


def test_pipeline_recovers_bpm_and_hrv() -> None:
    synthetic = generate_synthetic_signal(
        duration_s=40,
        fs=30,
        bpm=72,
        rmssd_ms=35,
        noise=0.0003,
        jitter_s=0.001,
        seed=11,
    )
    pipeline = SignalPipeline(SignalConfig(hrv_window_s=60, min_hrv_window_s=30))
    _feed(pipeline, synthetic)
    result = pipeline.analyze(float(synthetic.timestamps[-1]), force=True)
    assert result.bpm is not None
    assert abs(result.bpm - synthetic.true_bpm) < 2.0
    assert result.status in {"measuring", "good"}
    assert result.snr_db is not None and result.snr_db > 0
    assert result.hrv_ready
    assert result.rmssd_ms is not None
    assert abs(result.rmssd_ms - synthetic.true_rmssd_ms) < 10.0
    assert result.waveform_y is not None and len(result.waveform_y) > 100
    assert result.raw_rgb is not None and result.raw_rgb.shape[1] == 3


def test_pipeline_reports_no_face() -> None:
    pipeline = SignalPipeline()
    result = pipeline.analyze(10.0, force=True)
    assert result.status == "no_face"
    assert result.bpm is None


def test_pipeline_acquires_before_window_is_full() -> None:
    synthetic = generate_synthetic_signal(duration_s=4, fs=30, bpm=70, rmssd_ms=20)
    pipeline = SignalPipeline(SignalConfig(min_hr_window_s=10, hrv_window_s=60))
    _feed(pipeline, synthetic)
    result = pipeline.analyze(float(synthetic.timestamps[-1]), force=True)
    assert result.status == "acquiring"
    assert result.bpm is None
