from __future__ import annotations

import numpy as np

from clinura.config import SignalConfig
from clinura.pipeline import SignalPipeline
from clinura.synthetic import generate_synthetic_signal


def test_pipeline_emits_live_analysis_features() -> None:
    synthetic = generate_synthetic_signal(
        duration_s=45,
        fs=30,
        bpm=72,
        rmssd_ms=32,
        noise=0.0004,
        jitter_s=0.001,
        seed=21,
    )
    pipeline = SignalPipeline(SignalConfig())
    for timestamp, rgb in zip(synthetic.timestamps, synthetic.rgb, strict=True):
        pipeline.set_face_status(True, 0.95, None, float(timestamp))
        forehead = tuple(np.asarray(rgb, dtype=float).tolist())
        right = tuple((np.asarray(rgb, dtype=float) * 0.92).tolist())
        left = tuple((np.asarray(rgb, dtype=float) * 1.06).tolist())
        pipeline.add_sample(
            float(timestamp),
            tuple(rgb),
            0.95,
            220.0,
            15000,
            roi_rgb={"forehead": forehead, "right_cheek": right, "left_cheek": left},
            roi_pixels_by_name={"forehead": 5000, "right_cheek": 5000, "left_cheek": 5000},
        )

    result = pipeline.analyze(float(synthetic.timestamps[-1]), force=True)

    assert len(result.roi_quality) == 3
    assert {item["name"] for item in result.roi_quality} == {
        "forehead",
        "right_cheek",
        "left_cheek",
    }
    assert result.psd_f is not None and result.psd_f.size > 0
    assert result.luma_mean is not None and result.luma_std is not None
    assert result.hrv_ready
    assert result.hrv_sdnn_ms is not None
    assert result.hrv_pnn50 is not None
    assert result.hrv_mean_ibi_ms is not None
    assert result.ibi_times is not None and result.ibi_times.size > 2
    assert result.ibis_ms is not None and result.ibis_ms.size > 2
    assert result.respiration_bpm is not None
    assert result.respiration_y is not None and result.respiration_y.size > 20
