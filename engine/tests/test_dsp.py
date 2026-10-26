from __future__ import annotations

import numpy as np

from clinura.dsp import bandpass_filter, detect_peaks, estimate_hr, pos, resample_uniform
from clinura.synthetic import generate_synthetic_signal


def test_bandpass_passes_heart_rate_and_rejects_out_of_band() -> None:
    fs = 30.0
    t = np.arange(0, 20, 1 / fs)
    in_band = np.sin(2 * np.pi * 1.2 * t)
    low = np.sin(2 * np.pi * 0.2 * t)
    high = np.sin(2 * np.pi * 6.0 * t)
    filtered_in = bandpass_filter(in_band, fs)
    filtered_low = bandpass_filter(low, fs)
    filtered_high = bandpass_filter(high, fs)
    assert np.std(filtered_in[60:-60]) > 0.6
    assert np.std(filtered_low[60:-60]) < 0.05
    assert np.std(filtered_high[60:-60]) < 0.05


def test_spectral_estimate_on_clean_sine() -> None:
    fs = 30.0
    t = np.arange(0, 30, 1 / fs)
    signal = np.sin(2 * np.pi * 1.1 * t)
    estimate = estimate_hr(signal, fs)
    assert abs(estimate.bpm - 66.0) < 1.0
    assert estimate.snr_db > 0.0


def test_pos_and_peaks_recover_synthetic_signal() -> None:
    synthetic = generate_synthetic_signal(
        duration_s=40,
        fs=30,
        bpm=72,
        rmssd_ms=35,
        noise=0.0003,
        jitter_s=0.001,
        seed=11,
    )
    grid, rgb, fs = resample_uniform(synthetic.timestamps, synthetic.rgb)
    filtered = bandpass_filter(pos(rgb), fs)
    estimate = estimate_hr(filtered, fs)
    peaks = detect_peaks(filtered, fs)
    assert abs(estimate.bpm - synthetic.true_bpm) < 2.0
    assert peaks.hr_bpm is not None
    assert abs(peaks.hr_bpm - synthetic.true_bpm) < 3.0
    assert peaks.rmssd_ms is not None
    assert abs(peaks.rmssd_ms - synthetic.true_rmssd_ms) < 10.0
    assert peaks.ready
