from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticSignal:
    timestamps: np.ndarray
    rgb: np.ndarray
    true_bpm: float
    true_rmssd_ms: float
    beat_times: np.ndarray


def _ppg_waveform(t: np.ndarray, beats: np.ndarray) -> np.ndarray:
    waveform = np.zeros_like(t, dtype=np.float64)
    for beat in beats:
        waveform += np.exp(-0.5 * np.square((t - beat) / 0.055))
        waveform += 0.35 * np.exp(-0.5 * np.square((t - beat - 0.22) / 0.090))
    return waveform


def generate_synthetic_signal(
    duration_s: float = 30.0,
    fs: float = 30.0,
    bpm: float = 72.0,
    rmssd_ms: float = 35.0,
    noise: float = 0.0015,
    motion: float = 0.0,
    jitter_s: float = 0.004,
    seed: int = 7,
) -> SyntheticSignal:
    rng = np.random.default_rng(seed)
    rr_mean = 60.0 / float(bpm)
    rr_sigma = float(rmssd_ms) / 1000.0 / np.sqrt(2.0)
    beat_count = int(np.ceil(duration_s / rr_mean)) + 6
    rr = rng.normal(rr_mean, rr_sigma, size=beat_count)
    rr = np.clip(rr, 0.33, 1.5)
    beats = np.cumsum(rr)
    beats = beats[(beats >= 0.0) & (beats <= duration_s + 0.5)]
    if len(beats) < 3:
        raise ValueError("not enough beats generated")

    count = int(round(duration_s * fs))
    t = np.arange(count, dtype=np.float64) / fs
    waveform = _ppg_waveform(t, beats)
    waveform -= waveform.mean()
    scale = waveform.std()
    if scale > 0:
        waveform /= scale

    base = np.asarray([0.63, 0.47, 0.40], dtype=np.float64)
    amplitude = np.asarray([0.0060, 0.0120, 0.0040], dtype=np.float64)
    rgb = base[None, :] + waveform[:, None] * amplitude[None, :]

    drift = 0.010 * np.sin(2.0 * np.pi * 0.05 * t) + 0.006 * np.sin(2.0 * np.pi * 0.13 * t + 0.7)
    rgb += drift[:, None]

    if motion > 0:
        envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.23 * t + 1.1)
        rgb += motion * envelope[:, None] * np.sin(2.0 * np.pi * 1.7 * t)[:, None]

    if noise > 0:
        rgb += rng.normal(0.0, noise, size=rgb.shape)

    timestamps = t.copy()
    if jitter_s > 0:
        timestamps = timestamps + rng.normal(0.0, jitter_s, size=timestamps.shape)
        timestamps = np.maximum.accumulate(timestamps + 1e-9)

    true_rr = np.diff(beats)
    true_rmssd = (
        float(np.sqrt(np.mean(np.square(np.diff(true_rr)))) * 1000.0) if len(true_rr) >= 2 else 0.0
    )
    true_bpm = float(60.0 / np.mean(true_rr))

    return SyntheticSignal(
        timestamps=timestamps,
        rgb=np.clip(rgb * 255.0, 0.0, 255.0),
        true_bpm=true_bpm,
        true_rmssd_ms=true_rmssd,
        beat_times=beats,
    )
