"""Breathing-rate estimation by fusing several respiratory modulations.

Breathing shows up in a face video in several independent ways, each of which fails
in different conditions:

* ``intensity`` – slow change in skin brightness (RIIV: venous return + chest motion
  changing the light on the face);
* ``head`` – the head rises and falls slightly with each breath;
* ``pulse_amplitude`` – pulse strength is modulated by intrathoracic pressure (RIAV);
* ``pulse_rate`` – respiratory sinus arrhythmia speeds the heart on inhalation (RIFV).

Each source gets a spectrum and a quality score (how concentrated its power is around
one peak). Sources are fused weighted by quality, which is the "smart fusion" idea of
Karlen et al. (2013): estimates are only trusted when several modulations agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filtering import bandpass_filter
from .spectral import band_spectrum


@dataclass(frozen=True)
class RespSource:
    name: str
    quality: float
    rate_bpm: float
    freqs: np.ndarray
    power: np.ndarray
    t: np.ndarray
    y: np.ndarray


def _concentration(freqs: np.ndarray, power: np.ndarray, half_width_hz: float = 0.05) -> float:
    total = float(power.sum())
    if total <= 0:
        return 0.0
    peak = float(freqs[int(np.argmax(power))])
    return float(power[np.abs(freqs - peak) <= half_width_hz].sum() / total)


def make_source(
    name: str,
    t: np.ndarray,
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
) -> RespSource | None:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size < max(32, int(8 * fs)) or not np.all(np.isfinite(signal)):
        return None
    if float(np.std(signal)) < 1e-12:
        return None
    high = min(high_hz, 0.45 * fs)
    if high <= low_hz + 0.05:
        return None
    try:
        filtered = bandpass_filter(signal - np.mean(signal), fs, low_hz, high, order=2)
        freqs, power = band_spectrum(filtered, fs, low_hz, high, nfft=8192)
    except ValueError:
        return None
    scale = float(np.std(filtered))
    if scale <= 1e-12:
        return None
    quality = _concentration(freqs, power)
    return RespSource(
        name=name,
        quality=quality,
        rate_bpm=float(freqs[int(np.argmax(power))] * 60.0),
        freqs=freqs,
        power=power / max(float(power.sum()), 1e-20),
        t=np.asarray(t, dtype=np.float64),
        y=filtered / scale,
    )


def beat_series(
    peak_times: np.ndarray,
    pulse: np.ndarray,
    fs: float,
    grid_fs: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Pulse amplitude and inter-beat interval series resampled to ``grid_fs``."""
    peak_times = np.asarray(peak_times, dtype=np.float64)
    if peak_times.size < 8:
        return None
    indices = np.clip(np.round(peak_times * fs).astype(int), 0, pulse.size - 1)
    amplitudes = []
    for i, index in enumerate(indices):
        start = indices[i - 1] if i > 0 else max(0, index - int(0.6 * fs))
        trough = float(np.min(pulse[start : index + 1])) if index > start else float(pulse[index])
        amplitudes.append(float(pulse[index]) - trough)
    amplitudes = np.asarray(amplitudes)
    ibis = np.diff(peak_times)
    ibi_t = peak_times[1:]
    grid = np.arange(peak_times[1], peak_times[-1], 1.0 / grid_fs)
    if grid.size < 16:
        return None
    amp = np.interp(grid, peak_times, amplitudes)
    rate = np.interp(grid, ibi_t, ibis)
    return grid, amp, rate


def fuse(
    sources: list[RespSource],
    low_hz: float,
    high_hz: float,
    min_quality: float = 0.12,
    points: int = 480,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(low_hz, high_hz, points)
    fused = np.zeros_like(grid)
    total = 0.0
    for source in sources:
        if source.quality < min_quality:
            continue
        weight = source.quality**2
        fused += weight * np.interp(grid, source.freqs, source.power, left=0.0, right=0.0)
        total += weight
    if total > 0:
        fused /= total
    return grid, fused
