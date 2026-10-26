from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class PeakResult:
    peak_times: np.ndarray
    ibis: np.ndarray
    rmssd_ms: float | None
    hr_bpm: float | None
    ibi_count: int
    diff_count: int
    ready: bool
    polarity: int
    ibi_times: np.ndarray = field(default_factory=lambda: np.empty(0))
    successive_diffs: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def sdnn_ms(self) -> float | None:
        if len(self.ibis) < 3:
            return None
        return float(np.std(self.ibis, ddof=1) * 1000.0)

    @property
    def pnn50(self) -> float | None:
        if len(self.successive_diffs) < 2:
            return None
        return float(np.mean(np.abs(self.successive_diffs) > 0.050) * 100.0)

    @property
    def mean_ibi_ms(self) -> float | None:
        if len(self.ibis) == 0:
            return None
        return float(np.mean(self.ibis) * 1000.0)


def _empty(polarity: int = 1) -> PeakResult:
    return PeakResult(
        peak_times=np.empty(0),
        ibis=np.empty(0),
        rmssd_ms=None,
        hr_bpm=None,
        ibi_count=0,
        diff_count=0,
        ready=False,
        polarity=polarity,
    )


def _candidate_peaks(
    signal: np.ndarray,
    fs: float,
    min_ibi_s: float,
    max_ibi_s: float,
    prominence: float,
) -> tuple[np.ndarray, float]:
    distance = max(1, int(round(min_ibi_s * fs)))
    peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
    if len(peaks) < 2:
        return peaks, float("inf")
    ibis = np.diff(peaks) / fs
    valid = ibis[(ibis >= min_ibi_s) & (ibis <= max_ibi_s)]
    if len(valid) < 2:
        return peaks, float("inf")
    mean = float(np.mean(valid))
    if mean <= 0:
        return peaks, float("inf")
    cv = float(np.std(valid) / mean)
    score = cv + 0.02 / max(1, len(valid))
    return peaks, score


def _refine_peak_times(signal: np.ndarray, peaks: np.ndarray, fs: float) -> np.ndarray:
    times = np.empty(len(peaks), dtype=np.float64)
    for i, peak in enumerate(peaks):
        if peak <= 0 or peak >= len(signal) - 1:
            times[i] = peak / fs
            continue
        y0 = float(signal[peak - 1])
        y1 = float(signal[peak])
        y2 = float(signal[peak + 1])
        denominator = y0 - 2.0 * y1 + y2
        if abs(denominator) < 1e-12:
            delta = 0.0
        else:
            delta = 0.5 * (y0 - y2) / denominator
        delta = float(np.clip(delta, -0.5, 0.5))
        times[i] = (peak + delta) / fs
    return times


def detect_peaks(
    signal: np.ndarray,
    fs: float,
    min_ibi_s: float = 0.33,
    max_ibi_s: float = 1.5,
    prominence_std: float = 0.25,
    max_ibi_deviation: float = 0.20,
    min_valid_ibis: int = 8,
) -> PeakResult:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if x.size < 8 or not np.all(np.isfinite(x)):
        return _empty()
    std = float(np.std(x))
    if std < 1e-8:
        return _empty()
    normalized = (x - float(np.mean(x))) / std
    prominence = max(1e-6, float(prominence_std))

    positive_peaks, positive_score = _candidate_peaks(
        normalized, fs, min_ibi_s, max_ibi_s, prominence
    )
    negative_peaks, negative_score = _candidate_peaks(
        -normalized, fs, min_ibi_s, max_ibi_s, prominence
    )

    if negative_score < positive_score * 0.92:
        peaks = negative_peaks
        working = -normalized
        polarity = -1
    else:
        peaks = positive_peaks
        working = normalized
        polarity = 1

    if len(peaks) < 2:
        return _empty(polarity)

    peak_times = _refine_peak_times(working, peaks, fs)
    ibis = np.diff(peak_times)
    in_range = (ibis >= min_ibi_s) & (ibis <= max_ibi_s)
    if np.any(in_range):
        median_ibi = float(np.median(ibis[in_range]))
        deviation = np.abs(ibis - median_ibi) / max(median_ibi, 1e-9)
        clean_mask = in_range & (deviation <= max_ibi_deviation)
    else:
        clean_mask = np.zeros_like(ibis, dtype=bool)

    clean_ibis = np.full_like(ibis, np.nan)
    clean_ibis[clean_mask] = ibis[clean_mask]
    interval_times = peak_times[1:]
    clean_ibi_times = interval_times[clean_mask]
    diffs = np.diff(clean_ibis)
    diffs = diffs[np.isfinite(diffs)]
    valid_ibis = clean_ibis[np.isfinite(clean_ibis)]

    rmssd_ms = None
    if len(diffs) >= 2:
        rmssd_ms = float(np.sqrt(np.mean(np.square(diffs))) * 1000.0)
    hr_bpm = None
    if len(valid_ibis) >= 2:
        hr_bpm = float(60.0 / np.median(valid_ibis))

    ready = len(valid_ibis) >= min_valid_ibis and len(diffs) >= 2
    return PeakResult(
        peak_times=peak_times,
        ibis=valid_ibis,
        rmssd_ms=rmssd_ms,
        hr_bpm=hr_bpm,
        ibi_count=int(len(valid_ibis)),
        diff_count=int(len(diffs)),
        ready=bool(ready),
        polarity=polarity,
        ibi_times=clean_ibi_times,
        successive_diffs=diffs,
    )
