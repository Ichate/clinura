from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks, resample_poly


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


def _local_median(values: np.ndarray, half_width: int = 3) -> np.ndarray:
    out = np.empty_like(values)
    for i in range(values.size):
        lo = max(0, i - half_width)
        hi = min(values.size, i + half_width + 1)
        out[i] = np.median(values[lo:hi])
    return out


def detect_peaks(
    signal: np.ndarray,
    fs: float,
    min_ibi_s: float = 0.33,
    max_ibi_s: float = 1.5,
    prominence_std: float = 0.25,
    max_ibi_deviation: float = 0.20,
    min_valid_ibis: int = 8,
    expected_ibi_s: float | None = None,
    upsample: int = 1,
) -> PeakResult:
    """Detect beats and derive clean inter-beat intervals.

    ``expected_ibi_s`` (from the tracked spectral heart rate) tightens the minimum peak
    spacing so the dicrotic wave or noise wiggles cannot be counted as extra beats.
    ``upsample`` > 1 interpolates the band-limited pulse (polyphase) before peak picking,
    which reduces beat-timing error at webcam frame rates. IBIs are accepted only if
    they lie within ``max_ibi_deviation`` of the *local* median, so slow rate changes
    (breathing, recovery after exercise) are not rejected as artifacts.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if x.size < 8 or not np.all(np.isfinite(x)):
        return _empty()
    std = float(np.std(x))
    if std < 1e-8:
        return _empty()
    if upsample > 1:
        x = resample_poly(x, int(upsample), 1)
        fs = float(fs) * int(upsample)
    normalized = (x - float(np.mean(x))) / max(float(np.std(x)), 1e-12)
    prominence = max(1e-6, float(prominence_std))

    if expected_ibi_s is not None and np.isfinite(expected_ibi_s) and expected_ibi_s > 0:
        min_ibi_s = max(min_ibi_s, 0.62 * float(expected_ibi_s))
        max_ibi_s = min(max_ibi_s, 1.6 * float(expected_ibi_s))

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
    return intervals_from_beats(
        peak_times,
        min_ibi_s=min_ibi_s,
        max_ibi_s=max_ibi_s,
        max_ibi_deviation=max_ibi_deviation,
        min_valid_ibis=min_valid_ibis,
        polarity=polarity,
    )


def intervals_from_beats(
    peak_times: np.ndarray,
    min_ibi_s: float = 0.28,
    max_ibi_s: float = 1.6,
    max_ibi_deviation: float = 0.20,
    min_valid_ibis: int = 8,
    polarity: int = 1,
    beat_ok: np.ndarray | None = None,
) -> PeakResult:
    """Clean inter-beat intervals from beat times.

    An interval is kept only if both of its beats are trusted (``beat_ok``), it is in
    the physiological range, and it is within ``max_ibi_deviation`` of the local median.
    RMSSD/pNN50 use only successive differences of two adjacent clean intervals.
    """
    peak_times = np.asarray(peak_times, dtype=np.float64)
    if peak_times.size < 2:
        return _empty(polarity)
    ibis = np.diff(peak_times)
    in_range = (ibis >= min_ibi_s) & (ibis <= max_ibi_s)
    if beat_ok is not None and len(beat_ok) == peak_times.size:
        ok = np.asarray(beat_ok, dtype=bool)
        in_range &= ok[:-1] & ok[1:]
    if np.count_nonzero(in_range) >= 3:
        reference = np.where(in_range, ibis, np.nan)
        filled = np.where(np.isfinite(reference), reference, float(np.nanmedian(reference)))
        local = _local_median(filled)
        deviation = np.abs(ibis - local) / np.maximum(local, 1e-9)
        clean_mask = in_range & (deviation <= max_ibi_deviation)
    elif np.any(in_range):
        median_ibi = float(np.median(ibis[in_range]))
        deviation = np.abs(ibis - median_ibi) / max(median_ibi, 1e-9)
        clean_mask = in_range & (deviation <= max_ibi_deviation)
    else:
        clean_mask = np.zeros_like(ibis, dtype=bool)

    clean_ibis = np.full_like(ibis, np.nan)
    clean_ibis[clean_mask] = ibis[clean_mask]
    clean_ibi_times = peak_times[1:][clean_mask]
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


def align_beats(
    signal: np.ndarray,
    fs: float,
    beat_times: np.ndarray,
    upsample: int = 8,
    half_width: float = 0.4,
    search: float = 0.15,
    iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-time beats by cross-correlating each one with the average beat shape.

    Peak picking uses three samples around the maximum, so at 30 fps every beat time
    carries ~10-30 ms of noise, which inflates RMSSD. Matching the whole beat against a
    template averages that noise down. Returns refined times and each beat's
    correlation with the template (low correlation = artifact).
    """
    x = np.asarray(signal, dtype=np.float64)
    times = np.asarray(beat_times, dtype=np.float64)
    if times.size < 4 or x.size < 16:
        return times.copy(), np.zeros(times.size)
    ibi = float(np.median(np.diff(times)))
    if not np.isfinite(ibi) or ibi <= 0:
        return times.copy(), np.zeros(times.size)
    fine_fs = float(fs) * int(upsample)
    fine = resample_poly(x - x.mean(), int(upsample), 1)
    fine_t = np.arange(fine.size) / fine_fs
    offsets = np.arange(-half_width * ibi, half_width * ibi, 1.0 / fine_fs)
    lags = np.arange(-search * ibi, search * ibi + 1e-12, 1.0 / fine_fs)

    def segment(center: float) -> np.ndarray | None:
        positions = center + offsets
        if positions[0] < 0 or positions[-1] > fine_t[-1]:
            return None
        values = np.interp(positions, fine_t, fine)
        values = values - values.mean()
        norm = float(np.linalg.norm(values))
        return values / norm if norm > 1e-12 else None

    refined = times.copy()
    corr = np.zeros(times.size)
    for _ in range(max(1, iterations)):
        segments = [segment(t) for t in refined]
        usable = [seg for seg in segments if seg is not None]
        if len(usable) < 3:
            return times.copy(), np.zeros(times.size)
        template = np.median(np.stack(usable), axis=0)
        norm = float(np.linalg.norm(template))
        if norm <= 1e-12:
            return times.copy(), np.zeros(times.size)
        template /= norm
        updated = refined.copy()
        for k, center in enumerate(refined):
            positions = center + lags[:, None] + offsets[None, :]
            inside = (positions[:, 0] >= 0) & (positions[:, -1] <= fine_t[-1])
            if not inside.any():
                corr[k] = 0.0
                continue
            block = np.interp(positions, fine_t, fine)
            block -= block.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(block, axis=1)
            scores = np.where(
                inside & (norms > 1e-12), block @ template / np.maximum(norms, 1e-12), -np.inf
            )
            j = int(np.argmax(scores))
            delta = 0.0
            if 0 < j < lags.size - 1 and np.isfinite(scores[j - 1]) and np.isfinite(scores[j + 1]):
                y0, y1, y2 = scores[j - 1 : j + 2]
                denominator = y0 - 2.0 * y1 + y2
                if abs(denominator) > 1e-12:
                    delta = float(np.clip(0.5 * (y0 - y2) / denominator, -0.5, 0.5))
            updated[k] = center + lags[j] + delta / fine_fs
            corr[k] = float(scores[j])
        refined = updated
    return refined, corr
