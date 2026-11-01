from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, welch


@dataclass(frozen=True)
class HREstimate:
    bpm: float
    snr_db: float
    peak_hz: float
    peak_ratio: float
    peak_at_edge: bool
    freqs: np.ndarray
    power: np.ndarray
    used_prior: bool = False


def _refine_peak(band_f: np.ndarray, band_p: np.ndarray, index: int) -> float:
    if not 0 < index < len(band_p) - 1:
        return float(band_f[index])
    y0, y1, y2 = np.log(band_p[index - 1 : index + 2] + 1e-20)
    denominator = y0 - 2.0 * y1 + y2
    if abs(denominator) <= 1e-12:
        return float(band_f[index])
    delta = float(np.clip(0.5 * (y0 - y2) / denominator, -0.5, 0.5))
    return float(band_f[index] + delta * (band_f[1] - band_f[0]))


def estimate_hr(
    signal: np.ndarray,
    fs: float,
    low_hz: float = 0.75,
    high_hz: float = 2.8,
    nperseg: int = 256,
    nfft: int = 4096,
    prior_hz: float | None = None,
    prior_tolerance_hz: float = 0.13,
    prior_min_relative_power: float = 0.5,
    half_bandwidth_hz: float = 0.10,
) -> HREstimate:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if x.size < 16:
        raise ValueError("at least sixteen samples are required")
    if not np.all(np.isfinite(x)):
        raise ValueError("signal contains non-finite values")

    segment = min(int(nperseg), int(x.size))
    noverlap = min(segment // 2, segment - 1)
    f, pxx = welch(
        x,
        fs=float(fs),
        window="hann",
        nperseg=segment,
        noverlap=noverlap,
        nfft=max(int(nfft), segment),
        detrend="linear",
        scaling="density",
    )
    band = (f >= low_hz) & (f <= high_hz)
    if not np.any(band):
        raise ValueError("no PSD bins inside the configured band")
    band_f = f[band]
    band_p = np.maximum(pxx[band], 0.0)
    if band_p.size < 3:
        raise ValueError("not enough PSD bins in the band")
    if float(band_p.max()) <= 0.0:
        raise ValueError("no power inside the configured band")

    peak_index = int(np.argmax(band_p))
    used_prior = False
    if prior_hz is not None and np.isfinite(prior_hz):
        top_hz = float(band_f[peak_index])
        if abs(top_hz - prior_hz) > prior_tolerance_hz:
            local, _ = find_peaks(band_p)
            near = [
                int(i)
                for i in local
                if abs(float(band_f[i]) - prior_hz) <= prior_tolerance_hz
                and band_p[i] >= prior_min_relative_power * band_p[peak_index]
            ]
            if near:
                peak_index = max(near, key=lambda i: band_p[i])
                used_prior = True

    peak_hz = _refine_peak(band_f, band_p, peak_index)

    peak_band = band & (f >= peak_hz - half_bandwidth_hz) & (f <= peak_hz + half_bandwidth_hz)
    if np.count_nonzero(peak_band) >= 2:
        peak_power = float(np.trapezoid(pxx[peak_band], f[peak_band]))
    else:
        peak_power = float(np.max(pxx[peak_band])) if np.any(peak_band) else 0.0
    total_power = float(np.trapezoid(band_p, band_f))
    noise_power = max(total_power - peak_power, 1e-12)
    snr_db = float(10.0 * np.log10(max(peak_power, 1e-12) / noise_power))
    peak_ratio = float(peak_power / max(total_power, 1e-12))
    edge = peak_index <= 0 or peak_index >= len(band_p) - 1

    return HREstimate(
        bpm=peak_hz * 60.0,
        snr_db=snr_db,
        peak_hz=peak_hz,
        peak_ratio=peak_ratio,
        peak_at_edge=bool(edge),
        freqs=f,
        power=pxx,
        used_prior=used_prior,
    )


def band_spectrum(
    signal: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    nfft: int = 4096,
    segment_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD restricted to ``[low_hz, high_hz]``.

    With ``segment_s=None`` the whole window is one Hann segment (best frequency
    resolution); otherwise segments of that length are averaged with 50% overlap.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or x.size < 16 or not np.all(np.isfinite(x)):
        raise ValueError("signal must be a finite 1-D array with at least 16 samples")
    segment = x.size if segment_s is None else int(min(x.size, max(16, round(segment_s * fs))))
    f, pxx = welch(
        x,
        fs=float(fs),
        window="hann",
        nperseg=segment,
        noverlap=segment // 2,
        nfft=max(int(nfft), segment),
        detrend="linear",
        scaling="density",
    )
    band = (f >= low_hz) & (f <= high_hz)
    if np.count_nonzero(band) < 3:
        raise ValueError("not enough PSD bins in the band")
    return f[band], np.maximum(pxx[band], 0.0)


def snr_at(
    freqs: np.ndarray,
    power: np.ndarray,
    f0: float,
    half_bandwidth_hz: float = 0.1,
    harmonic_bandwidth_hz: float = 0.15,
) -> float:
    """De Haan style SNR: power around the fundamental and its first harmonic over
    everything else in the analysed band (dB)."""
    freqs = np.asarray(freqs, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)
    fundamental_mask = np.abs(freqs - f0) <= half_bandwidth_hz
    harmonic_mask = (np.abs(freqs - 2.0 * f0) <= harmonic_bandwidth_hz) & ~fundamental_mask
    fundamental = float(power[fundamental_mask].sum())
    harmonic = float(power[harmonic_mask].sum())
    # The harmonic only counts as signal up to the fundamental's own power: otherwise a
    # candidate at half the true rate would be credited with the real pulse peak.
    signal = fundamental + min(harmonic, fundamental)
    noise = float(power.sum()) - signal
    if signal <= 0:
        return -30.0
    return float(10.0 * np.log10(signal / max(noise, 1e-20)))
