from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low_hz: float = 0.75,
    high_hz: float = 2.8,
    order: int = 3,
) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if x.size < 4:
        return x.copy()
    if not np.all(np.isfinite(x)):
        raise ValueError("signal contains non-finite values")
    nyquist = 0.5 * float(fs)
    if nyquist <= 0:
        raise ValueError("fs must be positive")
    high = min(float(high_hz), 0.95 * nyquist)
    low = max(0.05, float(low_hz))
    if low >= high:
        raise ValueError("low_hz must be below high_hz and the Nyquist frequency")
    sos = butter(order, [low, high], btype="bandpass", fs=float(fs), output="sos")
    padlen = min(x.size - 1, 3 * (2 * len(sos) + 1))
    if padlen < 1:
        return x.copy()
    return sosfiltfilt(sos, x, padlen=padlen)
