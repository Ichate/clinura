from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_PROJECTION = np.asarray([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float64)


def pos(rgb: np.ndarray, fs: float = 30.0, window_s: float = 1.6) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape (n_samples, 3) in R, G, B order")
    n = rgb.shape[0]
    if n < 8:
        raise ValueError("at least eight samples are required for POS")
    if not np.all(np.isfinite(rgb)):
        raise ValueError("RGB trace contains non-finite values")
    if np.any(rgb.std(axis=0) < 1e-8):
        raise ValueError("RGB channels have no temporal variance")

    length = int(np.clip(round(window_s * float(fs)), 8, n))
    windows = sliding_window_view(rgb, length, axis=0)
    windows = np.swapaxes(windows, 1, 2)
    means = windows.mean(axis=1, keepdims=True)
    valid = np.all(means > 1e-6, axis=(1, 2))
    safe_means = np.where(means > 1e-6, means, 1.0)
    normalized = windows / safe_means

    projected = normalized @ _PROJECTION.T
    s1 = projected[:, :, 0]
    s2 = projected[:, :, 1]
    std1 = s1.std(axis=1)
    std2 = s2.std(axis=1)
    alpha = np.where(std2 > 1e-12, std1 / np.maximum(std2, 1e-12), 0.0)
    h = s1 + alpha[:, None] * s2
    h -= h.mean(axis=1, keepdims=True)
    h[~valid] = 0.0

    pulse = np.zeros(n, dtype=np.float64)
    count = h.shape[0]
    for offset in range(length):
        pulse[offset : offset + count] += h[:, offset]

    pulse -= pulse.mean()
    scale = float(np.std(pulse))
    if scale < 1e-10:
        raise ValueError("POS produced a flat signal")
    return pulse / scale


def robust_clip(signal: np.ndarray, limit: float = 3.5) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    if x.size < 8:
        return x.copy()
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    sigma = 1.4826 * mad
    if sigma < 1e-12:
        return x.copy()
    return np.clip(x, median - limit * sigma, median + limit * sigma)
