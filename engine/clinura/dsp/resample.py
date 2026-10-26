from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def estimate_fs(timestamps: np.ndarray, fallback: float = 30.0) -> float:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.size < 3:
        return float(fallback)
    diffs = np.diff(timestamps)
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-6)]
    if diffs.size == 0:
        return float(fallback)
    median = float(np.median(diffs))
    if not np.isfinite(median) or median <= 0:
        return float(fallback)
    return float(np.clip(1.0 / median, 10.0, 120.0))


def resample_uniform(
    timestamps: np.ndarray,
    values: np.ndarray,
    fs: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if timestamps.ndim != 1 or values.ndim != 2:
        raise ValueError("timestamps must be 1D and values must be 1D or 2D")
    if len(timestamps) != len(values):
        raise ValueError("timestamps and values must have the same length")
    if len(timestamps) < 4:
        raise ValueError("at least four samples are required for resampling")

    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    values = values[order]
    keep = np.concatenate([[True], np.diff(timestamps) > 1e-6])
    timestamps = timestamps[keep]
    values = values[keep]
    if len(timestamps) < 4:
        raise ValueError("not enough unique timestamps for resampling")

    if fs is None:
        fs = estimate_fs(timestamps)

    duration = float(timestamps[-1] - timestamps[0])
    if duration <= 0:
        raise ValueError("timestamps must span a positive duration")

    step = 1.0 / float(fs)
    count = int(np.floor(duration / step)) + 1
    if count < 2:
        raise ValueError("window is too short for the requested sampling rate")
    grid = timestamps[0] + np.arange(count, dtype=np.float64) * step
    grid = grid[grid <= timestamps[-1] + 1e-9]

    interpolator = interp1d(
        timestamps,
        values,
        axis=0,
        kind="linear",
        bounds_error=True,
        assume_sorted=True,
    )
    resampled = interpolator(grid)
    if resampled.ndim == 2 and resampled.shape[1] == 1:
        resampled = resampled[:, 0]
    return grid, resampled, float(fs)
