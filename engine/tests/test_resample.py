from __future__ import annotations

import numpy as np

from clinura.dsp import estimate_fs, resample_uniform


def test_estimate_fs_from_jittered_timestamps() -> None:
    rng = np.random.default_rng(3)
    fs = 30.0
    t = np.arange(0, 10, 1 / fs)
    t = t + rng.normal(0, 0.002, size=t.shape)
    t = np.maximum.accumulate(t)
    estimated = estimate_fs(t)
    assert abs(estimated - fs) < 0.5


def test_resample_produces_uniform_grid() -> None:
    rng = np.random.default_rng(4)
    fs = 30.0
    t = np.arange(0, 5, 1 / fs)
    t = t + rng.normal(0, 0.003, size=t.shape)
    t = np.maximum.accumulate(t)
    values = np.column_stack(
        [
            np.sin(2 * np.pi * 1.2 * t),
            np.cos(2 * np.pi * 1.2 * t),
            np.sin(2 * np.pi * 0.2 * t),
        ]
    )
    grid, resampled, actual_fs = resample_uniform(t, values)
    assert resampled.shape == (len(grid), 3)
    assert abs(actual_fs - fs) < 0.5
    assert np.allclose(np.diff(grid), 1.0 / actual_fs, atol=1e-6)
    assert grid[0] >= t[0] and grid[-1] <= t[-1]


def test_resample_rejects_short_windows() -> None:
    try:
        resample_uniform(np.array([0.0, 0.1, 0.2]), np.zeros((3, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("short window should raise")
