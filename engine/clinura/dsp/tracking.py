"""Temporal tracking of a rate (heart or breathing) from a stream of spectra.

A single 10 s spectrum is ambiguous: motion, lighting and noise regularly produce a
peak that is momentarily stronger than the pulse. Picking the arg-max of every window
independently is what makes a naive rPPG readout jump around.

``SpectralTracker`` runs an online Viterbi (max-product) pass over a fixed rate grid:

* the observation score of a rate is the log of the normalised spectrum there;
* moving between nearby rates costs a Gaussian random-walk penalty (rates drift
  smoothly, a few BPM per second at most);
* jumping anywhere costs a fixed amount, so a genuinely new rate is adopted once it
  has dominated the spectrum for a couple of seconds, but a transient peak is not.

Observation weight and transition cost are both proportional to elapsed time, so the
behaviour does not depend on how often ``update`` is called.

``RateSmoother`` is a 1-D Kalman filter that turns the tracked rate into the displayed
value: measurement noise grows as signal quality drops, and it re-initialises when
the tracker switches to a different peak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackState:
    rate: float
    switched: bool
    margin: float
    stable_s: float


class SpectralTracker:
    def __init__(
        self,
        low: float,
        high: float,
        step: float = 1.0,
        drift_per_sqrt_s: float = 3.0,
        jump_cost: float = 2.5,
        obs_tau_s: float = 1.0,
        floor: float = 0.04,
        max_dt_s: float = 2.0,
    ) -> None:
        self.grid = np.arange(float(low), float(high) + 1e-9, float(step))
        self.drift = float(drift_per_sqrt_s)
        self.jump_cost = float(jump_cost)
        self.obs_tau = float(obs_tau_s)
        self.floor = float(floor)
        self.max_dt = float(max_dt_s)
        self._dist2 = np.square(self.grid[:, None] - self.grid[None, :])
        self._score: np.ndarray | None = None
        self._last_t: float | None = None
        self._last_index: int | None = None
        self._stable_since: float | None = None

    @property
    def active(self) -> bool:
        return self._score is not None

    def reset(self) -> None:
        self._score = None
        self._last_t = None
        self._last_index = None
        self._stable_since = None

    def observation(self, rates: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
        values = np.interp(self.grid, rates, np.maximum(spectrum, 0.0), left=0.0, right=0.0)
        peak = float(values.max())
        if peak <= 0 or not np.isfinite(peak):
            return np.zeros_like(self.grid)
        return np.log(self.floor + values / peak) - np.log(self.floor + 1.0)

    def update(self, rates: np.ndarray, spectrum: np.ndarray, t: float) -> TrackState:
        obs = self.observation(np.asarray(rates, float), np.asarray(spectrum, float))
        if self._score is None or self._last_t is None:
            self._score = obs.copy()
            dt = 0.0
        else:
            dt = float(np.clip(t - self._last_t, 1e-3, self.max_dt))
            variance = max(self.drift**2 * dt, 1e-6)
            transition = np.minimum(self._dist2 / (2.0 * variance), self.jump_cost)
            carried = np.max(self._score[None, :] - transition, axis=1)
            self._score = carried + (dt / self.obs_tau) * obs
            self._score -= self._score.max()
        self._last_t = float(t)

        index = int(np.argmax(self._score))
        switched = False
        if self._last_index is not None:
            gap = abs(self.grid[index] - self.grid[self._last_index])
            switched = gap > max(4.0 * self.drift * np.sqrt(max(dt, 0.05)), 3.0 * self._step)
        if self._stable_since is None or switched:
            self._stable_since = float(t)
        self._last_index = index

        ordered = np.sort(self._score)
        # Margin to the best competitor that is not part of the same peak.
        far = np.abs(self.grid - self.grid[index]) > 6.0 * self._step
        rival = float(self._score[far].max()) if np.any(far) else float(ordered[0])
        return TrackState(
            rate=float(self.grid[index]),
            switched=switched,
            margin=float(self._score[index] - rival),
            stable_s=float(t - self._stable_since),
        )

    @property
    def _step(self) -> float:
        return float(self.grid[1] - self.grid[0]) if self.grid.size > 1 else 1.0


def refine_near(rates: np.ndarray, spectrum: np.ndarray, center: float, span: float) -> float:
    """Return the sub-bin location of the strongest local peak within ``span`` of
    ``center``; falls back to ``center`` when there is no peak there."""
    rates = np.asarray(rates, float)
    spectrum = np.asarray(spectrum, float)
    window = np.flatnonzero(np.abs(rates - center) <= span)
    if window.size < 3:
        return float(center)
    i = int(window[np.argmax(spectrum[window])])
    if i in (window[0], window[-1]) or i <= 0 or i >= rates.size - 1:
        return float(rates[i]) if abs(rates[i] - center) < span else float(center)
    y0, y1, y2 = np.log(spectrum[i - 1 : i + 2] + 1e-20)
    denominator = y0 - 2.0 * y1 + y2
    delta = (
        0.0
        if abs(denominator) < 1e-12
        else float(np.clip(0.5 * (y0 - y2) / denominator, -0.5, 0.5))
    )
    return float(rates[i] + delta * (rates[1] - rates[0]))


class RateSmoother:
    def __init__(self, process_per_s: float = 1.5, base_noise: float = 2.0) -> None:
        self.q = float(process_per_s) ** 2
        self.base_noise = float(base_noise)
        self.value: float | None = None
        self._p = 0.0
        self._t: float | None = None

    def reset(self) -> None:
        self.value = None
        self._p = 0.0
        self._t = None

    def update(self, measurement: float, t: float, quality: float, restart: bool = False) -> float:
        noise = self.base_noise + 6.0 * (1.0 - float(np.clip(quality, 0.0, 1.0)))
        r = noise * noise
        if self.value is None or self._t is None or restart:
            self.value = float(measurement)
            self._p = r
            self._t = float(t)
            return self.value
        dt = float(np.clip(t - self._t, 0.0, 5.0))
        self._t = float(t)
        self._p += self.q * dt
        gain = self._p / (self._p + r)
        self.value += gain * (float(measurement) - self.value)
        self._p *= 1.0 - gain
        return float(self.value)
