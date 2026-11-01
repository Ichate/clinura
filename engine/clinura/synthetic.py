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


# ---------------------------------------------------------------------------
# Realistic streaming scenarios
#
# The clean generator above validates the DSP maths. The scenarios below model
# what actually makes a webcam reading flaky: per-region signal strength,
# landmark/ROI jitter noise, head-motion bursts (shading + specular), auto-exposure
# steps, frame-time jitter and dropped frames, respiratory sinus arrhythmia and a
# heart rate that changes over time (including post-exercise recovery).
# ---------------------------------------------------------------------------

ROI_NAMES = ("forehead", "right_cheek", "left_cheek")


@dataclass(frozen=True)
class Scenario:
    name: str
    duration_s: float = 90.0
    fs: float = 30.0
    hr_start: float = 70.0
    hr_end: float | None = None
    hr_tau_s: float = 0.0
    hr_wander_bpm: float = 2.0
    rmssd_ms: float = 35.0
    resp_start: float = 14.0
    resp_end: float | None = None
    pulse_level: float = 0.0022
    noise_level: float = 0.0014
    motion_per_min: float = 0.0
    motion_level: float = 0.012
    exposure_steps_per_min: float = 0.0
    drop_rate: float = 0.02
    jitter_s: float = 0.003
    pink_level: float = 0.0


@dataclass
class ScenarioData:
    scenario: Scenario
    timestamps: np.ndarray
    rgb: np.ndarray
    roi_rgb: dict[str, np.ndarray]
    roi_pixels: dict[str, int]
    head_xy: np.ndarray
    face_width: float
    beat_times: np.ndarray
    resp_rate: np.ndarray

    def reference_hr(self, t: float, span_s: float = 6.0) -> float | None:
        beats = self.beat_times[(self.beat_times > t - span_s) & (self.beat_times <= t)]
        if len(beats) < 3:
            return None
        return float(60.0 / np.mean(np.diff(beats)))

    def reference_resp(self, t: float) -> float:
        index = int(np.clip(np.searchsorted(self.timestamps, t), 0, len(self.resp_rate) - 1))
        return float(self.resp_rate[index])


def _approach(start: float, end: float | None, tau: float, t: np.ndarray) -> np.ndarray:
    if end is None:
        return np.full_like(t, start)
    if tau <= 0:
        return start + (end - start) * (t / max(t[-1], 1e-6))
    return end + (start - end) * np.exp(-t / tau)


def _pink(rng: np.random.Generator, n: int, channels: int, fs: float) -> np.ndarray:
    white = rng.normal(0.0, 1.0, (n, channels))
    spectrum = np.fft.rfft(white, axis=0)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    shape = 1.0 / np.sqrt(np.maximum(freqs, 0.3))
    shape[0] = 0.0
    pink = np.fft.irfft(spectrum * shape[:, None], n=n, axis=0)
    return pink / max(float(pink.std()), 1e-12)


def generate_scenario(scenario: Scenario, seed: int = 0) -> ScenarioData:
    rng = np.random.default_rng(seed)
    dense_fs = 240.0
    td = np.arange(0.0, scenario.duration_s + 2.0, 1.0 / dense_fs)

    # Respiration rate trajectory and phase.
    resp_rate = _approach(scenario.resp_start, scenario.resp_end, scenario.hr_tau_s, td)
    resp_phase = 2.0 * np.pi * np.cumsum(resp_rate / 60.0) / dense_fs
    resp = np.sin(resp_phase)

    # Heart-rate trajectory: trend + slow wander, beats with RSA and beat-to-beat noise.
    hr = _approach(scenario.hr_start, scenario.hr_end, scenario.hr_tau_s, td)
    wander = np.cumsum(rng.normal(0.0, 1.0, td.size))
    wander -= np.linspace(wander[0], wander[-1], td.size)
    wander_std = float(np.std(wander))
    if wander_std > 0:
        hr = hr + scenario.hr_wander_bpm * wander / wander_std
    beat_sigma = scenario.rmssd_ms / 1000.0 / np.sqrt(2.0) * 0.6
    beats = [float(rng.uniform(0.0, 0.5))]
    while beats[-1] < td[-1]:
        index = min(int(beats[-1] * dense_fs), td.size - 1)
        ibi = 60.0 / hr[index]
        ibi *= 1.0 + 0.035 * resp[index] * min(1.0, 70.0 / hr[index])
        ibi += rng.normal(0.0, beat_sigma)
        beats.append(beats[-1] + float(np.clip(ibi, 0.27, 2.0)))
    beat_times = np.asarray(beats)

    # PPG waveform with dicrotic wave, respiratory amplitude modulation.
    ppg = np.zeros_like(td)
    for beat, nxt in zip(beat_times[:-1], beat_times[1:], strict=True):
        ibi = nxt - beat
        lo = int(max(0, (beat - 0.3) * dense_fs))
        hi = int(min(td.size, (beat + 1.2 * ibi + 0.3) * dense_fs))
        local = td[lo:hi]
        # Systolic wave ~0.1 s wide at rest (narrower when the heart is fast).
        w = 0.10 * min(1.0, ibi / 0.85) ** 0.5
        ppg[lo:hi] += np.exp(-0.5 * np.square((local - beat) / w))
        ppg[lo:hi] += 0.32 * np.exp(-0.5 * np.square((local - beat - 0.36 * ibi) / (1.7 * w)))
    ppg -= ppg.mean()
    ppg /= max(float(ppg.std()), 1e-9)
    ppg *= 1.0 + 0.12 * resp

    # Frame times with jitter and dropped frames (processing lag).
    nominal = np.arange(0.0, scenario.duration_s, 1.0 / scenario.fs)
    keep = rng.random(nominal.size) >= scenario.drop_rate
    times = nominal[keep] + rng.normal(0.0, scenario.jitter_s, int(keep.sum()))
    times = np.maximum.accumulate(times + 1e-6)

    def at(series: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.interp(t, td, series)

    n = times.size
    pulse = at(ppg, times)
    breathing = at(resp, times)

    # Global intensity: slow light drift, auto-exposure steps, breathing (RIIV).
    intensity = 1.0 + 0.004 * np.sin(2 * np.pi * 0.03 * times + rng.uniform(0, 6))
    intensity += 0.0015 * breathing
    step_count = rng.poisson(scenario.exposure_steps_per_min * scenario.duration_s / 60.0)
    for _ in range(step_count):
        at_t = rng.uniform(5.0, scenario.duration_s)
        size = rng.choice([-1, 1]) * rng.uniform(0.015, 0.05)
        ramp = np.clip((times - at_t) / 0.25, 0.0, 1.0)
        intensity += size * ramp

    # Motion bursts: shading change per region + specular + head displacement.
    head = np.zeros((n, 2))
    head[:, 1] += 0.35 * breathing
    shading = np.zeros(n)
    specular = np.zeros(n)
    burst_count = rng.poisson(scenario.motion_per_min * scenario.duration_s / 60.0)
    for _ in range(burst_count):
        start = rng.uniform(0.0, scenario.duration_s)
        length = rng.uniform(0.8, 3.0)
        env = np.clip(1.0 - np.abs((times - start - length / 2) / (length / 2)), 0.0, 1.0)
        env = np.sin(0.5 * np.pi * env) ** 2
        motion = np.zeros(n)
        for _k in range(3):
            freq = rng.uniform(0.6, 3.2)
            motion += rng.normal() * np.sin(2 * np.pi * freq * times + rng.uniform(0, 6))
        motion *= env / 1.7
        shading += scenario.motion_level * motion
        specular += 0.35 * scenario.motion_level * np.abs(motion)
        head[:, 0] += 18.0 * motion
        head[:, 1] += 9.0 * np.roll(motion, 7)

    base = {
        "forehead": np.asarray([168.0, 122.0, 101.0]),
        "right_cheek": np.asarray([172.0, 116.0, 98.0]),
        "left_cheek": np.asarray([164.0, 113.0, 95.0]),
    }
    pixels = {"forehead": 4200, "right_cheek": 3300, "left_cheek": 3100}
    strength = {"forehead": 1.0, "right_cheek": 0.8, "left_cheek": 0.7}
    pulse_dir = np.asarray([0.33, 0.77, 0.53])
    roi_rgb: dict[str, np.ndarray] = {}
    for name in ROI_NAMES:
        gain = rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0])
        region_intensity = intensity * (1.0 + gain * shading)
        pulsatile = 1.0 + scenario.pulse_level * strength[name] * pulse[:, None] * pulse_dir
        values = base[name][None, :] * region_intensity[:, None] * pulsatile
        values += (specular * 140.0)[:, None]
        noise = scenario.noise_level * np.sqrt(3500.0 / pixels[name])
        values += rng.normal(0.0, 1.0, values.shape) * noise * base[name][None, :]
        if scenario.pink_level > 0:
            # 1/f camera/lighting noise: strongest at low frequencies, like real webcams.
            values += _pink(rng, n, 3, scenario.fs) * scenario.pink_level * base[name][None, :]
        roi_rgb[name] = np.clip(values, 1.0, 255.0)

    weights = np.asarray([pixels[name] for name in ROI_NAMES], dtype=np.float64)
    stacked = np.stack([roi_rgb[name] for name in ROI_NAMES])
    combined = np.tensordot(weights / weights.sum(), stacked, axes=1)
    return ScenarioData(
        scenario=scenario,
        timestamps=times,
        rgb=combined,
        roi_rgb=roi_rgb,
        roi_pixels=pixels,
        head_xy=head + np.asarray([640.0, 360.0]),
        face_width=230.0,
        beat_times=beat_times,
        resp_rate=at(resp_rate, times),
    )


SCENARIOS = (
    Scenario("rest_clean", noise_level=0.0008),
    Scenario("rest_typical"),
    Scenario("rest_noisy", noise_level=0.0024, pulse_level=0.0018),
    Scenario("very_noisy", noise_level=0.0034, pulse_level=0.0017, motion_per_min=3.0),
    Scenario("talking_motion", motion_per_min=10.0),
    Scenario("low_light_exposure", noise_level=0.0026, exposure_steps_per_min=3.0),
    Scenario("hr_drift", hr_start=62.0, hr_end=84.0, hr_tau_s=0.0, hr_wander_bpm=1.0),
    Scenario("athlete_brady", hr_start=47.0, rmssd_ms=80.0, resp_start=10.0),
    Scenario(
        "exercise_recovery",
        hr_start=155.0,
        hr_end=92.0,
        hr_tau_s=45.0,
        rmssd_ms=8.0,
        resp_start=34.0,
        resp_end=17.0,
        motion_per_min=4.0,
        pulse_level=0.0028,
    ),
    Scenario("tachy_steady", hr_start=128.0, rmssd_ms=10.0, resp_start=24.0),
    Scenario("webcam_117", hr_start=117.0, rmssd_ms=12.0, resp_start=18.0, pink_level=0.0025),
    Scenario("webcam_rest_pink", hr_start=68.0, pink_level=0.0025),
)
