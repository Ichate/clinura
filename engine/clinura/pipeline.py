"""Real-time rPPG signal pipeline.

Per analysis (15 Hz) on the last ~10 s of samples:

1. Resample every trace (combined face, each skin region, head position) onto a
   uniform grid.
2. Turn each region into pulse signals with both POS and CHROM, clip impulsive
   artifacts and band-pass them.
3. Fuse the per-channel spectra, weighting each by how clearly it shows *one*
   dominant peak (its own SNR, smoothed over a few seconds so weights do not flap).
4. Attenuate frequencies where the head is visibly moving.
5. Track the heart rate over time with an online Viterbi tracker that uses the first
   harmonic as extra evidence, so a transient noise peak or the harmonic of a slow
   heart cannot hijack the reading.
6. Refine the tracked rate on the current spectrum and smooth it with a Kalman filter
   whose trust follows the signal quality.
7. Decide what to show with a small state machine (searching → locked ↔ holding) so
   the number does not blink on and off when the SNR hovers near the threshold.

HRV (1 Hz, 60 s) and breathing rate (1 Hz, ~30 s) reuse the fused pulse and the
tracked heart rate.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from .config import SignalConfig
from .dsp import (
    RateSmoother,
    SpectralTracker,
    align_beats,
    band_spectrum,
    bandpass_filter,
    chrom,
    detect_peaks,
    intervals_from_beats,
    pos,
    refine_near,
    resample_uniform,
    robust_clip,
    snr_at,
)
from .dsp.respiration import RespSource, beat_series, fuse, make_source
from .types import AnalysisResult, SignalSample

ROI_ORDER = ("forehead", "right_cheek", "left_cheek")
COMBINED = "combined"


@dataclass
class Channel:
    source: str
    method: str
    pulse: np.ndarray
    power: np.ndarray
    own_snr_db: float
    own_peak_hz: float

    @property
    def key(self) -> str:
        return f"{self.source}:{self.method}"


@dataclass
class PulseWindow:
    grid: np.ndarray
    rgb: np.ndarray
    fs: float
    pulse: np.ndarray
    freqs: np.ndarray
    spectrum: np.ndarray
    tracking_spectrum: np.ndarray
    flat_spectrum: np.ndarray
    channels: list[Channel] = field(default_factory=list)
    motion_level: float | None = None


def _empty_hrv(progress: float = 0.0) -> dict[str, object]:
    return {
        "ready": False,
        "progress": progress,
        "ibi_count": 0,
        "rmssd_ms": None,
        "sdnn_ms": None,
        "pnn50": None,
        "mean_ibi_ms": None,
        "quality": None,
        "ibi_times_abs": np.empty(0),
        "ibis_ms": np.empty(0),
        "beats_abs": np.empty(0),
    }


class SignalPipeline:
    def __init__(self, config: SignalConfig | None = None) -> None:
        self.config = config or SignalConfig()
        c = self.config
        self._samples: deque[SignalSample] = deque()
        self._tracker = SpectralTracker(
            low=round(c.band_low_hz * 60.0),
            high=round(c.band_high_hz * 60.0),
            step=1.0,
            drift_per_sqrt_s=c.tracker_drift_bpm,
            jump_cost=c.tracker_jump_cost,
            obs_tau_s=c.tracker_obs_tau_s,
        )
        self._smoother = RateSmoother(c.smoother_process_bpm, c.smoother_base_noise_bpm)
        self._resp_tracker = SpectralTracker(
            low=5.0,
            high=46.0,
            step=0.25,
            drift_per_sqrt_s=1.2,
            jump_cost=3.0,
            obs_tau_s=2.0,
        )
        self._resp_smoother = RateSmoother(process_per_s=0.8, base_noise=1.0)
        self._hrv_history: deque[float] = deque(maxlen=max(1, c.hrv_smoothing))
        self._channel_weights: dict[str, float] = {}
        self._last_analysis = 0.0
        self._last_hrv_at = 0.0
        self._last_respiration_at = 0.0
        self._hrv_cache: dict[str, object] | None = None
        self._respiration_cache: dict[str, object] | None = None
        self._snr_ema: float | None = None
        self._agreement_ema: float | None = None
        self._confidence_ema: float | None = None
        self._last_update_t: float | None = None
        self._lock = "searching"
        self._hold_since: float | None = None
        self._tracked_bpm: float | None = None
        self._last_result = AnalysisResult(
            timestamp=0.0,
            status="acquiring",
            message="Waiting for the camera",
        )
        self._face_present = False
        self._face_pose_ok = True
        self._last_face_time = 0.0
        self._face_quality = 0.0
        self._face_reason: str | None = None

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        self._samples.clear()
        self._clear_caches()
        self._last_analysis = 0.0
        self._last_result = AnalysisResult(
            timestamp=time.perf_counter(),
            status="acquiring",
            message="Waiting for the camera",
        )
        self._face_present = False
        self._face_pose_ok = True
        self._last_face_time = 0.0
        self._face_quality = 0.0
        self._face_reason = None

    def _clear_caches(self) -> None:
        self._tracker.reset()
        self._smoother.reset()
        self._resp_tracker.reset()
        self._resp_smoother.reset()
        self._hrv_history.clear()
        self._channel_weights.clear()
        self._hrv_cache = None
        self._respiration_cache = None
        self._last_hrv_at = 0.0
        self._last_respiration_at = 0.0
        self._snr_ema = None
        self._agreement_ema = None
        self._confidence_ema = None
        self._last_update_t = None
        self._lock = "searching"
        self._hold_since = None
        self._tracked_bpm = None

    def set_face_status(
        self,
        detected: bool,
        quality: float,
        reason: str | None,
        timestamp: float,
        pose_ok: bool = True,
    ) -> None:
        self._face_present = bool(detected)
        self._face_pose_ok = bool(detected and pose_ok)
        self._face_quality = float(quality) if np.isfinite(quality) else 0.0
        self._face_reason = reason
        if detected:
            self._last_face_time = float(timestamp)

    def add_sample(
        self,
        timestamp: float,
        rgb: tuple[float, float, float],
        quality: float,
        face_width_px: float,
        roi_pixels: int,
        roi_rgb: dict[str, tuple[float, float, float]] | None = None,
        roi_pixels_by_name: dict[str, int] | None = None,
        head_xy: tuple[float, float] | None = None,
    ) -> None:
        if not np.isfinite(timestamp) or not all(np.isfinite(rgb)):
            return
        if min(rgb) <= 0.0:
            return
        if self._samples and timestamp <= self._samples[-1].timestamp:
            timestamp = self._samples[-1].timestamp + 1e-6
        clean_roi_rgb = {
            name: (float(v[0]), float(v[1]), float(v[2]))
            for name, v in (roi_rgb or {}).items()
            if len(v) == 3 and all(np.isfinite(v)) and min(v) > 0.0
        }
        head = None
        if head_xy is not None and len(head_xy) == 2 and all(np.isfinite(head_xy)):
            head = (float(head_xy[0]), float(head_xy[1]))
        luma = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        self._samples.append(
            SignalSample(
                timestamp=float(timestamp),
                rgb=(float(rgb[0]), float(rgb[1]), float(rgb[2])),
                quality=float(quality),
                face_width_px=float(face_width_px),
                roi_pixels=int(roi_pixels),
                roi_rgb=clean_roi_rgb,
                roi_pixels_by_name={
                    name: int(count)
                    for name, count in (roi_pixels_by_name or {}).items()
                    if name in clean_roi_rgb
                },
                luma=float(luma),
                head_xy=head,
            )
        )
        cutoff = timestamp - (
            max(self.config.hrv_window_s, self.config.respiration_window_s) + 10.0
        )
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    @property
    def latest(self) -> AnalysisResult:
        return self._last_result

    def _window(self, now: float, duration_s: float) -> list[SignalSample]:
        start = now - duration_s
        samples = [
            s
            for s in self._samples
            if s.timestamp >= start and s.quality >= self.config.min_sample_quality
        ]
        if len(samples) < 2:
            return samples
        contiguous_start = 0
        for index in range(1, len(samples)):
            if samples[index].timestamp - samples[index - 1].timestamp > self.config.max_gap_s:
                contiguous_start = index
        return samples[contiguous_start:]

    # ------------------------------------------------------------ pulse fusion

    def _channel(self, source: str, method: str, rgb: np.ndarray, fs: float) -> Channel | None:
        c = self.config
        try:
            raw = (
                pos(rgb, fs=fs, window_s=c.pos_window_s)
                if method == "pos"
                else chrom(rgb, fs=fs, window_s=c.pos_window_s)
            )
            raw = robust_clip(raw, c.artifact_clip_sigma)
            pulse = bandpass_filter(raw, fs, c.filter_low_hz, c.filter_high_hz, c.filter_order)
            freqs, power = band_spectrum(
                pulse, fs, c.band_low_hz, min(c.spectrum_high_hz, 0.48 * fs), nfft=c.welch_nfft
            )
        except (ValueError, FloatingPointError):
            return None
        total = float(power.sum())
        if total <= 0 or not np.isfinite(total):
            return None
        power = power / total
        search = freqs <= c.band_high_hz
        peak_hz = float(freqs[search][int(np.argmax(power[search]))])
        return Channel(
            source=source,
            method=method,
            pulse=pulse,
            power=power,
            own_snr_db=snr_at(freqs, power, peak_hz),
            own_peak_hz=peak_hz,
        )

    def _analyze_window(
        self, samples: list[SignalSample], now: float | None = None, update_weights: bool = True
    ) -> PulseWindow:
        c = self.config
        timestamps = np.asarray([s.timestamp for s in samples], dtype=np.float64)
        columns = [np.asarray([s.rgb for s in samples], dtype=np.float64)]

        roi_names: list[str] = []
        if c.fuse_rois:
            shared = set(samples[0].roi_rgb)
            for sample in samples[1:]:
                shared &= sample.roi_rgb.keys()
                if not shared:
                    break
            roi_names = [name for name in ROI_ORDER if name in shared]
            roi_names.extend(sorted(name for name in shared if name not in ROI_ORDER))
        for name in roi_names:
            columns.append(np.asarray([s.roi_rgb[name] for s in samples], dtype=np.float64))
        has_head = all(s.head_xy is not None for s in samples)
        if has_head:
            columns.append(np.asarray([s.head_xy for s in samples], dtype=np.float64))

        grid, resampled, fs = resample_uniform(timestamps, np.hstack(columns))
        resampled = np.atleast_2d(resampled)
        combined_rgb = resampled[:, :3]

        sources = [(COMBINED, combined_rgb)]
        for index, name in enumerate(roi_names):
            sources.append((name, resampled[:, 3 + 3 * index : 6 + 3 * index]))
        methods = ("pos", "chrom") if c.use_chrom else ("pos",)
        channels = [
            channel
            for source, rgb in sources
            for method in methods
            if (channel := self._channel(source, method, rgb, fs)) is not None
        ]
        if not channels:
            raise ValueError("no usable pulse channel in this window")

        freqs, _ = band_spectrum(
            channels[0].pulse,
            fs,
            c.band_low_hz,
            min(c.spectrum_high_hz, 0.48 * fs),
            nfft=c.welch_nfft,
        )

        # Channel weights from each channel's own peak clarity, smoothed over time.
        dt = 0.0
        if now is not None and self._last_update_t is not None:
            dt = max(0.0, now - self._last_update_t)
        alpha = 1.0 - np.exp(-dt / max(c.channel_weight_tau_s, 1e-3)) if dt > 0 else 1.0
        weights = []
        for channel in channels:
            instant = float(np.clip(10.0 ** (channel.own_snr_db / 10.0), 0.05, 30.0))
            previous = self._channel_weights.get(channel.key)
            weight = instant if previous is None else previous + alpha * (instant - previous)
            if update_weights:
                self._channel_weights[channel.key] = weight
            weights.append(weight)
        weights_arr = np.asarray(weights)
        stacked = np.stack([ch.power for ch in channels])
        spectrum = (weights_arr[:, None] * stacked).sum(axis=0) / weights_arr.sum()

        # Time-domain fusion for the waveform and beat detection: align polarity to the
        # strongest channel, then weight-average the unit-variance pulses.
        reference = channels[int(np.argmax(weights_arr))].pulse
        fused = np.zeros_like(reference)
        for weight, channel in zip(weights_arr, channels, strict=True):
            scale = float(np.std(channel.pulse))
            if scale <= 1e-12:
                continue
            sign = 1.0 if float(np.dot(channel.pulse, reference)) >= 0 else -1.0
            fused += weight * sign * channel.pulse / scale
        fused /= max(float(weights_arr.sum()), 1e-12)

        # Motion suppression: down-weight frequencies where the head is moving.
        tracking = spectrum.copy()
        motion_level = None
        if has_head:
            head = resampled[:, -2:]
            width = float(np.median([s.face_width_px for s in samples])) or 1.0
            try:
                dx = bandpass_filter(head[:, 0] - head[0, 0], fs, c.filter_low_hz, c.filter_high_hz)
                dy = bandpass_filter(head[:, 1] - head[0, 1], fs, c.filter_low_hz, c.filter_high_hz)
                motion_level = float(np.sqrt(np.var(dx) + np.var(dy)) / max(width, 1.0))
                if motion_level > c.motion_threshold:
                    _, mx = band_spectrum(
                        dx, fs, c.band_low_hz, freqs[-1] + 1e-9, nfft=c.welch_nfft
                    )
                    _, my = band_spectrum(
                        dy, fs, c.band_low_hz, freqs[-1] + 1e-9, nfft=c.welch_nfft
                    )
                    motion = mx[: freqs.size] + my[: freqs.size]
                    motion = motion / max(float(motion.max()), 1e-20)
                    strength = np.clip(
                        (motion_level - c.motion_threshold)
                        / max(c.motion_full_scale - c.motion_threshold, 1e-6),
                        0.0,
                        1.0,
                    )
                    tracking *= 1.0 - c.motion_max_suppression * strength * motion
            except ValueError:
                motion_level = None

        # Partial whitening: webcam noise is strongest at low frequencies (lighting drift,
        # small movements), which biases the tracker toward slow rates. Dividing by the
        # square root of a broad running-median floor favours narrow peaks wherever they are.
        flat = spectrum
        if freqs.size > 8:
            bin_hz = float(freqs[1] - freqs[0])
            width = max(3, int(round(c.whitening_width_hz / bin_hz)) | 1)
            floor = median_filter(spectrum, size=width, mode="nearest")
            floor = np.maximum(floor, 1e-3 * float(spectrum.max()) + 1e-20)
            flat = spectrum / floor**c.snr_whitening
            flat = flat / max(float(flat.sum()), 1e-20)
        if c.whitening > 0 and freqs.size > 8:
            bin_hz = float(freqs[1] - freqs[0])
            width = max(3, int(round(c.whitening_width_hz / bin_hz)) | 1)
            floor = median_filter(tracking, size=width, mode="nearest")
            floor = np.maximum(floor, 1e-3 * float(tracking.max()) + 1e-20)
            tracking = tracking / floor**c.whitening

        # Harmonic support: a true pulse at f also has power at 2f. Where 2f lies beyond
        # the analysed band, assume a typical harmonic so fast rates are not penalised
        # relative to their own sub-harmonic.
        if c.harmonic_weight > 0:
            beyond = 2.0 * freqs > freqs[-1]
            harmonic = np.interp(2.0 * freqs, freqs, tracking, left=0.0, right=0.0)
            harmonic = np.where(beyond, 0.3 * tracking, harmonic)
            tracking = tracking + c.harmonic_weight * harmonic
        search = freqs <= c.band_high_hz
        return PulseWindow(
            grid=grid,
            rgb=combined_rgb,
            fs=fs,
            pulse=fused,
            freqs=freqs,
            spectrum=spectrum,
            tracking_spectrum=np.where(search, tracking, 0.0),
            flat_spectrum=flat,
            channels=channels,
            motion_level=motion_level,
        )

    # -------------------------------------------------------------------- HRV

    def _hrv_metrics(self, now: float) -> dict[str, object]:
        c = self.config
        samples = self._window(now, c.hrv_window_s)
        duration = samples[-1].timestamp - samples[0].timestamp if samples else 0.0
        progress = float(np.clip(duration / max(c.min_hrv_window_s, 1e-6), 0.0, 1.0))
        if len(samples) < 16 or duration < c.min_hrv_window_s:
            self._hrv_cache = None
            self._hrv_history.clear()
            return _empty_hrv(progress)

        if self._hrv_cache is None or now - self._last_hrv_at >= c.hrv_interval_s:
            self._last_hrv_at = now
            try:
                self._hrv_cache = self._compute_hrv(samples)
            except (ValueError, FloatingPointError):
                self._hrv_cache = _empty_hrv()

        cache = dict(self._hrv_cache or _empty_hrv())
        cache["progress"] = progress
        return cache

    def _compute_hrv(self, samples: list[SignalSample]) -> dict[str, object]:
        c = self.config
        bpm = self._tracked_bpm
        if bpm is None or self._lock == "searching":
            return _empty_hrv()
        window = self._analyze_window(samples, update_weights=False)
        f0 = bpm / 60.0
        low = max(c.filter_low_hz, 0.6 * f0)
        high = min(c.filter_high_hz, 2.6 * f0, 0.48 * window.fs)
        pulse = bandpass_filter(window.pulse, window.fs, low, high, c.filter_order)
        peaks = detect_peaks(
            pulse,
            fs=window.fs,
            min_ibi_s=c.min_ibi_s,
            max_ibi_s=c.max_ibi_s,
            prominence_std=c.peak_prominence_std,
            max_ibi_deviation=c.max_ibi_deviation,
            min_valid_ibis=c.min_valid_ibis,
            expected_ibi_s=60.0 / bpm,
            upsample=c.peak_upsample,
        )
        if c.beat_template and len(peaks.peak_times) >= 6:
            wide = window.pulse * peaks.polarity
            times, corr = align_beats(wide, window.fs, peaks.peak_times)
            peaks = intervals_from_beats(
                times,
                min_ibi_s=max(c.min_ibi_s, 0.62 * 60.0 / bpm),
                max_ibi_s=min(c.max_ibi_s, 1.6 * 60.0 / bpm),
                max_ibi_deviation=c.max_ibi_deviation,
                min_valid_ibis=c.min_valid_ibis,
                polarity=peaks.polarity,
                beat_ok=corr >= c.beat_min_correlation,
            )
        timing_sd = self._timing_noise(window, peaks) if c.hrv_bias_correction else None
        expected_beats = max(1.0, (window.grid[-1] - window.grid[0]) * f0)
        quality = float(np.clip(peaks.ibi_count / expected_beats, 0.0, 1.0))
        rate_agrees = peaks.hr_bpm is not None and (
            abs(peaks.hr_bpm - bpm) <= c.hrv_rate_tolerance_bpm
        )
        snr = self._snr_ema if self._snr_ema is not None else -99.0
        ready = (
            peaks.ready
            and peaks.rmssd_ms is not None
            and snr >= c.min_snr_db
            and rate_agrees
            and quality >= 0.6
        )
        if not ready:
            cache = _empty_hrv()
            cache["ibi_count"] = int(peaks.ibi_count)
            cache["quality"] = quality
            cache["beats_abs"] = window.grid[0] + peaks.peak_times
            return cache
        rmssd = float(peaks.rmssd_ms)
        sdnn = peaks.sdnn_ms
        if timing_sd is not None:
            # Independent beat-timing errors add 6*sd^2 to RMSSD^2 and 2*sd^2 to SDNN^2.
            noise_ms2 = (timing_sd * 1000.0) ** 2
            rmssd = float(np.sqrt(max(rmssd**2 - 6.0 * noise_ms2, (0.35 * rmssd) ** 2)))
            if sdnn is not None:
                sdnn = float(np.sqrt(max(sdnn**2 - 2.0 * noise_ms2, (0.5 * sdnn) ** 2)))
        self._hrv_history.append(rmssd)
        return {
            "ready": True,
            "ibi_count": int(peaks.ibi_count),
            "rmssd_ms": float(np.median(self._hrv_history)),
            "sdnn_ms": sdnn,
            "timing_sd_ms": None if timing_sd is None else timing_sd * 1000.0,
            "pnn50": peaks.pnn50,
            "mean_ibi_ms": peaks.mean_ibi_ms,
            "quality": quality,
            "ibi_times_abs": window.grid[0] + peaks.ibi_times,
            "ibis_ms": peaks.ibis * 1000.0,
            "beats_abs": window.grid[0] + peaks.peak_times,
        }

    def _timing_noise(self, window: PulseWindow, peaks: object) -> float | None:
        """Estimate beat-timing noise (s) of the fused pulse from the disagreement
        between beats timed on the forehead alone and on the cheeks alone. Camera noise
        is independent between regions, so SD(t_forehead - t_cheeks) ~ 2 * SD(t_fused) for an
        ideal fusion; ``hrv_timing_noise_factor`` (0.6) was calibrated on synthetic data.
        """
        times = np.asarray(getattr(peaks, "peak_times", np.empty(0)))
        if times.size < 10:
            return None
        groups: dict[str, list[Channel]] = {"forehead": [], "cheeks": []}
        for channel in window.channels:
            if channel.source == "forehead":
                groups["forehead"].append(channel)
            elif channel.source.endswith("cheek"):
                groups["cheeks"].append(channel)
        if not groups["forehead"] or not groups["cheeks"]:
            return None
        reference = window.pulse * getattr(peaks, "polarity", 1)
        timed = []
        for members in groups.values():
            combined = np.zeros_like(reference)
            for channel in members:
                weight = float(np.clip(10.0 ** (channel.own_snr_db / 10.0), 0.05, 30.0))
                scale = float(np.std(channel.pulse)) or 1.0
                sign = 1.0 if float(np.dot(channel.pulse, reference)) >= 0 else -1.0
                combined += weight * sign * channel.pulse / scale
            aligned, corr = align_beats(combined, window.fs, times, iterations=1)
            timed.append((aligned, corr))
        (ta, ca), (tb, cb) = timed
        keep = (ca >= self.config.beat_min_correlation) & (cb >= self.config.beat_min_correlation)
        if np.count_nonzero(keep) < 8:
            return None
        diff = (ta - tb)[keep]
        diff = diff - np.median(diff)
        robust_sd = 1.4826 * float(np.median(np.abs(diff)))
        return self.config.hrv_timing_noise_factor * robust_sd

    # ------------------------------------------------------------ respiration

    def _respiration_metrics(self, now: float) -> dict[str, object]:
        c = self.config
        samples = self._window(now, c.respiration_window_s)
        duration = samples[-1].timestamp - samples[0].timestamp if samples else 0.0
        progress = float(np.clip(duration / max(c.min_respiration_window_s, 1e-6), 0.0, 1.0))
        empty = {
            "bpm": None,
            "snr_db": None,
            "ready": False,
            "progress": progress,
            "t": np.empty(0),
            "y": np.empty(0),
            "sources": [],
        }
        if len(samples) < 32 or duration < c.min_respiration_window_s:
            self._respiration_cache = None
            self._resp_tracker.reset()
            self._resp_smoother.reset()
            return empty

        if (
            self._respiration_cache is None
            or now - self._last_respiration_at >= c.respiration_interval_s
        ):
            self._last_respiration_at = now
            try:
                self._respiration_cache = self._compute_respiration(samples, now)
            except (ValueError, FloatingPointError):
                self._respiration_cache = None

        if self._respiration_cache is None:
            return empty
        cache = dict(self._respiration_cache)
        cache["progress"] = progress
        cache["t"] = np.asarray(cache.pop("t_abs")) - now
        return cache

    def _compute_respiration(
        self, samples: list[SignalSample], now: float
    ) -> dict[str, object] | None:
        c = self.config
        low, high = c.respiration_low_hz, c.respiration_high_hz
        timestamps = np.asarray([s.timestamp for s in samples], dtype=np.float64)
        columns = [np.asarray([s.rgb for s in samples], dtype=np.float64)]
        has_head = all(s.head_xy is not None for s in samples)
        if has_head:
            columns.append(np.asarray([s.head_xy for s in samples], dtype=np.float64))
        grid, resampled, fs = resample_uniform(timestamps, np.hstack(columns))
        resampled = np.atleast_2d(resampled)
        rgb = resampled[:, :3]

        sources: list[RespSource] = []
        green = rgb[:, 1] / max(float(np.mean(rgb[:, 1])), 1e-6)
        intensity = make_source("intensity", grid, green, fs, low, high)
        if intensity is not None:
            sources.append(intensity)
        if has_head:
            width = float(np.median([s.face_width_px for s in samples])) or 1.0
            head = make_source("head", grid, resampled[:, 4] / width, fs, low, high)
            if head is not None:
                sources.append(head)

        bpm = self._tracked_bpm
        if bpm is not None and self._lock != "searching":
            pulse = None
            try:
                pulse_rgb = pos(rgb, fs=fs, window_s=c.pos_window_s)
                pulse = bandpass_filter(
                    robust_clip(pulse_rgb, c.artifact_clip_sigma),
                    fs,
                    max(c.filter_low_hz, 0.6 * bpm / 60.0),
                    min(c.filter_high_hz, 2.6 * bpm / 60.0, 0.48 * fs),
                    c.filter_order,
                )
            except ValueError:
                pulse = None
            if pulse is not None:
                peaks = detect_peaks(pulse, fs, expected_ibi_s=60.0 / bpm, min_valid_ibis=4)
                series = beat_series(peaks.peak_times, pulse, fs)
                if series is not None:
                    beat_grid, amplitude, interval = series
                    beat_high = min(high, 0.45 * bpm / 60.0)
                    for name, values in (("pulse_amplitude", amplitude), ("pulse_rate", interval)):
                        source = make_source(name, grid[0] + beat_grid, values, 4.0, low, beat_high)
                        if source is not None:
                            sources.append(source)

        if not sources:
            return None
        fused_f, fused_p = fuse(sources, low, high)
        if float(fused_p.max()) <= 0:
            return None
        state = self._resp_tracker.update(fused_f * 60.0, fused_p, now)
        rate = refine_near(fused_f * 60.0, fused_p, state.rate, 2.5)
        quality = max(s.quality for s in sources)
        smoothed = self._resp_smoother.update(rate, now, quality, restart=state.switched)
        agreeing = [s for s in sources if s.quality >= 0.12 and abs(s.rate_bpm - smoothed) <= 3.0]
        snr_db = snr_at(
            fused_f, fused_p, smoothed / 60.0, half_bandwidth_hz=0.05, harmonic_bandwidth_hz=0.0
        )
        best = max(sources, key=lambda s: s.quality)
        concentration = float(
            fused_p[np.abs(fused_f - smoothed / 60.0) <= 0.05].sum() / max(fused_p.sum(), 1e-20)
        )
        ready = (
            snr_db >= c.respiration_min_snr_db
            and concentration >= c.respiration_min_quality
            and (len(agreeing) >= 2 or best.quality >= 0.45)
        )
        return {
            "bpm": float(smoothed),
            "snr_db": float(snr_db),
            "ready": bool(ready),
            "t_abs": best.t,
            "y": best.y,
            "sources": [
                {"name": s.name, "rate": round(s.rate_bpm, 1), "quality": round(s.quality, 3)}
                for s in sources
            ],
        }

    # ---------------------------------------------------------------- analyze

    def _status_result(
        self, now: float, start: float, status: str, message: str, **extra: object
    ) -> AnalysisResult:
        result = AnalysisResult(
            timestamp=now,
            status=status,
            message=message,
            analysis_ms=(time.perf_counter() - start) * 1000.0,
            **extra,
        )
        self._last_result = result
        return result

    def _update_lock(
        self, now: float, snr: float, agreement: float, stable_s: float, switched: bool
    ) -> None:
        c = self.config
        if switched and self._lock != "searching":
            self._lock = "searching"
            self._hold_since = None
            return
        clear = snr >= c.min_snr_db and stable_s >= c.lock_confirm_s
        # Slow rates sit where webcam noise is strongest: they need the clear path.
        slow = self._tracked_bpm is not None and self._tracked_bpm < c.slow_rate_bpm
        assisted = (
            not slow
            and snr >= c.assisted_snr_db
            and agreement >= c.min_agreement
            and stable_s >= c.assisted_stable_s
        )
        lost = snr < c.assisted_snr_db - c.lock_hysteresis_db or (
            agreement < 0.6 * c.min_agreement and snr < c.min_snr_db - c.lock_hysteresis_db
        )
        if self._lock == "searching":
            if clear or assisted:
                self._lock = "locked"
        elif self._lock == "locked":
            if lost:
                self._lock = "holding"
                self._hold_since = now
        elif self._lock == "holding":
            if clear or assisted:
                self._lock = "locked"
                self._hold_since = None
            elif self._hold_since is not None and now - self._hold_since > c.hold_s:
                self._lock = "searching"
                self._hold_since = None

    def analyze(self, timestamp: float | None = None, force: bool = False) -> AnalysisResult:
        c = self.config
        now = float(timestamp if timestamp is not None else time.perf_counter())
        if not force and (now - self._last_analysis) < c.analysis_interval_s:
            return self._last_result
        self._last_analysis = now
        start = time.perf_counter()

        latest_sample_age = now - self._samples[-1].timestamp if self._samples else float("inf")
        if not self._face_present:
            if latest_sample_age > c.reset_after_loss_s:
                self._clear_caches()
            return self._status_result(
                now, start, "no_face", self._face_reason or "No face detected"
            )
        if not self._face_pose_ok:
            if latest_sample_age > c.reset_after_loss_s:
                self._clear_caches()
            return self._status_result(
                now, start, "adjust_pose", self._face_reason or "Face the camera"
            )
        if latest_sample_age > 5.0:
            self._clear_caches()
        if latest_sample_age > c.reset_after_loss_s:
            return self._status_result(
                now,
                start,
                "low_signal",
                self._face_reason or "Signal lost — hold still and keep your face visible",
            )

        hr_samples = self._window(now, c.hr_window_s)
        if not hr_samples:
            return self._status_result(
                now, start, "acquiring", self._face_reason or "Acquiring signal"
            )

        window_s = hr_samples[-1].timestamp - hr_samples[0].timestamp
        if len(hr_samples) < 16 or window_s < c.min_hr_window_s:
            progress = float(np.clip(window_s / max(c.min_hr_window_s, 1e-6), 0, 1))
            return self._status_result(
                now,
                start,
                "acquiring",
                self._face_reason or "Acquiring signal",
                confidence=0.25 * progress,
                window_s=float(window_s),
                sample_count=len(hr_samples),
            )

        try:
            window = self._analyze_window(hr_samples, now=now)
        except (ValueError, FloatingPointError) as exc:
            return self._status_result(now, start, "low_signal", f"Signal unavailable: {exc}")

        dt = 0.0 if self._last_update_t is None else max(0.0, now - self._last_update_t)
        self._last_update_t = now

        bpm_axis = window.freqs * 60.0
        state = self._tracker.update(bpm_axis, window.tracking_spectrum, now)
        raw_bpm = refine_near(bpm_axis, window.spectrum, state.rate, c.refine_span_bpm)
        raw_bpm = float(np.clip(raw_bpm, c.band_low_hz * 60.0, c.band_high_hz * 60.0))
        f0 = raw_bpm / 60.0
        snr_db = snr_at(window.freqs, window.flat_spectrum, f0)
        at_edge = raw_bpm <= c.band_low_hz * 60.0 + 1.0 or raw_bpm >= c.band_high_hz * 60.0 - 1.0

        smoothing = 1.0 if self._snr_ema is None else 1.0 - np.exp(-dt / max(c.snr_tau_s, 1e-3))
        self._snr_ema = (
            snr_db
            if self._snr_ema is None
            else self._snr_ema + smoothing * (snr_db - self._snr_ema)
        )
        tolerance_hz = c.agreement_tolerance_bpm / 60.0
        agreement_now = float(
            np.mean([abs(ch.own_peak_hz - f0) <= tolerance_hz for ch in window.channels])
        )
        self._agreement_ema = (
            agreement_now
            if self._agreement_ema is None
            else self._agreement_ema + smoothing * (agreement_now - self._agreement_ema)
        )
        previous_lock = self._lock
        self._update_lock(
            now,
            self._snr_ema - (3.0 if at_edge else 0.0),
            self._agreement_ema,
            state.stable_s,
            state.switched,
        )

        mean_quality = float(np.mean([s.quality for s in hr_samples]))
        snr_score = float(
            np.clip(
                (self._snr_ema - c.confidence_snr_low_db)
                / max(c.good_snr_db - c.confidence_snr_low_db, 1e-6),
                0.0,
                1.0,
            )
        )
        margin_score = float(np.clip(state.margin / c.tracker_jump_cost, 0.0, 1.0))
        duration_score = float(np.clip(window_s / c.hr_window_s, 0.0, 1.0))
        motion_penalty = 0.0
        if window.motion_level is not None:
            motion_penalty = float(
                np.clip(
                    (window.motion_level - c.motion_threshold)
                    / max(c.motion_full_scale - c.motion_threshold, 1e-6),
                    0.0,
                    1.0,
                )
            )
        instant_confidence = float(
            np.clip(
                (
                    0.4 * snr_score
                    + 0.2 * self._agreement_ema
                    + 0.15 * margin_score
                    + 0.15 * mean_quality
                    + 0.1 * duration_score
                )
                * (1.0 - 0.4 * motion_penalty),
                0.0,
                1.0,
            )
        )
        self._confidence_ema = (
            instant_confidence
            if self._confidence_ema is None
            else self._confidence_ema + smoothing * (instant_confidence - self._confidence_ema)
        )
        confidence = float(self._confidence_ema)

        restart = state.switched or (previous_lock == "searching" and self._lock == "locked")
        smoothed = self._smoother.update(
            raw_bpm, now, snr_score * (1.0 - 0.5 * motion_penalty), restart
        )
        self._tracked_bpm = smoothed

        bpm: float | None = None
        provisional_bpm: float | None = None
        if self._lock == "locked":
            good = self._snr_ema >= c.min_snr_db and mean_quality >= 0.45 and confidence >= 0.6
            status, message = (
                ("good", "Good signal") if good else ("measuring", "Measuring — hold still")
            )
            if motion_penalty > 0.5:
                message = "Movement detected — hold still"
            bpm = smoothed
        elif self._lock == "holding":
            status = "measuring"
            message = "Holding last reading — signal briefly weak"
            bpm = smoothed
        else:
            status = "low_signal"
            message = (
                "Movement detected — hold still"
                if motion_penalty > 0.5
                else "Low signal — improve lighting and hold still"
            )
            if self._snr_ema >= c.provisional_min_snr_db and not at_edge:
                provisional_bpm = smoothed

        luma_values = np.asarray([s.luma for s in hr_samples], dtype=np.float64)
        luma_mean = float(np.mean(luma_values)) if luma_values.size else None
        luma_std = float(np.std(luma_values)) if luma_values.size > 1 else None

        hrv = self._hrv_metrics(now)
        respiration = self._respiration_metrics(now)
        hrv_ready = bool(hrv.get("ready"))

        beats_abs = np.asarray(hrv.get("beats_abs", np.empty(0)))
        if not len(beats_abs):
            peaks = detect_peaks(
                window.pulse, fs=window.fs, expected_ibi_s=60.0 / smoothed, min_valid_ibis=2
            )
            beats_abs = peaks.peak_times + window.grid[0]
        ibi_times_abs = np.asarray(hrv.get("ibi_times_abs", np.empty(0)))
        relative_t = window.grid - now

        roi_quality = []
        for channel in window.channels:
            if channel.method != "pos" or channel.source == COMBINED:
                continue
            roi_quality.append(
                {
                    "name": channel.source,
                    "bpm": round(channel.own_peak_hz * 60.0, 1),
                    "snr_db": round(snr_at(window.freqs, channel.power, f0), 2),
                    "pixels": int(
                        np.median([s.roi_pixels_by_name.get(channel.source, 0) for s in hr_samples])
                    ),
                }
            )

        result = AnalysisResult(
            timestamp=now,
            status=status,
            message=message,
            bpm=bpm,
            provisional_bpm=provisional_bpm,
            snr_db=float(self._snr_ema),
            rmssd_ms=hrv.get("rmssd_ms") if hrv_ready else None,
            hrv_ready=hrv_ready,
            hrv_progress=float(hrv.get("progress", 0.0)),
            confidence=confidence,
            fs=float(window.fs),
            window_s=float(window_s),
            sample_count=len(hr_samples),
            ibi_count=int(hrv.get("ibi_count", 0)),
            waveform_t=relative_t,
            waveform_y=_smooth_for_display(window.pulse, window.fs),
            psd_f=window.freqs,
            psd_p=window.spectrum,
            raw_t=relative_t,
            raw_rgb=window.rgb,
            beats=beats_abs,
            analysis_ms=(time.perf_counter() - start) * 1000.0,
            hrv_sdnn_ms=hrv.get("sdnn_ms") if hrv_ready else None,
            hrv_pnn50=hrv.get("pnn50") if hrv_ready else None,
            hrv_mean_ibi_ms=hrv.get("mean_ibi_ms") if hrv_ready else None,
            ibi_times=ibi_times_abs if len(ibi_times_abs) else None,
            ibis_ms=np.asarray(hrv.get("ibis_ms", np.empty(0))) if hrv_ready else None,
            roi_quality=roi_quality,
            luma_mean=luma_mean,
            luma_std=luma_std,
            respiration_bpm=respiration.get("bpm"),
            respiration_snr_db=respiration.get("snr_db"),
            respiration_ready=bool(respiration.get("ready")),
            respiration_progress=float(respiration.get("progress", 0.0)),
            respiration_t=np.asarray(respiration.get("t", np.empty(0))),
            respiration_y=np.asarray(respiration.get("y", np.empty(0))),
            respiration_sources=list(respiration.get("sources", [])),
            motion_level=window.motion_level,
            lock_state=self._lock,
            hrv_quality=hrv.get("quality"),
        )
        self._last_result = result
        return result


def _smooth_for_display(signal: np.ndarray, fs: float) -> np.ndarray:
    window = int(round(0.15 * fs))
    if window % 2 == 0:
        window += 1
    if window < 5 or len(signal) < window:
        return signal
    try:
        return savgol_filter(signal, window_length=window, polyorder=2, mode="interp")
    except ValueError:
        return signal
