from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter

from .config import SignalConfig
from .dsp import (
    HREstimate,
    bandpass_filter,
    detect_peaks,
    estimate_hr,
    pos,
    resample_uniform,
    robust_clip,
)
from .types import AnalysisResult, SignalSample

ROI_ORDER = ("forehead", "right_cheek", "left_cheek")


@dataclass
class PulseWindow:
    grid: np.ndarray
    rgb: np.ndarray
    fs: float
    pulse: np.ndarray
    roi_quality: list[dict[str, object]] = field(default_factory=list)


def _empty_hrv(progress: float = 0.0) -> dict[str, object]:
    return {
        "ready": False,
        "progress": progress,
        "ibi_count": 0,
        "rmssd_ms": None,
        "sdnn_ms": None,
        "pnn50": None,
        "mean_ibi_ms": None,
        "ibi_times_abs": np.empty(0),
        "ibis_ms": np.empty(0),
        "beats_abs": np.empty(0),
    }


class SignalPipeline:
    def __init__(self, config: SignalConfig | None = None) -> None:
        self.config = config or SignalConfig()
        self._samples: deque[SignalSample] = deque()
        self._bpm_history: deque[float] = deque(maxlen=max(1, self.config.bpm_smoothing))
        self._provisional_history: deque[float] = deque(maxlen=max(1, self.config.bpm_smoothing))
        self._hrv_history: deque[float] = deque(maxlen=max(1, self.config.hrv_smoothing))
        self._prior_overrides = 0
        self._last_analysis = 0.0
        self._last_hrv_at = 0.0
        self._last_respiration_at = 0.0
        self._hrv_cache: dict[str, object] | None = None
        self._respiration_cache: dict[str, object] | None = None
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
        self._bpm_history.clear()
        self._provisional_history.clear()
        self._hrv_history.clear()
        self._prior_overrides = 0
        self._hrv_cache = None
        self._respiration_cache = None
        self._last_hrv_at = 0.0
        self._last_respiration_at = 0.0

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

    def _prior_hz(self) -> float | None:
        if len(self._bpm_history) < 3:
            return None
        return float(np.median(self._bpm_history)) / 60.0

    def _spectrum(self, signal: np.ndarray, fs: float, prior_hz: float | None = None) -> HREstimate:
        return estimate_hr(
            signal,
            fs=fs,
            low_hz=self.config.band_low_hz,
            high_hz=self.config.band_high_hz,
            nperseg=self.config.welch_nperseg,
            nfft=self.config.welch_nfft,
            prior_hz=prior_hz,
            prior_tolerance_hz=self.config.prior_tolerance_hz,
            prior_min_relative_power=self.config.prior_min_relative_power,
        )

    def _channel_pulse(self, rgb: np.ndarray, fs: float) -> np.ndarray:
        pulse = pos(rgb, fs=fs, window_s=self.config.pos_window_s)
        pulse = robust_clip(pulse, self.config.artifact_clip_sigma)
        return bandpass_filter(
            pulse,
            fs=fs,
            low_hz=self.config.band_low_hz,
            high_hz=self.config.band_high_hz,
            order=self.config.filter_order,
        )

    def _analyze_window(
        self, samples: list[SignalSample], prior_hz: float | None = None
    ) -> PulseWindow:
        timestamps = np.asarray([s.timestamp for s in samples], dtype=np.float64)
        combined = np.asarray([s.rgb for s in samples], dtype=np.float64)

        roi_names: list[str] = []
        if self.config.fuse_rois:
            shared = set(samples[0].roi_rgb)
            for sample in samples[1:]:
                shared &= sample.roi_rgb.keys()
                if not shared:
                    break
            roi_names = [name for name in ROI_ORDER if name in shared]
            roi_names.extend(sorted(name for name in shared if name not in ROI_ORDER))

        columns = [combined]
        for name in roi_names:
            columns.append(np.asarray([s.roi_rgb[name] for s in samples], dtype=np.float64))
        grid, resampled, fs = resample_uniform(timestamps, np.hstack(columns))
        resampled = np.atleast_2d(resampled)
        combined_rgb = resampled[:, :3]
        combined_pulse = self._channel_pulse(combined_rgb, fs)
        if not roi_names:
            return PulseWindow(grid=grid, rgb=combined_rgb, fs=fs, pulse=combined_pulse)

        pulses = [combined_pulse]
        weights: list[float] = []
        roi_quality: list[dict[str, object]] = []
        for index, name in enumerate(roi_names):
            block = resampled[:, 3 + 3 * index : 6 + 3 * index]
            try:
                channel = self._channel_pulse(block, fs)
                estimate = self._spectrum(channel, fs, prior_hz)
            except (ValueError, FloatingPointError):
                continue
            pixels = [s.roi_pixels_by_name.get(name, 0) for s in samples]
            roi_quality.append(
                {
                    "name": name,
                    "bpm": round(float(estimate.bpm), 1),
                    "snr_db": round(float(estimate.snr_db), 2),
                    "pixels": int(np.median(pixels)) if pixels else 0,
                }
            )
            if estimate.peak_at_edge or estimate.snr_db < self.config.fusion_min_snr_db:
                continue
            pulses.append(channel)
            weights.append(10.0 ** (estimate.snr_db / 10.0))

        try:
            combined_estimate = self._spectrum(combined_pulse, fs, prior_hz)
            combined_weight = (
                0.0 if combined_estimate.peak_at_edge else 10.0 ** (combined_estimate.snr_db / 10.0)
            )
        except (ValueError, FloatingPointError):
            combined_weight = 0.0
        weights.insert(0, combined_weight)

        total = float(sum(weights))
        if total <= 1e-9 or len(pulses) == 1:
            return PulseWindow(
                grid=grid, rgb=combined_rgb, fs=fs, pulse=combined_pulse, roi_quality=roi_quality
            )
        fused = np.zeros_like(combined_pulse)
        for weight, channel in zip(weights, pulses, strict=True):
            scale = float(np.std(channel))
            if scale > 1e-12 and weight > 0:
                fused += weight * channel / scale
        fused /= total
        return PulseWindow(grid=grid, rgb=combined_rgb, fs=fs, pulse=fused, roi_quality=roi_quality)

    def _hrv_metrics(self, now: float) -> dict[str, object]:
        samples = self._window(now, self.config.hrv_window_s)
        duration = samples[-1].timestamp - samples[0].timestamp if samples else 0.0
        progress = float(np.clip(duration / max(self.config.min_hrv_window_s, 1e-6), 0.0, 1.0))
        if len(samples) < 16 or duration < self.config.min_hrv_window_s:
            self._hrv_cache = None
            self._hrv_history.clear()
            return _empty_hrv(progress)

        if self._hrv_cache is None or now - self._last_hrv_at >= self.config.hrv_interval_s:
            try:
                window = self._analyze_window(samples, self._prior_hz())
                estimate = self._spectrum(window.pulse, window.fs, self._prior_hz())
                peaks = detect_peaks(
                    window.pulse,
                    fs=window.fs,
                    min_ibi_s=self.config.min_ibi_s,
                    max_ibi_s=self.config.max_ibi_s,
                    prominence_std=self.config.peak_prominence_std,
                    max_ibi_deviation=self.config.max_ibi_deviation,
                    min_valid_ibis=self.config.min_valid_ibis,
                )
                rate_agrees = peaks.hr_bpm is not None and (
                    abs(peaks.hr_bpm - estimate.bpm) <= self.config.hrv_rate_tolerance_bpm
                )
                ready = (
                    peaks.ready
                    and peaks.rmssd_ms is not None
                    and estimate.snr_db >= self.config.min_snr_db
                    and rate_agrees
                )
                if ready:
                    self._hrv_history.append(float(peaks.rmssd_ms))
                    self._hrv_cache = {
                        "ready": True,
                        "ibi_count": int(peaks.ibi_count),
                        "rmssd_ms": float(np.median(self._hrv_history)),
                        "sdnn_ms": peaks.sdnn_ms,
                        "pnn50": peaks.pnn50,
                        "mean_ibi_ms": peaks.mean_ibi_ms,
                        "ibi_times_abs": window.grid[0] + peaks.ibi_times,
                        "ibis_ms": peaks.ibis * 1000.0,
                        "beats_abs": window.grid[0] + peaks.peak_times,
                    }
                else:
                    cache = _empty_hrv()
                    cache["ibi_count"] = int(peaks.ibi_count)
                    self._hrv_cache = cache
            except (ValueError, FloatingPointError):
                self._hrv_cache = _empty_hrv()
            self._last_hrv_at = now

        cache = dict(self._hrv_cache or _empty_hrv())
        cache["progress"] = progress
        return cache

    def _respiration_metrics(self, now: float) -> dict[str, object]:
        samples = self._window(now, self.config.respiration_window_s)
        duration = samples[-1].timestamp - samples[0].timestamp if samples else 0.0
        progress = float(
            np.clip(duration / max(self.config.min_respiration_window_s, 1e-6), 0.0, 1.0)
        )
        empty = {
            "bpm": None,
            "snr_db": None,
            "ready": False,
            "progress": progress,
            "t": np.empty(0),
            "y": np.empty(0),
        }
        if len(samples) < 16 or duration < self.config.min_respiration_window_s:
            self._respiration_cache = None
            return empty

        if (
            self._respiration_cache is None
            or now - self._last_respiration_at >= self.config.respiration_interval_s
        ):
            try:
                timestamps = np.asarray([s.timestamp for s in samples], dtype=np.float64)
                rgb = np.asarray([s.rgb for s in samples], dtype=np.float64)
                grid, resampled, fs = resample_uniform(timestamps, rgb)
                green = resampled[:, 1] / max(float(np.mean(resampled[:, 1])), 1e-6)
                filtered = bandpass_filter(
                    green,
                    fs=fs,
                    low_hz=self.config.respiration_low_hz,
                    high_hz=self.config.respiration_high_hz,
                    order=2,
                )
                estimate = estimate_hr(
                    filtered,
                    fs=fs,
                    low_hz=self.config.respiration_low_hz,
                    high_hz=self.config.respiration_high_hz,
                    nperseg=min(1024, len(filtered)),
                    nfft=8192,
                    half_bandwidth_hz=0.05,
                )
                scale = float(np.std(filtered))
                waveform = filtered / scale if scale > 1e-12 else filtered
                self._respiration_cache = {
                    "bpm": float(estimate.bpm),
                    "snr_db": float(estimate.snr_db),
                    "ready": bool(
                        not estimate.peak_at_edge
                        and estimate.snr_db >= self.config.respiration_min_snr_db
                    ),
                    "t_abs": grid,
                    "y": waveform,
                }
            except (ValueError, FloatingPointError):
                self._respiration_cache = None
            self._last_respiration_at = now

        if self._respiration_cache is None:
            return empty
        cache = dict(self._respiration_cache)
        cache["progress"] = progress
        cache["t"] = np.asarray(cache.pop("t_abs")) - now
        return cache

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

    def analyze(self, timestamp: float | None = None, force: bool = False) -> AnalysisResult:
        now = float(timestamp if timestamp is not None else time.perf_counter())
        if not force and (now - self._last_analysis) < self.config.analysis_interval_s:
            return self._last_result
        self._last_analysis = now
        start = time.perf_counter()

        latest_sample_age = now - self._samples[-1].timestamp if self._samples else float("inf")
        if not self._face_present:
            if latest_sample_age > self.config.reset_after_loss_s:
                self._clear_caches()
            return self._status_result(
                now, start, "no_face", self._face_reason or "No face detected"
            )
        if not self._face_pose_ok:
            if latest_sample_age > self.config.reset_after_loss_s:
                self._clear_caches()
            return self._status_result(
                now, start, "adjust_pose", self._face_reason or "Face the camera"
            )
        if latest_sample_age > 5.0:
            self._clear_caches()
        if latest_sample_age > self.config.reset_after_loss_s:
            return self._status_result(
                now,
                start,
                "low_signal",
                self._face_reason or "Signal lost — hold still and keep your face visible",
            )

        hr_samples = self._window(now, self.config.hr_window_s)
        if not hr_samples:
            return self._status_result(
                now, start, "acquiring", self._face_reason or "Acquiring signal"
            )

        window_s = hr_samples[-1].timestamp - hr_samples[0].timestamp
        if len(hr_samples) < 16 or window_s < self.config.min_hr_window_s:
            progress = float(np.clip(window_s / max(self.config.min_hr_window_s, 1e-6), 0, 1))
            return self._status_result(
                now,
                start,
                "acquiring",
                self._face_reason or "Acquiring signal",
                confidence=0.25 * progress,
                window_s=float(window_s),
                sample_count=len(hr_samples),
            )

        prior_hz = self._prior_hz()
        try:
            window = self._analyze_window(hr_samples, prior_hz)
            hr = self._spectrum(window.pulse, window.fs, prior_hz)
            peaks = detect_peaks(
                window.pulse,
                fs=window.fs,
                min_ibi_s=self.config.min_ibi_s,
                max_ibi_s=self.config.max_ibi_s,
                prominence_std=self.config.peak_prominence_std,
                max_ibi_deviation=self.config.max_ibi_deviation,
                min_valid_ibis=self.config.min_valid_ibis,
            )
        except (ValueError, FloatingPointError) as exc:
            return self._status_result(now, start, "low_signal", f"Signal unavailable: {exc}")

        self._prior_overrides = self._prior_overrides + 1 if hr.used_prior else 0
        if self._prior_overrides > self.config.max_prior_overrides:
            self._bpm_history.clear()
            self._prior_overrides = 0

        mean_quality = float(np.mean([s.quality for s in hr_samples]))
        snr_score = np.clip(
            (hr.snr_db - self.config.min_snr_db)
            / max(self.config.good_snr_db - self.config.min_snr_db, 1e-6),
            0.0,
            1.0,
        )
        duration_score = np.clip(window_s / self.config.hr_window_s, 0.0, 1.0)
        confidence = float(
            np.clip(0.55 * snr_score + 0.30 * mean_quality + 0.15 * duration_score, 0.0, 1.0)
        )

        if hr.snr_db < self.config.min_snr_db or hr.peak_at_edge:
            status = "low_signal"
            message = "Low signal — improve lighting and hold still"
        elif hr.snr_db >= self.config.good_snr_db and mean_quality >= 0.45 and confidence >= 0.55:
            status = "good"
            message = "Good signal"
        else:
            status = "measuring"
            message = "Measuring — hold still"

        bpm: float | None = None
        provisional_bpm: float | None = None
        if status in ("measuring", "good"):
            self._bpm_history.append(hr.bpm)
            self._provisional_history.clear()
            bpm = float(np.median(self._bpm_history))
        elif not hr.peak_at_edge and hr.snr_db >= self.config.provisional_min_snr_db:
            self._provisional_history.append(hr.bpm)
            provisional_bpm = float(np.median(self._provisional_history))

        luma_values = np.asarray([s.luma for s in hr_samples], dtype=np.float64)
        luma_mean = float(np.mean(luma_values)) if luma_values.size else None
        luma_std = float(np.std(luma_values)) if luma_values.size > 1 else None

        hrv = self._hrv_metrics(now)
        respiration = self._respiration_metrics(now)
        hrv_ready = bool(hrv.get("ready"))

        hrv_beats = np.asarray(hrv.get("beats_abs", np.empty(0)))
        beats_abs = hrv_beats if len(hrv_beats) else peaks.peak_times + window.grid[0]
        ibi_times_abs = np.asarray(hrv.get("ibi_times_abs", np.empty(0)))
        relative_t = window.grid - now
        result = AnalysisResult(
            timestamp=now,
            status=status,
            message=message,
            bpm=bpm,
            provisional_bpm=provisional_bpm,
            snr_db=float(hr.snr_db),
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
            psd_f=hr.freqs,
            psd_p=hr.power,
            raw_t=relative_t,
            raw_rgb=window.rgb,
            beats=beats_abs,
            analysis_ms=(time.perf_counter() - start) * 1000.0,
            hrv_sdnn_ms=hrv.get("sdnn_ms") if hrv_ready else None,
            hrv_pnn50=hrv.get("pnn50") if hrv_ready else None,
            hrv_mean_ibi_ms=hrv.get("mean_ibi_ms") if hrv_ready else None,
            ibi_times=ibi_times_abs if len(ibi_times_abs) else None,
            ibis_ms=np.asarray(hrv.get("ibis_ms", np.empty(0))) if hrv_ready else None,
            roi_quality=window.roi_quality,
            luma_mean=luma_mean,
            luma_std=luma_std,
            respiration_bpm=respiration.get("bpm"),
            respiration_snr_db=respiration.get("snr_db"),
            respiration_ready=bool(respiration.get("ready")),
            respiration_progress=float(respiration.get("progress", 0.0)),
            respiration_t=np.asarray(respiration.get("t", np.empty(0))),
            respiration_y=np.asarray(respiration.get("y", np.empty(0))),
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
