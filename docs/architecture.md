# Clinura architecture

## Runtime topology

The Python engine owns the camera and all signal processing. The browser is a thin,
polished renderer.

```text
Camera thread ──► latest-frame queue ──► processing thread
                                           ├─ FaceLandmarker
                                           ├─ ROI masks + RGB means
                                           ├─ SignalPipeline (POS, filters, PSD, peaks)
                                           ├─ SessionRecorder (optional)
                                           └─ latest preview + metrics
                                                     │
                                              FastAPI /ws
                                              ├─ binary JPEG frames (30 fps)
                                              └─ JSON metrics (10 Hz)
                                                     │
                                               React UI
```

Why not send browser camera frames to the server? Browser/WebRTC/JPEG compression destroys
the tiny pulsatile color change. Capturing in Python keeps frames uncompressed for analysis
and gives access to native camera controls (when the platform exposes them).

## Capture

- `cv2.VideoCapture` with `CAP_AVFOUNDATION` on macOS.
- A background thread reads frames continuously and publishes the latest frame through a
  size-1 queue. Slow consumers drop stale frames instead of building latency.
- Each frame gets `time.perf_counter()` at capture and is optionally mirrored for a selfie
  view.
- The recorder also stores `time.time()` (epoch) per frame. Analysis timestamps are mapped
  back to epoch wall time during replay, which lets Apple Watch/Polar/ECG readings be
  compared by nearest timestamp.

## Face and ROI

`FaceLandmarker` runs in VIDEO mode on a downscaled copy (default max width 1280) for speed.
Because the resize preserves aspect ratio, normalized landmarks map directly to the full-res
frame used for RGB extraction. A second IMAGE-mode landmarker is used as a fallback on a
center crop (or around the last face box) upscaled to 1280 px; this helps small/distant
faces.

ROIs are masks, not fixed rectangles:

- **Face oval** from the canonical 36-point MediaPipe contour.
- **Forehead**: the top band of the face oval, inset horizontally, eroded.
- **Cheeks**: ellipses centered on the line between the eye center and the mouth corner,
  shifted outward, intersected with the face mask, then excluding eye and mouth regions.
- **RGB mean**: pixels inside the mask, with the brightest/darkest tails trimmed to reduce
  specular highlights and shadows.
- **Pose**: yaw proxy from the nose relative to the face edges; roll from the eye-center
  line. If the pose is outside the envelope, the pipeline reports `adjust_pose` and does not
  add samples.

The ROI geometry is intentionally conservative. A later phase can fuse per-ROI pulse
waveforms by SNR or use a skin mask for more aggressive inclusion.

## Signal chain

1. **Buffer** — 10 s of live frames for BPM/waveform; 60 s for HRV. This is a rolling
   window, not a recording: every displayed frame is current, and the window is only used
   to estimate frequency. Only samples with quality above the
   threshold are used. The most recent contiguous segment is selected; a gap larger than
   0.5 s resets the effective window.
2. **Resample** — `scipy.interpolate.interp1d(kind="linear")` onto a uniform grid at the
   measured median frame rate (10–120 Hz clamp). Duplicate/non-monotonic timestamps are
   removed.
3. **POS** — sliding 1.6 s windows (Wang et al. 2017): each window is divided by its
   per-channel mean, projected with `P = [[0, 1, -1], [-2, 1, 1]]`, combined as
   `h = S1 + alpha*S2` with `alpha = std(S1)/std(S2)`, and overlap-added. Impulsive artifacts are
   clipped at 3.5 robust SD. POS runs on the combined trace and on every ROI; the band-passed
   pulses are fused with weights equal to each channel's linear spectral SNR.
4. **Filter** — 3rd-order Butterworth band-pass 0.75–2.8 Hz (45–168 BPM) with
   `sosfiltfilt` for numerical stability and zero phase.
5. **BPM** — Welch PSD (Hann, nperseg 256, nfft 4096, 50% overlap). The peak is refined
   with a parabolic fit on the log PSD; BPM is the refined peak × 60. Once three readings are
   accepted, a local maximum within 0.13 Hz of the recent median is preferred over a stronger
   transient peak if it holds at least half its power; after 30 consecutive overrides the history
   resets so genuine rate changes are followed. The final display BPM
   is a median over recent accepted estimates.
6. **SNR** — 10·log10(power in ±0.1 Hz around the peak / remaining 0.75–2.8 Hz power).
   A peak at the band edge is treated as unreliable. Thresholds are calibrated on data; the
   current placeholder is min 0 dB / good 3 dB.
7. **Peaks / HRV** — `find_peaks` with an IBI-distance constraint and adaptive prominence.
   Both polarities are evaluated; the candidate with more regular IBIs is selected. Peak
   times are refined parabolically. IBIs outside 0.33–1.5 s or >20% from the local median
   are rejected; RMSSD and pNN50 are computed only across adjacent valid intervals. HRV requires
   at least 8 valid IBIs, a 30 s clean window, and a beat rate within 8 BPM of the spectral rate.
8. **Quality gate** — face size, ROI pixel area, pose, brightness, spectral SNR, and window
   duration produce a confidence score and a status. BPM is hidden outside
   `measuring`/`good`; HRV is hidden until ready. This is deliberate: the system should
   refuse to guess.

## Additional live panels

- **PSD spectrum** — the Welch PSD from the 10 s BPM window is sent to the UI, plotted in
  dB from 0.5–3.5 Hz, with the current peak marked.
- **Per-ROI quality** — POS + Butterworth + Welch are run per ROI at 1 Hz, producing BPM,
  SNR, and pixel count for forehead, right cheek, and left cheek. The combined BPM still
  uses the weighted RGB trace; the per-ROI view shows which region is carrying the signal.
- **BPM trend** — the client keeps a 1 Hz, 60 s history of confirmed BPM estimates only;
  provisional estimates are excluded from the trend.
- **IBI tachogram and HRV metrics** — the 60 s HRV window produces RMSSD, SDNN, pNN50, and
  mean IBI from clean IBIs. IBI times and intervals are streamed to the UI for the
  tachogram. HRV computation is throttled to 1 Hz.
- **Lighting/exposure** — mean and standard deviation of ROI luma over the 10 s window.
  High drift indicates auto-exposure/illumination movement that can dominate the pulse.
- **Signal-quality timeline** — the client stores one status per second for 60 s, renders a
  colored strip, and plots the same history as an SNR trend with 0 dB / 3 dB thresholds.
- **Respiration (beta)** — the green channel over a 45 s live window is band-passed at
  0.1–0.5 Hz and analyzed with Welch; the result is breaths/min plus a normalized
  respiration waveform. This is an estimate from PPG modulation, not a respiratory
  medical measurement. It is computed at 1 Hz.

The BPM/waveform path runs at 15 Hz; per-ROI, HRV, and respiration run at 1 Hz and are
cached between analyses. The preview encoder runs independently at 30 fps.

## UI structure

The React app is split into three views, selected in the top toolbar:

- **Monitor** — video, face vitals, waveform, respiration, PSD/per-ROI/HRV, trends,
  lighting, signal diagnostics.
- **Wellness** — guided check-in, breathing pacer, recovery index, personalization summary,
  and time-of-day baseline schedule.
- **Research** — tongue/palm experimental ROI, eye activity, head stability, PSD, per-ROI,
  HRV, and lighting. Switching away from Research resets the ROI mode to Face.

The grid uses auto-placement; each view renders only its own cards, so the page stays
focused instead of one very long dashboard.

## Personalization

Profile data is stored locally in `clinura.profile.v2` (with a migration from the earlier
`clinura.baseline.v1`):

- Name and optional skin-tone context.
- Time-of-day baselines: morning, afternoon, evening. A check-in updates only the current
  period. The recovery index and lighting reference use the matching period.
- Lighting reference: face luma and exposure drift captured during the check-in; the
  Lighting card compares the live values and adjusts the coaching target for darker skin
  tones.
- A `showProvisional` preference controls whether low-confidence gray BPM estimates are
  shown in the main Heart rate card.
- Skin tone is used for coaching/context only. It does **not** alter the POS/Welch signal
  chain or the accuracy gate.

## Wellness and research features

### Guided check-in

The UI runs a 60 s check-in without changing the engine:

- 10 s settle-in, then 50 s of averaged live metrics.
- Confirmed BPM, RMSSD, SDNN, pNN50, respiration, confidence, coverage, lighting, SNR, and
  face quality are summarized.
- The baseline is stored in the local profile under the current time period; no video or
  session is stored by the check-in.
- The recovery index compares current HRV/HR/respiration to that personal baseline; it is
  a wellness heuristic, not a medical score.

### Breathing pacer

A browser-side 6 breaths/min (10 s cycle) pacer with a solid expanding/contracting orb.
The live RMSSD is shown; no engine-side coherence metric is claimed.

### Experimental ROI modes

The **Face / Tongue / Palm** switch sends `{"type":"set_mode","mode":"..."}` over the
existing WebSocket. The face pipeline continues to run; a second `SignalPipeline` analyzes
the selected experimental ROI:

- **Tongue** — the inner-mouth landmarks plus a lower mouth ellipse define a candidate
  region. YCrCb color thresholding (red/pink) selects the tongue, including a protruding
  tongue; mouth-open ratio, area, and brightness produce a quality score.
- **Palm** — MediaPipe `HandLandmarker` finds the palm; a convex hull of the wrist and MCP
  landmarks defines the ROI. Handedness and landmark confidence are reported.
- The experimental pipeline uses the same POS/Butterworth/Welch chain over a 10 s window
  and is throttled to 5 Hz. It is **not validated** and is labeled experimental in the UI.
- The face vitals above remain face-based; a research banner makes this explicit.

### Behavioral panels

- **Eye activity** — FaceLandmarker blendshapes `eyeBlinkLeft`/`eyeBlinkRight` drive blink
  detection, a 60 s blink-rate estimate, and an openness value.
- **Head stability** — a 2 s window of face bbox motion plus yaw/roll produces a 0–100
  stability score. It is a signal-quality diagnostic, not a clinical measure.

## Buffers and latency

- Analysis runs at most every 0.1 s on the rolling window; BPM is median-smoothed over
  recent accepted estimates.
- BPM display is a median of recent accepted estimates, so it is stable but not instant.
- HRV needs 30–60 s of clean data. This is a physiological requirement, not a UI delay or
  a recording.
- Preview frames are 900 px wide at 30 fps; metrics update at 15 Hz. Analysis uses the
  full capture resolution.
- The waveform view can show the last 1, 5, or 10 s. This is a display choice only; BPM
  uses the 10 s rolling analysis window because a 1 s window cannot resolve a heart rate
  reliably.

## Performance notes

- FaceLandmarker at 1080p runs ~25–30 ms/frame on the tested Apple Silicon machine; it is
  fast enough for 30 fps without recording.
- Full-resolution FFV1 recording is CPU-intensive. Recording at 720p or using a faster
  lossless workflow is recommended for long sessions. The live runtime does not record by
  default.
- MediaPipe may use CPU/XNNPACK depending on the build. The GPU delegate is a later
  optimization.

## Camera requirements for long distance

- External 4K, large-sensor camera; 1080p60 for HRV, 4K30 for distance.
- **Manual exposure, white balance, and focus** are essential. Auto-exposure changes gain
  and color balance at frequencies that overlap the pulse band.
- Soft, diffuse, flicker-free light. Avoid backlight and direct sun.
- Stable head pose for the 15–60 s window.

## Runtime supervision

`Runtime` starts a supervisor thread that opens the camera, processes frames and reopens the
camera after failures or stalls (`CaptureConfig.stall_timeout_s`, `ServerConfig.camera_retry_s`).
Mode switches and resets arrive over the WebSocket and are queued; the processing thread applies
them between frames. Each WebSocket session runs its send/receive loops in one anyio task group.

## Telemetry, events and the assistant

`TelemetryStore` writes one sample per second to SQLite (`data/telemetry.db`). `EventDetector`
runs episode rules (tachycardia, bradycardia, spikes over a 10-minute baseline, HRV dips,
respiration out of range, movement, out of view, eyes-closed rest) with minimum durations and
release delays, and annotates episodes that overlap movement. `build_context` produces a compact
JSON summary (statistics, binned series, events, staff notes) that `Assistant` sends to DeepSeek
(`deepseek-flash` by default) for the handover note and copilot answers. Without an API key,
`local_handover` renders the same structure from a template.

## Extension points

- `ROI`/`FaceObservation` types: add skin masks or per-ROI SNR.
- `SignalPipeline._analyze_window`: swap POS for CHROM or a fused method, and add a
  harmonic-aware BPM estimator.
- `Runtime.metrics_payload`: add raw PSD, per-ROI waveforms, or camera-control state.
- `SessionRecorder`: add reference streams, camera metadata, and frame-level quality.
- `docs/validation.md`: run the protocol and replace placeholder thresholds with measured
  operating-envelope numbers.
