# Validation plan

## Claim to validate

> “97% of heart-rate estimates are within ±5 BPM of a reference measurement inside a
> defined operating envelope.”

Always report:

- **MAE** and **RMSE** (BPM)
- **Bland–Altman** bias and 95% limits of agreement
- **% within ±3 / ±5 BPM**
- **Coverage** — fraction of time the system displays an HR estimate
- Per-condition breakdown: distance, lighting, skin tone, motion, camera

High accuracy with low coverage is not a usable product. The UI intentionally hides BPM
when SNR/quality is inadequate.

## Reference devices

| Device | HR | HRV (RR) | Notes |
| --- | --- | --- | --- |
| Polar H10 chest strap | Yes | Yes | Best affordable reference for RR intervals |
| ECG | Yes | Yes | Best overall; needed for clinical-grade HRV |
| Fingertip pulse oximeter | Yes | No | Good for spot HR only |
| Apple Watch | Yes (averaged) | No | Useful for spot HR; not valid for RMSSD |
| Public datasets (UBFC-rPPG, PURE, MMPD) | Yes | Varies | Algorithm sanity, not your camera/room |

Record reference readings with wall-clock timestamps:

```bash
python scripts/reference.py data/sessions/session-... --bpm 68 --source apple_watch
python -m clinura.cli replay data/sessions/session-... --reference data/sessions/session-.../reference.jsonl
```

The replay writes `analysis.jsonl` with `wall_time` mapped from the recorded frames, so
nearest-timestamp matching is meaningful.

## Protocol

1. **Subjects** — 20+ adults, balanced skin tones (Fitzpatrick I–VI), varied face shapes,
   glasses/facial hair/makeup represented.
2. **Distances** — 0.5 m, 1 m, 2 m, 3 m for v1; add 4–5 m only with telephoto/PTZ optics.
3. **Lighting** — diffuse office light, window light, low light, mixed color temperature.
   Record lux and whether exposure/WB were locked.
4. **Motion** — still, talking, slight head motion, typing/shoulder motion.
5. **Duration** — 60 s per condition; 5 min for HRV if using a chest strap/ECG.
6. **Reference sync** — log at least 10 paired readings per condition; use the nearest
   timestamp within a fixed tolerance (e.g., 5 s) for spot HR.
7. **Camera** — record model, resolution, FPS, exposure/WB settings, focus. Built-in
   webcams with auto-exposure are a separate, weaker condition.

## Synthetic checks

`make benchmark` runs clean synthetic signals across 55–85 BPM and reports MAE and
coverage. These tests validate the DSP chain; they do **not** predict human accuracy. The
unit test suite also covers frame jitter, band-pass behavior, POS, spectral estimation,
peak detection, RMSSD, ROI masks, and the full pipeline.

## Known current observations

- With the built-in macOS camera and this room’s lighting, spectral SNR was negative
  (−3 to −6 dB) despite a detected face and ~13k ROI pixels. The UI correctly reported
  **Low signal** and withheld BPM.
- OpenCV’s AVFoundation backend did not expose `CAP_PROP_EXPOSURE`/`CAP_PROP_AUTO_EXPOSURE`
  (returned `-1`) on the built-in camera, so auto-exposure could not be locked. This is a
  likely contributor.
- Full-resolution FFV1 recording reduced the effective sample rate to ~15 fps on the test
  machine; the live path without recording runs faster. For validation, record at 720p or
  use a hardware encoder.
- HRV from 30 FPS is timing-limited; use 60 FPS and 30–60 s windows.

## Panel caveats

- **Per-ROI quality** is diagnostic; it is not yet a validated fusion/selection method.
- **HRV metrics** (SDNN, pNN50, mean IBI) require 60 s of clean live data and are sensitive
  to peak-detection errors. RMSSD remains the primary beta HRV number.
- **Respiration** is derived from low-frequency PPG modulation and is beta-only; it needs
  dedicated validation against a respiratory belt or capnography.
- **Lighting drift** is a camera/exposure diagnostic, not a clinical measurement.
- **Signal-quality timeline/coverage** is client-side history and resets on page reload.
- The BPM trend shows confirmed estimates only; provisional estimates are intentionally
  excluded.

## Experimental-feature caveats

- **Tongue ROI** is a color-segmentation heuristic. Tongue color, saliva, teeth, shadows,
  and motion can break the mask. It is not a validated PPG site.
- **Palm ROI** uses hand landmarks and a convex hull; hand orientation, lighting, and
  motion change the signal. Left/right handedness labels may be affected by mirroring.
- **Eye activity** is a behavioral estimate from blendshapes; blink rate/openness are not
  diagnostic measures.
- **Head stability** is a quality diagnostic derived from bbox motion and pose.
- **Guided check-in, time-of-day baselines, and recovery index** use a local baseline and
  population-style normalization; they are wellness features, not clinical assessments.
- **Personalization** (name, skin-tone context, lighting reference, provisional-BPM
  preference) changes coaching/display behavior only. Skin tone must not be used to claim
  different pulse accuracy without validation.
- **Time-of-day baselines** are separate; a recovery score is only produced when the
  current period has a baseline. Baselines are not a substitute for a longitudinal study.
- Do not publish tongue/palm accuracy, blink-based diagnoses, or recovery-score claims
  without dedicated validation against appropriate reference devices.

## Long-distance physics

Field of view and face size dominate. For a 1080p camera with ~65° horizontal FOV:

| Distance | Face width (approx.) | Forehead ROI (approx.) |
| --- | --- | --- |
| 1 m | 227 px | ~90 × 20 px |
| 2 m | 113 px | ~45 × 10 px |
| 3 m | 76 px | ~30 × 7 px |
| 5 m | 45 px | ~18 × 4 px |

4K doubles those pixel counts. A 2× optical zoom doubles them again. Digital zoom does not
add information. At 5 m a normal webcam does not have enough ROI pixels for reliable rPPG;
use a large-sensor 4K camera with a narrow FOV or telephoto/PTZ optics.

## Operating envelope (to be filled with data)

| Condition | HR | HRV |
| --- | --- | --- |
| 0.5–2 m, 1080p, diffuse light, still | Target: ≥97% within ±5 BPM | 60 fps, 30–60 s; validate |
| 2–3 m, 4K, locked exposure | Target: validate | Marginal; validate |
| 3–5 m, 4K + telephoto | Research | Not reliable at 30 fps |
| 5 m+, normal webcam | Not supported | Not supported |

Do not publish the 97% claim until the protocol above has been run and the per-condition
coverage and agreement metrics are available.

## Clinical path

Clinical/medical claims require a separate program: intended-use definition, risk analysis,
software lifecycle (e.g., IEC 62304), quality system, clinical study, and regulatory
clearance. The current code is a research prototype and must not be presented as a medical
device.
