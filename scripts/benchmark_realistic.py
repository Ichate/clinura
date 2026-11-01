"""Streaming accuracy + stability benchmark on realistic synthetic scenarios.

Frames are fed one by one exactly like the live runtime does, and the value the UI
would show is sampled once per second. Besides accuracy (MAE, % within ±5 BPM) it
reports how *stable* the display is: second-to-second jitter, big jumps and how often
the reading flickers between shown and hidden.

    PYTHONPATH=engine python scripts/benchmark_realistic.py [--seeds 3] [--only name]

These are synthetic signals. They are good at catching regressions in tracking and
stability; they do not replace validation against a chest strap or ECG.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time

import numpy as np

from clinura.config import SignalConfig
from clinura.pipeline import SignalPipeline
from clinura.synthetic import ROI_NAMES, SCENARIOS, ScenarioData, generate_scenario

WARMUP_S = 12.0


def _true_rmssd(data: ScenarioData, t: float, span: float = 60.0) -> float | None:
    beats = data.beat_times[(data.beat_times > t - span) & (data.beat_times <= t)]
    if len(beats) < 10:
        return None
    ibis = np.diff(beats)
    return float(np.sqrt(np.mean(np.square(np.diff(ibis)))) * 1000.0)


def run(data: ScenarioData, config: SignalConfig | None = None) -> dict[str, float]:
    pipeline = SignalPipeline(config or SignalConfig())
    params = inspect.signature(pipeline.add_sample).parameters
    accepts_head = "head_xy" in params
    shown: list[tuple[float, float | None, float | None, float | None, float | None]] = []
    next_tick = 1.0
    started = time.perf_counter()
    for i, ts in enumerate(data.timestamps):
        ts = float(ts)
        pipeline.set_face_status(True, 0.9, None, ts)
        kwargs = {
            "roi_rgb": {name: tuple(data.roi_rgb[name][i]) for name in ROI_NAMES},
            "roi_pixels_by_name": dict(data.roi_pixels),
        }
        if accepts_head:
            kwargs["head_xy"] = (float(data.head_xy[i, 0]), float(data.head_xy[i, 1]))
        pipeline.add_sample(ts, tuple(data.rgb[i]), 0.9, data.face_width, 10600, **kwargs)
        result = pipeline.analyze(ts)
        if ts >= next_tick:
            next_tick += 1.0
            resp = result.respiration_bpm if result.respiration_ready else None
            rmssd = result.rmssd_ms if result.hrv_ready else None
            shown.append((ts, result.bpm, result.provisional_bpm, resp, rmssd))
    elapsed = time.perf_counter() - started

    rows = [row for row in shown if row[0] >= WARMUP_S]
    errors, deltas, resp_errors, hrv_errors = [], [], [], []
    flips = 0
    previous_bpm: float | None = None
    previous_shown: bool | None = None
    for t, bpm, _prov, resp, rmssd in rows:
        visible = bpm is not None
        if previous_shown is not None and visible != previous_shown:
            flips += 1
        previous_shown = visible
        if bpm is not None:
            reference = data.reference_hr(t)
            if reference is not None:
                errors.append(bpm - reference)
            if previous_bpm is not None:
                deltas.append(abs(bpm - previous_bpm))
            previous_bpm = bpm
        else:
            previous_bpm = None
        if resp is not None:
            resp_errors.append(resp - data.reference_resp(t))
        if rmssd is not None:
            truth = _true_rmssd(data, t)
            if truth is not None:
                hrv_errors.append(rmssd - truth)

    minutes = max(len(rows) / 60.0, 1e-6)
    err = np.abs(np.asarray(errors)) if errors else np.asarray([np.nan])
    d = np.asarray(deltas) if deltas else np.asarray([np.nan])
    return {
        "coverage": 100.0 * sum(r[1] is not None for r in rows) / max(len(rows), 1),
        "mae": float(np.nanmean(err)),
        "within5": float(100.0 * np.mean(err <= 5.0)) if errors else float("nan"),
        "jitter": float(np.nanmean(d)),
        "jumps_per_min": float(np.sum(d > 5.0) / minutes) if deltas else 0.0,
        "flips_per_min": flips / minutes,
        "resp_cov": 100.0 * len(resp_errors) / max(len(rows), 1),
        "resp_mae": float(np.mean(np.abs(resp_errors))) if resp_errors else float("nan"),
        "hrv_cov": 100.0 * len(hrv_errors) / max(len(rows), 1),
        "hrv_mae": float(np.mean(np.abs(hrv_errors))) if hrv_errors else float("nan"),
        "ms_per_frame": 1000.0 * elapsed / max(len(data.timestamps), 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--only", default=None, help="run only scenarios containing this text")
    parser.add_argument("--json", default=None, help="write the summary to this file")
    args = parser.parse_args(argv)

    keys = [
        "coverage",
        "mae",
        "within5",
        "jitter",
        "jumps_per_min",
        "flips_per_min",
        "resp_cov",
        "resp_mae",
        "hrv_cov",
        "hrv_mae",
    ]
    header = f"{'scenario':<20}" + "".join(f"{k:>10}" for k in keys)
    print(header)
    summary: dict[str, dict[str, float]] = {}
    for scenario in SCENARIOS:
        if args.only and args.only not in scenario.name:
            continue
        runs = [run(generate_scenario(scenario, seed=1000 + s)) for s in range(args.seeds)]
        averaged = {k: float(np.nanmean([r[k] for r in runs])) for k in runs[0]}
        summary[scenario.name] = averaged
        print(f"{scenario.name:<20}" + "".join(f"{averaged[k]:>10.2f}" for k in keys))
        sys.stdout.flush()
    overall = {k: float(np.nanmean([v[k] for v in summary.values()])) for k in keys}
    print(f"{'MEAN':<20}" + "".join(f"{overall[k]:>10.2f}" for k in keys))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"scenarios": summary, "overall": overall}, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
