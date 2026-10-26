from __future__ import annotations

import argparse
import statistics
import sys

from clinura.config import SignalConfig
from clinura.pipeline import SignalPipeline
from clinura.synthetic import generate_synthetic_signal


def run_case(bpm: float, seed: int, noise: float, duration: float) -> tuple[float, float]:
    synthetic = generate_synthetic_signal(
        duration_s=duration,
        fs=30,
        bpm=bpm,
        rmssd_ms=32,
        noise=noise,
        jitter_s=0.001,
        seed=seed,
    )
    pipeline = SignalPipeline(SignalConfig(hrv_window_s=60, min_hrv_window_s=30))
    for timestamp, rgb in zip(synthetic.timestamps, synthetic.rgb, strict=True):
        pipeline.set_face_status(True, 0.9, None, float(timestamp))
        pipeline.add_sample(float(timestamp), tuple(rgb), 0.9, 180.0, 800)
    result = pipeline.analyze(float(synthetic.timestamps[-1]), force=True)
    if result.bpm is None:
        return float("nan"), float("nan")
    return result.bpm, result.rmssd_ms or float("nan")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic DSP sanity benchmark")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--noise", type=float, default=0.0003)
    parser.add_argument("--duration", type=float, default=35)
    args = parser.parse_args(argv)

    rows = []
    for bpm in (55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0):
        errors = []
        for seed in range(args.seeds):
            estimate, _ = run_case(bpm, 1000 + seed, args.noise, args.duration)
            if estimate == estimate:
                errors.append(estimate - bpm)
        if errors:
            mae = statistics.mean(abs(error) for error in errors)
            within = sum(abs(error) <= 5.0 for error in errors) / len(errors) * 100.0
            rows.append((bpm, mae, within, len(errors)))

    print("Synthetic DSP sanity check (not a substitute for human validation)")
    print(" bpm    MAE    within ±5 bpm   n")
    for bpm, mae, within, count in rows:
        print(f"{bpm:5.1f}  {mae:5.2f}   {within:5.1f}%          {count}")
    if rows:
        overall = statistics.mean(row[1] for row in rows)
        covered = statistics.mean(row[2] for row in rows)
        print(f"\nmean MAE {overall:.2f} bpm, mean coverage within ±5 bpm {covered:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
