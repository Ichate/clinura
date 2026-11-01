from __future__ import annotations

import math
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .securestore import EncryptedDatabase

SAMPLE_FIELDS = (
    "ts",
    "bpm",
    "provisional_bpm",
    "rmssd_ms",
    "sdnn_ms",
    "respiration_bpm",
    "snr_db",
    "confidence",
    "status",
    "stability",
    "motion_px_s",
    "face_detected",
    "eyes_closed",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    ts REAL PRIMARY KEY,
    bpm REAL,
    provisional_bpm REAL,
    rmssd_ms REAL,
    sdnn_ms REAL,
    respiration_bpm REAL,
    snr_db REAL,
    confidence REAL,
    status TEXT,
    stability REAL,
    motion_px_s REAL,
    face_detected INTEGER,
    eyes_closed REAL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    end_ts REAL,
    kind TEXT NOT NULL,
    severity TEXT NOT NULL,
    title TEXT NOT NULL,
    detail TEXT,
    value REAL,
    source TEXT NOT NULL DEFAULT 'detector'
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
"""


def clock(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def stamp(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def humanize(seconds: float) -> str:
    seconds = max(1, round(seconds))
    if seconds < 90:
        return f"{seconds} s"
    minutes = round(seconds / 60)
    if minutes < 90:
        return f"{minutes} min"
    hours, rest = divmod(minutes, 60)
    return f"{hours} h {rest} min" if rest else f"{hours} h"


def _finite(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class Event:
    id: int
    ts: float
    end_ts: float | None
    kind: str
    severity: str
    title: str
    detail: str | None
    value: float | None
    source: str

    def to_dict(self) -> dict[str, object]:
        duration = None if self.end_ts is None else round(self.end_ts - self.ts)
        return {
            "id": self.id,
            "ts": self.ts,
            "time": stamp(self.ts),
            "clock": clock(self.ts),
            "end_ts": self.end_ts,
            "end_clock": clock(self.end_ts),
            "duration_s": duration,
            "ongoing": self.end_ts is None and self.source == "detector",
            "kind": self.kind,
            "severity": self.severity,
            "title": self.title,
            "detail": self.detail,
            "value": self.value,
            "source": self.source,
        }


class TelemetryStore:
    def __init__(self, path: str | Path | None = None, retention_days: float = 7.0) -> None:
        self._lock = threading.Lock()
        self._secure: EncryptedDatabase | None = None
        if path is None:
            self._db = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        else:
            self._secure = EncryptedDatabase(Path(path), self._lock)
            self._db = self._secure.conn
        self._db.executescript(SCHEMA)
        self._pending: list[tuple[object, ...]] = []
        self._last_flush = time.monotonic()
        cutoff = time.time() - retention_days * 86400.0
        with self._lock:
            self._db.execute("DELETE FROM samples WHERE ts < ?", (cutoff,))
            self._db.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
            self._db.execute(
                "UPDATE events SET end_ts = ts WHERE end_ts IS NULL AND source = 'detector'"
            )

    def record(self, sample: dict[str, object]) -> None:
        row = tuple(sample.get(name) for name in SAMPLE_FIELDS)
        with self._lock:
            self._pending.append(row)
            if len(self._pending) >= 10 or time.monotonic() - self._last_flush > 5.0:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._pending:
            return
        placeholders = ",".join("?" for _ in SAMPLE_FIELDS)
        self._db.executemany(
            f"INSERT OR REPLACE INTO samples ({','.join(SAMPLE_FIELDS)}) VALUES ({placeholders})",
            self._pending,
        )
        self._pending.clear()
        self._last_flush = time.monotonic()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()
            if self._secure is not None:
                self._secure.save()

    def add_event(
        self,
        kind: str,
        severity: str,
        title: str,
        detail: str | None = None,
        value: float | None = None,
        ts: float | None = None,
        end_ts: float | None = None,
        source: str = "detector",
    ) -> int:
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO events (ts, end_ts, kind, severity, title, detail, value, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ts or time.time(), end_ts, kind, severity, title, detail, value, source),
            )
            return int(cursor.lastrowid or 0)

    def close_event(self, event_id: int, end_ts: float, detail: str, value: float | None) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE events SET end_ts = ?, detail = ?, value = ? WHERE id = ?",
                (end_ts, detail, value, event_id),
            )

    def samples(self, since: float, until: float | None = None) -> list[dict[str, object]]:
        with self._lock:
            self._flush_locked()
            rows = self._db.execute(
                f"SELECT {','.join(SAMPLE_FIELDS)} FROM samples WHERE ts >= ? AND ts <= ? "  # nosec B608 - column names from fixed whitelist; values are parameters
                "ORDER BY ts",
                (since, until or time.time() + 1.0),
            ).fetchall()
        return [dict(zip(SAMPLE_FIELDS, row, strict=True)) for row in rows]

    def events(self, since: float, limit: int = 500) -> list[Event]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, ts, end_ts, kind, severity, title, detail, value, source FROM events "
                "WHERE ts >= ? OR end_ts IS NULL ORDER BY ts DESC LIMIT ?",
                (since, limit),
            ).fetchall()
        return [Event(*row) for row in reversed(rows)]

    def close(self) -> None:
        with self._lock:
            self._flush_locked()
            if self._secure is not None:
                self._secure.close()
            else:
                self._db.close()

    def wipe(self, key: bytes | None = None) -> None:
        with self._lock:
            self._pending.clear()
            self._db.execute("DELETE FROM samples")
            self._db.execute("DELETE FROM events")
            self._db.execute("VACUUM")
            if self._secure is not None:
                if key is not None:
                    self._secure.rekey(key)
                else:
                    self._secure.save(force=True)


@dataclass
class EpisodeRule:
    kind: str
    severity: str
    title: str
    unit: str
    active: Callable[[dict[str, object], dict[str, float | None]], bool]
    value: Callable[[dict[str, object]], float | None]
    extreme: str
    min_duration_s: float
    release_s: float = 8.0


@dataclass
class _Episode:
    rule: EpisodeRule
    started: float
    last_true: float
    event_id: int | None = None
    values: list[float] = field(default_factory=list)
    motion_seconds: int = 0
    seconds: int = 0


class EventDetector:
    def __init__(self, store: TelemetryStore) -> None:
        self.store = store
        self._recent_bpm: deque[tuple[float, float]] = deque()
        self._recent_rmssd: deque[tuple[float, float]] = deque()
        self._episodes: dict[str, _Episode] = {}
        self.rules = [
            EpisodeRule(
                "tachycardia",
                "warning",
                "Heart rate above 100 BPM",
                "BPM",
                lambda s, ctx: _gt(s.get("bpm"), 100.0),
                lambda s: _finite(s.get("bpm")),
                "max",
                10.0,
            ),
            EpisodeRule(
                "hr_spike",
                "warning",
                "Transient heart-rate spike",
                "BPM",
                lambda s, ctx: (
                    ctx["bpm_baseline"] is not None
                    and _gt(s.get("bpm"), ctx["bpm_baseline"] + 20.0)
                    and not _gt(s.get("bpm"), 100.0)
                ),
                lambda s: _finite(s.get("bpm")),
                "max",
                5.0,
            ),
            EpisodeRule(
                "bradycardia",
                "warning",
                "Heart rate below 50 BPM",
                "BPM",
                lambda s, ctx: _lt(s.get("bpm"), 50.0),
                lambda s: _finite(s.get("bpm")),
                "min",
                10.0,
            ),
            EpisodeRule(
                "hrv_dip",
                "info",
                "HRV dip",
                "ms",
                lambda s, ctx: (
                    ctx["rmssd_baseline"] is not None
                    and _lt(s.get("rmssd_ms"), 0.6 * ctx["rmssd_baseline"])
                ),
                lambda s: _finite(s.get("rmssd_ms")),
                "min",
                30.0,
            ),
            EpisodeRule(
                "tachypnea",
                "warning",
                "Respiration above 24 breaths/min",
                "breaths/min",
                lambda s, ctx: _gt(s.get("respiration_bpm"), 24.0),
                lambda s: _finite(s.get("respiration_bpm")),
                "max",
                30.0,
            ),
            EpisodeRule(
                "bradypnea",
                "warning",
                "Respiration below 8 breaths/min",
                "breaths/min",
                lambda s, ctx: _lt(s.get("respiration_bpm"), 8.0),
                lambda s: _finite(s.get("respiration_bpm")),
                "min",
                30.0,
            ),
            EpisodeRule(
                "movement",
                "info",
                "Movement",
                "stability",
                lambda s, ctx: bool(s.get("face_detected")) and _lt(s.get("stability"), 40.0),
                lambda s: _finite(s.get("stability")),
                "min",
                5.0,
                release_s=5.0,
            ),
            EpisodeRule(
                "out_of_view",
                "info",
                "Patient out of camera view",
                "",
                lambda s, ctx: s.get("status") in ("no_face", "error"),
                lambda s: None,
                "max",
                30.0,
                release_s=5.0,
            ),
            EpisodeRule(
                "eyes_closed",
                "info",
                "Eyes-closed rest period",
                "",
                lambda s, ctx: bool(s.get("face_detected")) and _gt(s.get("eyes_closed"), 0.7),
                lambda s: None,
                "max",
                120.0,
                release_s=20.0,
            ),
        ]

    def _baseline(self, history: deque[tuple[float, float]], now: float) -> float | None:
        while history and history[0][0] < now - 600.0:
            history.popleft()
        if len(history) < 60:
            return None
        return float(np.median([value for _, value in history]))

    def update(self, sample: dict[str, object]) -> None:
        now = float(sample["ts"])
        context = {
            "bpm_baseline": self._baseline(self._recent_bpm, now),
            "rmssd_baseline": self._baseline(self._recent_rmssd, now),
        }
        bpm = _finite(sample.get("bpm"))
        rmssd = _finite(sample.get("rmssd_ms"))
        moving = _lt(sample.get("stability"), 50.0)
        for rule in self.rules:
            episode = self._episodes.get(rule.kind)
            if rule.active(sample, context):
                if episode is None:
                    episode = _Episode(rule=rule, started=now, last_true=now)
                    self._episodes[rule.kind] = episode
                episode.last_true = now
                episode.seconds += 1
                episode.motion_seconds += int(moving)
                value = rule.value(sample)
                if value is not None:
                    episode.values.append(value)
                if episode.event_id is None and now - episode.started >= rule.min_duration_s:
                    episode.event_id = self.store.add_event(
                        rule.kind,
                        rule.severity,
                        rule.title,
                        detail="In progress",
                        value=self._extreme(episode),
                        ts=episode.started,
                    )
            elif episode is not None and now - episode.last_true >= rule.release_s:
                if episode.event_id is not None:
                    self.store.close_event(
                        episode.event_id,
                        episode.last_true,
                        self._describe(episode, context),
                        self._extreme(episode),
                    )
                del self._episodes[rule.kind]

        if bpm is not None and not moving:
            self._recent_bpm.append((now, bpm))
        if rmssd is not None:
            self._recent_rmssd.append((now, rmssd))

    def _extreme(self, episode: _Episode) -> float | None:
        if not episode.values:
            return None
        values = episode.values
        return round(max(values) if episode.rule.extreme == "max" else min(values), 1)

    def _describe(self, episode: _Episode, context: dict[str, float | None]) -> str:
        parts = [f"Lasted {humanize(episode.last_true - episode.started)}"]
        extreme = self._extreme(episode)
        if extreme is not None and episode.rule.unit:
            label = "peak" if episode.rule.extreme == "max" else "lowest"
            parts.append(f"{label} {extreme:g} {episode.rule.unit}")
        if episode.rule.kind in ("hr_spike", "tachycardia") and context["bpm_baseline"]:
            parts.append(f"baseline {context['bpm_baseline']:.0f} BPM")
        if episode.rule.kind not in ("movement", "out_of_view") and episode.seconds:
            if episode.motion_seconds / episode.seconds >= 0.3:
                parts.append("coincided with movement, likely artifact")
        return "; ".join(parts)

    def close_all(self) -> None:
        for episode in self._episodes.values():
            if episode.event_id is not None:
                self.store.close_event(
                    episode.event_id,
                    episode.last_true,
                    self._describe(episode, {"bpm_baseline": None, "rmssd_baseline": None}),
                    self._extreme(episode),
                )
        self._episodes.clear()


def _gt(value: object, threshold: float) -> bool:
    number = _finite(value)
    return number is not None and number > threshold


def _lt(value: object, threshold: float) -> bool:
    number = _finite(value)
    return number is not None and number < threshold


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 1),
        "sd": round(float(array.std()), 1),
        "min": round(float(array.min()), 1),
        "max": round(float(array.max()), 1),
        "median": round(float(np.median(array)), 1),
    }


def _bin_seconds(hours: float) -> int:
    if hours <= 1.0:
        return 60
    if hours <= 6.0:
        return 300
    if hours <= 24.0:
        return 900
    return 1800


def build_context(
    store: TelemetryStore, hours: float, now: float | None = None
) -> dict[str, object]:
    end = now or time.time()
    start = end - hours * 3600.0
    samples = store.samples(start, end)
    events = store.events(start)

    def column(name: str) -> list[float]:
        return [v for s in samples if (v := _finite(s.get(name))) is not None]

    confirmed = [s for s in samples if _finite(s.get("bpm")) is not None]
    peak = max(confirmed, key=lambda s: float(s["bpm"]), default=None)
    trough = min(confirmed, key=lambda s: float(s["bpm"]), default=None)

    step = _bin_seconds(hours)
    bins: dict[int, list[dict[str, object]]] = {}
    for sample in samples:
        bins.setdefault(int(float(sample["ts"]) // step), []).append(sample)
    series = []
    for key in sorted(bins):
        rows = bins[key]
        bpm = [v for r in rows if (v := _finite(r.get("bpm"))) is not None]
        rmssd = [v for r in rows if (v := _finite(r.get("rmssd_ms"))) is not None]
        resp = [v for r in rows if (v := _finite(r.get("respiration_bpm"))) is not None]
        stability = [v for r in rows if (v := _finite(r.get("stability"))) is not None]
        eyes = [v for r in rows if (v := _finite(r.get("eyes_closed"))) is not None]
        series.append(
            {
                "from": datetime.fromtimestamp(key * step).strftime("%Y-%m-%d %H:%M"),
                "hr_mean": round(float(np.mean(bpm)), 1) if bpm else None,
                "hr_max": round(float(np.max(bpm)), 1) if bpm else None,
                "rmssd_mean": round(float(np.mean(rmssd)), 1) if rmssd else None,
                "resp_mean": round(float(np.mean(resp)), 1) if resp else None,
                "stability_mean": round(float(np.mean(stability)), 0) if stability else None,
                "eyes_closed_pct": round(float(np.mean([e > 0.7 for e in eyes])) * 100)
                if eyes
                else None,
                "hr_coverage_pct": round(len(bpm) / max(len(rows), 1) * 100),
            }
        )

    present = sum(1 for s in samples if s.get("face_detected"))
    return {
        "window": {
            "start": stamp(start),
            "end": stamp(end),
            "hours": round(hours, 2),
            "first_sample": stamp(_finite(samples[0]["ts"])) if samples else None,
            "seconds_recorded": len(samples),
        },
        "coverage": {
            "patient_in_view_pct": round(present / len(samples) * 100, 1) if samples else 0.0,
            "confirmed_hr_pct": round(len(confirmed) / len(samples) * 100, 1) if samples else 0.0,
        },
        "heart_rate_bpm": _stats(column("bpm")),
        "heart_rate_peak": None
        if peak is None
        else {"bpm": peak["bpm"], "time": clock(_finite(peak["ts"]))},
        "heart_rate_lowest": None
        if trough is None
        else {"bpm": trough["bpm"], "time": clock(_finite(trough["ts"]))},
        "rmssd_ms": _stats(column("rmssd_ms")),
        "sdnn_ms": _stats(column("sdnn_ms")),
        "respiration_bpm": _stats(column("respiration_bpm")),
        "signal_snr_db": _stats(column("snr_db")),
        "head_stability": _stats(column("stability")),
        "series_bin_seconds": step,
        "series": series,
        "events": [
            {
                key: value
                for key, value in event.to_dict().items()
                if key
                in (
                    "time",
                    "clock",
                    "end_clock",
                    "duration_s",
                    "kind",
                    "severity",
                    "title",
                    "detail",
                    "value",
                    "source",
                    "ongoing",
                )
            }
            for event in events
        ],
    }
