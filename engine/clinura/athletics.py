"""Exercise prescriptions, schedule and session history (SQLite).

A physician, physiotherapist or trainer builds a *program* for a patient/athlete and
adds *items*: which exercise, how many sets and reps (or seconds to hold), rest,
on which weekdays and at what times, plus optional limits (target range of motion,
maximum heart rate) and instructions. The patient's "Today" list is generated from
that schedule; every completed session is stored with reps, form, range of motion,
heart rate and reported effort/pain so the prescriber can review adherence.

Editing can be protected with a clinician PIN (salted SHA-256, stored locally).
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from .exercise import LIBRARY
from .securestore import EncryptedDatabase

SCHEMA = """
CREATE TABLE IF NOT EXISTS programs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    patient TEXT,
    prescriber TEXT,
    prescriber_role TEXT,
    goal TEXT,
    notes TEXT,
    start_date TEXT,
    end_date TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    program_id INTEGER NOT NULL REFERENCES programs(id) ON DELETE CASCADE,
    exercise TEXT NOT NULL,
    sets INTEGER NOT NULL,
    reps INTEGER NOT NULL,
    hold_s INTEGER NOT NULL DEFAULT 0,
    rest_s INTEGER NOT NULL DEFAULT 60,
    side TEXT NOT NULL DEFAULT 'auto',
    days TEXT NOT NULL DEFAULT '1111111',
    times TEXT NOT NULL DEFAULT '',
    target_rom REAL,
    max_hr REAL,
    instructions TEXT,
    position INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    program_id INTEGER,
    item_id INTEGER,
    exercise TEXT NOT NULL,
    slot_date TEXT,
    slot_time TEXT,
    started_at REAL NOT NULL,
    ended_at REAL NOT NULL,
    sets_done INTEGER NOT NULL DEFAULT 0,
    reps_done INTEGER NOT NULL DEFAULT 0,
    reps_target INTEGER NOT NULL DEFAULT 0,
    hold_s REAL,
    partials INTEGER NOT NULL DEFAULT 0,
    avg_form REAL,
    avg_rom REAL,
    best_rom REAL,
    hr_pre REAL,
    hr_peak REAL,
    hr_post REAL,
    hr_recovery REAL,
    rpe REAL,
    pain REAL,
    notes TEXT,
    completed INTEGER NOT NULL DEFAULT 1,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS sessions_slot ON sessions(slot_date);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
"""

PROGRAM_FIELDS = (
    "title", "patient", "prescriber", "prescriber_role", "goal", "notes",
    "start_date", "end_date", "active",
)  # fmt: skip
ITEM_FIELDS = (
    "exercise", "sets", "reps", "hold_s", "rest_s", "side", "days", "times",
    "target_rom", "max_hr", "instructions", "position",
)  # fmt: skip
SESSION_FIELDS = (
    "program_id", "item_id", "exercise", "slot_date", "slot_time", "started_at", "ended_at",
    "sets_done", "reps_done", "reps_target", "hold_s", "partials", "avg_form", "avg_rom",
    "best_rom", "hr_pre", "hr_peak", "hr_post", "hr_recovery", "rpe", "pain", "notes",
    "completed", "detail",
)  # fmt: skip


class AthleticsStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._secure: EncryptedDatabase | None = None
        if path is None:
            self._db = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        else:
            self._secure = EncryptedDatabase(Path(path), self._lock)
            self._db = self._secure.conn
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript(SCHEMA)
        self._failed_pins: list[float] = []

    # ------------------------------------------------------------- PIN

    def pin_set(self) -> bool:
        with self._lock:
            row = self._db.execute("SELECT value FROM settings WHERE key='pin'").fetchone()
        return row is not None

    def pin_locked_for(self) -> float:
        """Seconds until PIN attempts are allowed again (5 failures -> 60 s lockout)."""
        now = time.time()
        self._failed_pins = [t for t in self._failed_pins if now - t < 60.0]
        if len(self._failed_pins) >= 5:
            return 60.0 - (now - self._failed_pins[0])
        return 0.0

    def check_pin(self, pin: str | None) -> bool:
        with self._lock:
            row = self._db.execute("SELECT value FROM settings WHERE key='pin'").fetchone()
        if row is None:
            return True
        if self.pin_locked_for() > 0:
            return False
        stored = str(row["value"])
        if stored.startswith("pbkdf2$"):
            _, iterations, salt, digest = stored.split("$")
            candidate = hashlib.pbkdf2_hmac(
                "sha256", (pin or "").encode(), bytes.fromhex(salt), int(iterations)
            ).hex()
        else:  # legacy salted sha256
            salt, digest = stored.split(":", 1)
            candidate = hashlib.sha256((salt + (pin or "")).encode()).hexdigest()
        ok = secrets.compare_digest(candidate, digest)
        if not ok:
            self._failed_pins.append(time.time())
        else:
            self._failed_pins.clear()
            if not stored.startswith("pbkdf2$"):
                self.set_pin(pin)
        return ok

    def set_pin(self, pin: str | None) -> None:
        with self._lock:
            if not pin:
                self._db.execute("DELETE FROM settings WHERE key='pin'")
                return
            salt = secrets.token_bytes(16)
            iterations = 300_000
            digest = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, iterations).hex()
            self._db.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES ('pin', ?)",
                (f"pbkdf2${iterations}${salt.hex()}${digest}",),
            )

    def wipe(self, key: bytes | None = None) -> None:
        with self._lock:
            for table in ("sessions", "items", "programs", "settings"):
                self._db.execute(f"DELETE FROM {table}")  # noqa: S608 - fixed table names  # nosec B608 - column names from fixed whitelist; values are parameters
            self._db.execute("VACUUM")
            if self._secure is not None:
                if key is not None:
                    self._secure.rekey(key)
                else:
                    self._secure.save(force=True)

    # -------------------------------------------------------- programs

    def programs(self) -> list[dict[str, object]]:
        with self._lock:
            programs = [
                dict(r) for r in self._db.execute("SELECT * FROM programs ORDER BY id DESC")
            ]
            items = [dict(r) for r in self._db.execute("SELECT * FROM items ORDER BY position, id")]
        for program in programs:
            program["active"] = bool(program["active"])
            program["items"] = [_item_out(i) for i in items if i["program_id"] == program["id"]]
        return programs

    def program(self, program_id: int) -> dict[str, object] | None:
        return next((p for p in self.programs() if p["id"] == program_id), None)

    def create_program(self, data: dict[str, object]) -> int:
        now = time.time()
        values = {k: data.get(k) for k in PROGRAM_FIELDS}
        values["active"] = 1 if values.get("active", True) is not False else 0
        values["start_date"] = values.get("start_date") or date.today().isoformat()
        with self._lock:
            cursor = self._db.execute(
                f"INSERT INTO programs ({','.join(PROGRAM_FIELDS)}, created_at, updated_at) "  # nosec B608 - column names from fixed whitelist; values are parameters
                f"VALUES ({','.join('?' for _ in PROGRAM_FIELDS)}, ?, ?)",
                (*[values[k] for k in PROGRAM_FIELDS], now, now),
            )
            return int(cursor.lastrowid or 0)

    def update_program(self, program_id: int, data: dict[str, object]) -> bool:
        fields = [k for k in PROGRAM_FIELDS if k in data]
        if not fields:
            return self.program(program_id) is not None
        values = [int(bool(data[k])) if k == "active" else data[k] for k in fields]
        with self._lock:
            cursor = self._db.execute(
                f"UPDATE programs SET {','.join(f'{k}=?' for k in fields)}, updated_at=? WHERE id=?",  # nosec B608 - column names from fixed whitelist; values are parameters
                (*values, time.time(), program_id),
            )
            return cursor.rowcount > 0

    def delete_program(self, program_id: int) -> bool:
        with self._lock:
            return self._db.execute("DELETE FROM programs WHERE id=?", (program_id,)).rowcount > 0

    def add_item(self, program_id: int, data: dict[str, object]) -> int:
        values = _item_in(data)
        with self._lock:
            if (
                self._db.execute("SELECT 1 FROM programs WHERE id=?", (program_id,)).fetchone()
                is None
            ):
                raise KeyError("program not found")
            cursor = self._db.execute(
                f"INSERT INTO items (program_id, {','.join(ITEM_FIELDS)}) "  # nosec B608 - column names from fixed whitelist; values are parameters
                f"VALUES (?, {','.join('?' for _ in ITEM_FIELDS)})",
                (program_id, *[values[k] for k in ITEM_FIELDS]),
            )
            self._db.execute(
                "UPDATE programs SET updated_at=? WHERE id=?", (time.time(), program_id)
            )
            return int(cursor.lastrowid or 0)

    def update_item(self, item_id: int, data: dict[str, object]) -> bool:
        with self._lock:
            row = self._db.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        if row is None:
            return False
        merged = {**_item_out(dict(row)), **data}
        values = _item_in(merged)
        with self._lock:
            self._db.execute(
                f"UPDATE items SET {','.join(f'{k}=?' for k in ITEM_FIELDS)} WHERE id=?",  # nosec B608 - column names from fixed whitelist; values are parameters
                (*[values[k] for k in ITEM_FIELDS], item_id),
            )
        return True

    def delete_item(self, item_id: int) -> bool:
        with self._lock:
            return self._db.execute("DELETE FROM items WHERE id=?", (item_id,)).rowcount > 0

    # -------------------------------------------------------- schedule

    def schedule(self, day: date) -> list[dict[str, object]]:
        weekday = day.weekday()
        day_s = day.isoformat()
        now = datetime.now()
        with self._lock:
            done = [
                dict(r)
                for r in self._db.execute(
                    "SELECT id, item_id, slot_time, completed, reps_done, avg_form FROM sessions "
                    "WHERE slot_date=?",
                    (day_s,),
                )
            ]
        slots = []
        for program in self.programs():
            if not program["active"]:
                continue
            if program.get("start_date") and str(program["start_date"]) > day_s:
                continue
            if program.get("end_date") and str(program["end_date"]) < day_s:
                continue
            for item in program["items"]:
                if item["days"][weekday] != "1":
                    continue
                for slot_time in item["times"] or [""]:
                    session = next(
                        (
                            s
                            for s in done
                            if s["item_id"] == item["id"] and (s["slot_time"] or "") == slot_time
                        ),
                        None,
                    )
                    if session is not None:
                        status = "done" if session["completed"] else "partial"
                    elif day < now.date():
                        status = "missed"
                    elif day > now.date() or not slot_time:
                        status = "due" if day == now.date() else "upcoming"
                    else:
                        hour, minute = (int(x) for x in slot_time.split(":"))
                        due_at = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                        if now < due_at - timedelta(minutes=30):
                            status = "upcoming"
                        elif now <= due_at + timedelta(hours=3):
                            status = "due"
                        else:
                            status = "overdue"
                    slots.append(
                        {
                            "date": day_s,
                            "time": slot_time or None,
                            "status": status,
                            "session_id": None if session is None else session["id"],
                            "program": {
                                "id": program["id"],
                                "title": program["title"],
                                "prescriber": program["prescriber"],
                                "prescriber_role": program["prescriber_role"],
                            },
                            "item": item,
                        }
                    )
        slots.sort(key=lambda s: (s["time"] or "99:99", s["item"]["position"]))
        return slots

    def adherence(self, days: int = 14) -> list[dict[str, object]]:
        today = date.today()
        out = []
        for offset in range(days - 1, -1, -1):
            day = today - timedelta(days=offset)
            slots = self.schedule(day)
            out.append(
                {
                    "date": day.isoformat(),
                    "scheduled": len(slots),
                    "done": sum(1 for s in slots if s["status"] == "done"),
                    "partial": sum(1 for s in slots if s["status"] == "partial"),
                }
            )
        return out

    # -------------------------------------------------------- sessions

    def add_session(self, data: dict[str, object]) -> int:
        values = {k: data.get(k) for k in SESSION_FIELDS}
        if values["exercise"] not in LIBRARY:
            raise ValueError("unknown exercise")
        values["completed"] = 1 if data.get("completed", True) else 0
        if isinstance(values.get("detail"), (dict, list)):
            values["detail"] = json.dumps(values["detail"])
        values["started_at"] = float(values.get("started_at") or time.time())
        values["ended_at"] = float(values.get("ended_at") or time.time())
        for key in ("sets_done", "reps_done", "reps_target", "partials"):
            values[key] = int(values.get(key) or 0)
        with self._lock:
            cursor = self._db.execute(
                f"INSERT INTO sessions ({','.join(SESSION_FIELDS)}) "  # nosec B608 - column names from fixed whitelist; values are parameters
                f"VALUES ({','.join('?' for _ in SESSION_FIELDS)})",
                tuple(values[k] for k in SESSION_FIELDS),
            )
            return int(cursor.lastrowid or 0)

    def sessions(self, days: float = 30.0, limit: int = 300) -> list[dict[str, object]]:
        since = time.time() - days * 86400.0
        with self._lock:
            rows = [
                dict(r)
                for r in self._db.execute(
                    "SELECT * FROM sessions WHERE started_at >= ? ORDER BY started_at DESC LIMIT ?",
                    (since, limit),
                )
            ]
        for row in rows:
            row["completed"] = bool(row["completed"])
            exercise = LIBRARY.get(str(row["exercise"]))
            row["exercise_name"] = exercise.name if exercise else row["exercise"]
            try:
                row["detail"] = json.loads(row["detail"]) if row["detail"] else None
            except (TypeError, ValueError):
                row["detail"] = None
        return rows

    def close(self) -> None:
        with self._lock:
            if self._secure is not None:
                self._secure.close()
            else:
                self._db.close()


def _item_in(data: dict[str, object]) -> dict[str, object]:
    exercise = str(data.get("exercise") or "")
    if exercise not in LIBRARY:
        raise ValueError(f"unknown exercise '{exercise}'")
    spec = LIBRARY[exercise]
    days = data.get("days")
    if isinstance(days, list):
        days = "".join("1" if int(d) in days else "0" for d in range(7))
    days = str(days or "1111111")
    if len(days) != 7 or set(days) - {"0", "1"}:
        raise ValueError("days must be 7 characters of 0/1 (Mon..Sun)")
    times = data.get("times") or []
    if isinstance(times, str):
        times = [t.strip() for t in times.split(",") if t.strip()]
    clean_times = []
    for value in times:
        hour, minute = (int(x) for x in str(value).split(":"))
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError("times must be HH:MM")
        clean_times.append(f"{hour:02d}:{minute:02d}")

    def number(key: str, default: float | None) -> float | None:
        value = data.get(key)
        return default if value in (None, "") else float(value)

    return {
        "exercise": exercise,
        "sets": max(1, min(20, int(number("sets", spec.default_sets) or 1))),
        "reps": max(1, min(200, int(number("reps", spec.default_reps) or 1))),
        "hold_s": max(
            0,
            min(600, int(number("hold_s", spec.default_hold_s if spec.kind == "hold" else 0) or 0)),
        ),
        "rest_s": max(0, min(900, int(number("rest_s", spec.default_rest_s) or 0))),
        "side": str(data.get("side") or "auto")
        if data.get("side") in ("auto", "left", "right", "both")
        else "auto",
        "days": days,
        "times": ",".join(sorted(set(clean_times))),
        "target_rom": number("target_rom", None),
        "max_hr": number("max_hr", None),
        "instructions": (str(data.get("instructions")) if data.get("instructions") else None),
        "position": int(number("position", 0) or 0),
    }


def _item_out(row: dict[str, object]) -> dict[str, object]:
    out = dict(row)
    out["times"] = [t for t in str(row.get("times") or "").split(",") if t]
    exercise = LIBRARY.get(str(row.get("exercise")))
    out["exercise_name"] = exercise.name if exercise else row.get("exercise")
    out["kind"] = exercise.kind if exercise else "reps"
    return out
