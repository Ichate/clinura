"""Exercise library and camera-based rep counting.

Joint angles are computed from MediaPipe *world* landmarks (metric 3-D), which makes
them far less sensitive to camera angle than 2-D pixel angles. Each exercise is a
small state machine on one smoothed angle with hysteresis:

    rest ──(angle passes the target)──► reached ──(angle returns past start)──► rep ✓

A movement that turns back before reaching the target is reported as a partial rep
("go deeper") and not counted. Every rep records its range of motion, duration and
form issues, which the clinician sees afterwards. Timed holds (plank, wall sit) only
accumulate time while the position is held correctly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# MediaPipe Pose landmark indices.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

SIDES = {
    "left": {
        "shoulder": L_SHOULDER,
        "elbow": L_ELBOW,
        "wrist": L_WRIST,
        "hip": L_HIP,
        "knee": L_KNEE,
        "ankle": L_ANKLE,
    },
    "right": {
        "shoulder": R_SHOULDER,
        "elbow": R_ELBOW,
        "wrist": R_WRIST,
        "hip": R_HIP,
        "knee": R_KNEE,
        "ankle": R_ANKLE,
    },
}

SKELETON = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32), (27, 29), (28, 30),
]  # fmt: skip


@dataclass(frozen=True)
class Exercise:
    key: str
    name: str
    category: str
    joint: tuple[str, str, str]
    start: float
    target: float
    kind: str = "reps"  # "reps" | "hold"
    source: str = "pose"  # "pose" (body) | "hand" (hand landmarks)
    seated: bool = False
    per_side: bool = False
    alternate: bool = False
    hold_range: tuple[float, float] | None = None
    default_sets: int = 3
    default_reps: int = 10
    default_hold_s: int = 30
    default_rest_s: int = 60
    setup: str = ""
    cues: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()
    min_rep_s: float = 0.5

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "name": self.name,
            "category": self.category,
            "kind": self.kind,
            "source": self.source,
            "seated": self.seated,
            "joint": "-".join(self.joint),
            "start_angle": self.start,
            "target_angle": self.target,
            "per_side": self.per_side,
            "default_sets": self.default_sets,
            "default_reps": self.default_reps,
            "default_hold_s": self.default_hold_s,
            "default_rest_s": self.default_rest_s,
            "setup": self.setup,
            "cues": list(self.cues),
        }


LIBRARY: dict[str, Exercise] = {
    e.key: e
    for e in (
        Exercise(
            "squat",
            "Squat",
            "strength",
            ("hip", "knee", "ankle"),
            160,
            100,
            setup="Stand side-on or at 45° to the camera, whole body in view.",
            cues=("Sit back as if into a chair", "Keep your chest up", "Knees over toes"),
            checks=("trunk_lean", "knee_symmetry"),
        ),
        Exercise(
            "sit_to_stand",
            "Sit to stand",
            "rehab",
            ("shoulder", "hip", "knee"),
            160,
            110,
            default_reps=8,
            setup="Chair side-on to the camera. Arms crossed if you can.",
            cues=("Lean forward, then stand tall", "Sit down slowly with control"),
        ),
        Exercise(
            "lunge",
            "Lunge",
            "strength",
            ("hip", "knee", "ankle"),
            155,
            105,
            per_side=True,
            setup="Side-on to the camera with room to step forward.",
            cues=("Step long, drop the back knee", "Front knee stays over the ankle"),
            checks=("trunk_lean",),
        ),
        Exercise(
            "pushup",
            "Push-up",
            "strength",
            ("shoulder", "elbow", "wrist"),
            150,
            95,
            setup="Side-on to the camera, whole body in view. Knees down is fine.",
            cues=("Lower your chest toward the floor", "Body in one straight line"),
            checks=("body_line",),
        ),
        Exercise(
            "bicep_curl",
            "Biceps curl",
            "strength",
            ("shoulder", "elbow", "wrist"),
            145,
            55,
            per_side=True,
            setup="Face the camera, upper body in view.",
            cues=("Elbows stay by your sides", "Lower all the way down"),
            checks=("elbow_drift",),
        ),
        Exercise(
            "shoulder_press",
            "Shoulder press",
            "strength",
            ("shoulder", "elbow", "wrist"),
            95,
            155,
            setup="Face the camera, upper body in view. Start with hands at shoulder height.",
            cues=("Press straight up", "Don't arch your back"),
        ),
        Exercise(
            "lateral_raise",
            "Lateral raise",
            "strength",
            ("hip", "shoulder", "elbow"),
            25,
            80,
            default_reps=12,
            setup="Face the camera, upper body in view.",
            cues=("Raise arms out to shoulder height", "Lead with the elbows"),
            checks=("over_raise",),
        ),
        Exercise(
            "arm_raise",
            "Forward arm raise (shoulder flexion)",
            "rehab",
            ("hip", "shoulder", "elbow"),
            30,
            140,
            per_side=True,
            setup="Side-on to the camera so the arm moves across the view.",
            cues=("Raise the straight arm forward and up", "Stop if it hurts"),
        ),
        Exercise(
            "knee_extension",
            "Seated knee extension",
            "rehab",
            ("hip", "knee", "ankle"),
            100,
            155,
            per_side=True,
            setup="Sit side-on to the camera, the working leg nearest the camera.",
            cues=("Straighten the knee fully", "Hold 1 second at the top"),
        ),
        Exercise(
            "glute_bridge",
            "Glute bridge",
            "rehab",
            ("shoulder", "hip", "knee"),
            130,
            165,
            default_reps=12,
            setup="Lie on your back side-on to the camera, knees bent.",
            cues=("Squeeze glutes and lift the hips", "Shoulders to knees in one line"),
        ),
        Exercise(
            "marching",
            "High-knee march",
            "cardio",
            ("shoulder", "hip", "knee"),
            155,
            115,
            alternate=True,
            default_reps=20,
            default_rest_s=45,
            setup="Face the camera, whole body in view.",
            cues=("Bring each knee to hip height", "Swing the opposite arm"),
        ),
        Exercise(
            "jumping_jack",
            "Jumping jacks",
            "cardio",
            ("hip", "shoulder", "elbow"),
            40,
            135,
            default_reps=20,
            default_rest_s=45,
            setup="Face the camera, whole body in view.",
            cues=("Arms overhead as the feet go wide", "Land softly"),
        ),
        Exercise(
            "plank",
            "Plank",
            "core",
            ("shoulder", "hip", "ankle"),
            0,
            0,
            kind="hold",
            hold_range=(160.0, 195.0),
            default_reps=1,
            default_hold_s=30,
            setup="Side-on to the camera, whole body in view.",
            cues=("Body in one straight line", "Brace your stomach"),
        ),
        Exercise(
            "wall_sit",
            "Wall sit",
            "strength",
            ("hip", "knee", "ankle"),
            0,
            0,
            kind="hold",
            hold_range=(70.0, 115.0),
            default_reps=1,
            default_hold_s=30,
            setup="Back against a wall, side-on to the camera.",
            cues=("Thighs parallel to the floor", "Knees over ankles"),
        ),
        Exercise(
            "seated_march",
            "Seated march",
            "seated",
            ("shoulder", "hip", "knee"),
            95,
            72,
            alternate=True,
            seated=True,
            default_reps=20,
            default_rest_s=45,
            setup="Sit tall on a chair side-on or at 45° to the camera, hips and knees in view.",
            cues=("Lift one knee up, then the other", "Keep your back tall"),
        ),
        Exercise(
            "seated_overhead_reach",
            "Seated overhead reach",
            "seated",
            ("hip", "shoulder", "elbow"),
            35,
            145,
            seated=True,
            default_reps=10,
            setup="Sit facing the camera, upper body in view.",
            cues=("Reach both arms up overhead", "Lower slowly"),
        ),
        Exercise(
            "seated_curl",
            "Seated biceps curl",
            "seated",
            ("shoulder", "elbow", "wrist"),
            145,
            55,
            per_side=True,
            seated=True,
            default_reps=12,
            setup="Sit facing the camera, upper body in view. A water bottle works as a weight.",
            cues=("Elbow stays by your side", "Lower all the way"),
            checks=("elbow_drift",),
        ),
        Exercise(
            "seated_press",
            "Seated shoulder press",
            "seated",
            ("shoulder", "elbow", "wrist"),
            95,
            155,
            seated=True,
            setup="Sit facing the camera, hands at shoulder height.",
            cues=("Press straight up", "Keep your back against the chair"),
        ),
        Exercise(
            "seated_side_raise",
            "Seated side arm raise",
            "seated",
            ("hip", "shoulder", "elbow"),
            25,
            80,
            seated=True,
            default_reps=12,
            setup="Sit facing the camera, upper body in view.",
            cues=("Raise arms out to shoulder height", "Shoulders relaxed"),
            checks=("over_raise",),
        ),
        Exercise(
            "fist_open_close",
            "Fist open / close",
            "hand",
            ("hand", "hand", "hand"),
            165,
            115,
            source="hand",
            seated=True,
            default_reps=15,
            default_rest_s=30,
            min_rep_s=0.3,
            setup="Hold one hand up to the camera, palm facing it, about 40-60 cm away.",
            cues=("Make a full fist", "Then spread your fingers wide"),
        ),
        Exercise(
            "pinch",
            "Thumb-to-finger pinch",
            "hand",
            ("hand", "hand", "hand"),
            55,
            18,
            source="hand",
            seated=True,
            default_reps=15,
            default_rest_s=30,
            min_rep_s=0.3,
            setup="Hold one hand up to the camera, palm facing it.",
            cues=("Touch thumb and index fingertip", "Open fully between pinches"),
        ),
        Exercise(
            "finger_spread",
            "Finger spread",
            "hand",
            ("hand", "hand", "hand"),
            25,
            55,
            source="hand",
            seated=True,
            default_reps=15,
            default_rest_s=30,
            min_rep_s=0.3,
            setup="Hold one hand up to the camera, palm facing it, fingers together.",
            cues=("Spread your fingers as wide as you can", "Bring them back together"),
        ),
    )
}

CHECK_MESSAGES = {
    "trunk_lean": "Keep your chest up",
    "knee_symmetry": "Even weight on both legs",
    "body_line": "Keep your hips in line",
    "elbow_drift": "Keep your elbow by your side",
    "over_raise": "Stop at shoulder height",
}


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees (0-180)."""
    v1 = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    v2 = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


@dataclass
class RepRecord:
    index: int
    side: str
    t_start: float
    t_end: float
    rom_min: float
    rom_max: float
    issues: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    @property
    def form_score(self) -> int:
        return max(0, 100 - 25 * len(set(self.issues)))

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "side": self.side,
            "duration_s": round(self.duration, 2),
            "rom_min": round(self.rom_min, 1),
            "rom_max": round(self.rom_max, 1),
            "rom": round(self.rom_max - self.rom_min, 1),
            "form_score": self.form_score,
            "issues": sorted(set(self.issues)),
        }


class _SideCounter:
    """Hysteresis state machine for one angle stream."""

    def __init__(self, exercise: Exercise, side: str) -> None:
        self.ex = exercise
        self.side = side
        self.state = "rest"
        self.angle: float | None = None
        self.t_start = 0.0
        self.lo = 999.0
        self.hi = -999.0
        self.issue_frames: dict[str, int] = {}
        self.frames = 0

    @property
    def descending(self) -> bool:
        return self.ex.target < self.ex.start

    def _past(self, angle: float, threshold: float, toward_target: bool) -> bool:
        if self.descending == toward_target:
            return angle <= threshold
        return angle >= threshold

    def update(self, angle: float, t: float, issues: list[str]) -> tuple[RepRecord | None, bool]:
        """Returns (completed rep, partial rep flag)."""
        alpha = 0.45
        self.angle = angle if self.angle is None else self.angle + alpha * (angle - self.angle)
        a = self.angle
        span = abs(self.ex.start - self.ex.target)
        midway = self.ex.start + (self.ex.target - self.ex.start) * 0.45
        if self.state == "rest":
            if self._past(a, midway, toward_target=True):
                self.state = "moving"
                self.t_start = t
                self.lo, self.hi = a, a
                self.issue_frames = {}
                self.frames = 0
            return None, False
        self.lo = min(self.lo, a)
        self.hi = max(self.hi, a)
        self.frames += 1
        for issue in issues:
            self.issue_frames[issue] = self.issue_frames.get(issue, 0) + 1
        if self.state == "moving" and self._past(a, self.ex.target, toward_target=True):
            self.state = "reached"
        returned = self._past(
            a,
            self.ex.start - (0.08 * span if self.descending else -0.08 * span),
            toward_target=False,
        )
        if self.state == "reached" and returned:
            self.state = "rest"
            if t - self.t_start < self.ex.min_rep_s:
                return None, False
            persistent = [
                k for k, n in self.issue_frames.items() if n >= max(3, 0.25 * self.frames)
            ]
            return RepRecord(0, self.side, self.t_start, t, self.lo, self.hi, persistent), False
        if self.state == "moving":
            back_to_start = self._past(
                a, midway + (0.15 * span if self.descending else -0.15 * span), toward_target=False
            )
            if back_to_start:
                self.state = "rest"
                return None, True
        return None, False


def hand_metric(exercise: str, world: np.ndarray) -> float:
    """Scalar 'angle-like' value for hand exercises, from 21 hand world landmarks.

    fist_open_close: mean fingertip-to-wrist distance / palm length x 100
                     (~170 open hand, ~100 fist).
    pinch:           thumb-index tip distance / palm length x 100 (~60 open, <15 touching).
    finger_spread:   mean angle between adjacent fingers (degrees x ~2).
    """
    w = np.asarray(world, dtype=np.float64)
    palm = float(np.linalg.norm(w[9] - w[0])) or 1e-6
    if exercise == "pinch":
        return float(np.linalg.norm(w[4] - w[8]) / palm * 100.0)
    if exercise == "finger_spread":
        angles = [joint_angle(w[a], w[0], w[b]) for a, b in ((8, 12), (12, 16), (16, 20))]
        return float(np.mean(angles) * 2.0 + joint_angle(w[4], w[1], w[8]) * 0.3)
    tips = [np.linalg.norm(w[i] - w[0]) for i in (8, 12, 16, 20)]
    return float(np.mean(tips) / palm * 100.0)


class RepCounter:
    def __init__(self, exercise: Exercise | str, side: str = "auto") -> None:
        self.exercise = LIBRARY[exercise] if isinstance(exercise, str) else exercise
        self.side = side if side in ("left", "right", "both", "auto") else "auto"
        self.reps: list[RepRecord] = []
        self.partials = 0
        self.hold_s = 0.0
        self.hold_best_s = 0.0
        self._hold_streak_start: float | None = None
        self._last_t: float | None = None
        self._counters: dict[str, _SideCounter] = {}
        self.cues: list[str] = []
        self.visible = False
        self.angle: float | None = None
        self.active_side: str | None = None

    def _counter(self, side: str) -> _SideCounter:
        if side not in self._counters:
            self._counters[side] = _SideCounter(self.exercise, side)
        return self._counters[side]

    def _side_angle(self, world: np.ndarray, vis: np.ndarray, side: str) -> tuple[float, float]:
        joints = SIDES[side]
        a, b, c = (joints[name] for name in self.exercise.joint)
        visibility = float(min(vis[a], vis[b], vis[c]))
        return joint_angle(world[a], world[b], world[c]), visibility

    def _issues(self, world: np.ndarray, side: str) -> list[str]:
        j = SIDES[side]
        issues = []
        for check in self.exercise.checks:
            if check == "trunk_lean":
                trunk = world[j["shoulder"]] - world[j["hip"]]
                vertical = np.array([0.0, -1.0, 0.0])
                cos = float(np.dot(trunk, vertical) / max(np.linalg.norm(trunk), 1e-9))
                if math.degrees(math.acos(np.clip(cos, -1, 1))) > 50:
                    issues.append("trunk_lean")
            elif check == "knee_symmetry":
                left = joint_angle(world[L_HIP], world[L_KNEE], world[L_ANKLE])
                right = joint_angle(world[R_HIP], world[R_KNEE], world[R_ANKLE])
                if abs(left - right) > 20:
                    issues.append("knee_symmetry")
            elif check == "body_line":
                if joint_angle(world[j["shoulder"]], world[j["hip"]], world[j["ankle"]]) < 155:
                    issues.append("body_line")
            elif check == "elbow_drift":
                if joint_angle(world[j["hip"]], world[j["shoulder"]], world[j["elbow"]]) > 35:
                    issues.append("elbow_drift")
            elif check == "over_raise":
                if joint_angle(world[j["hip"]], world[j["shoulder"]], world[j["elbow"]]) > 105:
                    issues.append("over_raise")
        return issues

    def update_hand(self, world: np.ndarray | None, t: float) -> None:
        self._last_t = t
        if world is None:
            self.visible = False
            self.cues = ["Hold your hand up to the camera, palm facing it"]
            return
        self.visible = True
        self.active_side = "hand"
        value = hand_metric(self.exercise.key, world)
        self.angle = value
        self.cues = []
        rep, partial = self._counter("hand").update(value, t, [])
        if partial:
            self.partials += 1
            self.cues = ["Partial — move through the full range"]
        if rep is not None:
            rep.index = len(self.reps) + 1
            self.reps.append(rep)

    def update(self, world: np.ndarray | None, image: np.ndarray | None, t: float) -> None:
        dt = 0.0 if self._last_t is None else float(np.clip(t - self._last_t, 0.0, 0.5))
        self._last_t = t
        if world is None or image is None:
            self.visible = False
            self.cues = ["Step into view so your whole body is visible"]
            self._hold_streak_start = None
            return
        vis = image[:, 2]
        readings = {side: self._side_angle(world, vis, side) for side in ("left", "right")}
        if self.side in ("left", "right"):
            sides = [self.side]
        elif self.exercise.alternate or self.side == "both":
            sides = ["left", "right"]
        else:
            sides = [max(readings, key=lambda s: readings[s][1])]
        usable = [s for s in sides if readings[s][1] >= 0.5 and math.isfinite(readings[s][0])]
        self.visible = bool(usable)
        if not usable:
            self.cues = ["Make sure the moving joints are in view"]
            self._hold_streak_start = None
            return
        self.active_side = usable[0]
        self.angle = float(readings[usable[0]][0])
        issues = self._issues(world, usable[0])
        self.cues = [CHECK_MESSAGES[i] for i in issues]

        if self.exercise.kind == "hold":
            lo, hi = self.exercise.hold_range or (0.0, 180.0)
            holding = lo <= self.angle <= hi and not issues
            if holding:
                self.hold_s += dt
                if self._hold_streak_start is None:
                    self._hold_streak_start = t
                self.hold_best_s = max(self.hold_best_s, t - self._hold_streak_start)
            else:
                self._hold_streak_start = None
                self.cues = self.cues or ["Get into position"]
            return

        for side in usable:
            angle, _ = readings[side]
            rep, partial = self._counter(side).update(angle, t, issues if side == usable[0] else [])
            if partial:
                self.partials += 1
                direction = "deeper" if self.exercise.target < self.exercise.start else "higher"
                self.cues = [f"Partial rep — go {direction}"]
            if rep is not None:
                rep.index = len(self.reps) + 1
                self.reps.append(rep)

    def summary(self) -> dict[str, object]:
        reps = [r.to_dict() for r in self.reps]
        roms = [r["rom"] for r in reps]
        forms = [r["form_score"] for r in reps]
        phase = None
        if self.active_side in self._counters:
            phase = self._counters[self.active_side].state
        return {
            "exercise": self.exercise.key,
            "kind": self.exercise.kind,
            "side": self.side,
            "active_side": self.active_side,
            "reps": len(reps),
            "partials": self.partials,
            "hold_s": round(self.hold_s, 1),
            "hold_best_s": round(self.hold_best_s, 1),
            "angle": None if self.angle is None else round(self.angle, 1),
            "phase": phase,
            "visible": self.visible,
            "cues": self.cues,
            "avg_rom": round(float(np.mean(roms)), 1) if roms else None,
            "best_rom": round(float(max(roms)), 1) if roms else None,
            "avg_form": round(float(np.mean(forms)), 1) if forms else None,
            "avg_rep_s": round(float(np.mean([r["duration_s"] for r in reps])), 2)
            if reps
            else None,
            "last_rep": reps[-1] if reps else None,
            "rep_log": reps[-50:],
        }
