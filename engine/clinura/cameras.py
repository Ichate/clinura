"""List available cameras with their names (e.g. "iPhone Camera" for Continuity Camera)."""

from __future__ import annotations

import platform
import subprocess

import cv2


def _avfoundation_names() -> list[str] | None:
    """Device names in the same order OpenCV's AVFoundation backend uses."""
    try:
        import AVFoundation as AV  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        devices = AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo)
        return [str(d.localizedName()) for d in devices]
    except Exception:
        return None


def _system_profiler_names() -> list[str]:
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType"], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [
        line.strip().rstrip(":")
        for line in out.splitlines()
        if line.startswith("    ") and not line.startswith("      ") and line.strip().endswith(":")
    ]


def _backends_for(system: str) -> list[int]:
    """Backends to probe, in order. Windows needs DirectShow *and* Media Foundation."""
    if system == "Windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    if system == "Linux":
        return [cv2.CAP_V4L2]
    return [cv2.CAP_ANY]


def _probe(index: int, backend: int) -> bool:
    """True when this backend can open the index and hand back a real frame."""
    try:
        cap = cv2.VideoCapture(index, backend)
    except Exception:
        return False
    try:
        if not cap.isOpened():
            return False
        # Opening is not enough: some Windows drivers open but never deliver frames.
        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None and getattr(frame, "size", 0):
                return True
        return False
    finally:
        cap.release()


def list_cameras(max_index: int = 5, active: int | None = None) -> list[dict[str, object]]:
    system = platform.system()
    if system == "Darwin":
        names = _avfoundation_names()
        if names is None:
            names = _system_profiler_names()
        if names:
            return [{"index": i, "name": name} for i, name in enumerate(names)]
    cameras = []
    for index in range(max_index + 1):
        if index == active:
            cameras.append({"index": index, "name": f"Camera {index}"})
            continue
        for backend in _backends_for(system):
            if _probe(index, backend):
                cameras.append({"index": index, "name": f"Camera {index}"})
                break
    return cameras
