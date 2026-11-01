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


def list_cameras(max_index: int = 5, active: int | None = None) -> list[dict[str, object]]:
    if platform.system() == "Darwin":
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
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cameras.append({"index": index, "name": f"Camera {index}"})
        cap.release()
    return cameras
