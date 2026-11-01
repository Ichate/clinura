from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"
HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
POSE_MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_full.task"


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"Model already present: {path} ({path.stat().st_size} bytes)")
        return
    print(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, path)
    except OSError:
        # Some Python installs (e.g. python.org on macOS) lack CA certificates.
        curl = shutil.which("curl")
        if curl is None:
            raise
        subprocess.run([curl, "-sSfL", "-o", str(path), url], check=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"Saved {path} ({path.stat().st_size} bytes)")
    print(f"sha256 {digest}")


def main() -> int:
    download(FACE_MODEL_URL, MODEL_PATH)
    download(HAND_MODEL_URL, HAND_MODEL_PATH)
    download(POSE_MODEL_URL, POSE_MODEL_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
