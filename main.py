from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "engine"
APP = ROOT / "app"
DIST = APP / "dist"
REQUIRED_MODULES = ("mediapipe", "numpy", "scipy", "cv2", "fastapi", "uvicorn", "pydantic", "httpx")

log = logging.getLogger("clinura.main")


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def ensure_python_packages(install: bool) -> None:
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if not missing:
        return
    if not install:
        sys.exit(
            f"Missing Python packages: {', '.join(missing)}. "
            f"Run: {sys.executable} -m pip install -r requirements.txt"
        )
    log.info("Installing Python requirements (missing: %s)", ", ".join(missing))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
        check=True,
    )


def ensure_models() -> None:
    models = [
        ROOT / "models" / "face_landmarker.task",
        ROOT / "models" / "hand_landmarker.task",
        ROOT / "models" / "pose_landmarker_full.task",
    ]
    if all(path.exists() and path.stat().st_size > 1_000_000 for path in models):
        return
    log.info("Downloading MediaPipe models")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "download_model.py")], check=True)


def _newest_mtime(paths: list[Path]) -> float:
    newest = 0.0
    for base in paths:
        if base.is_file():
            newest = max(newest, base.stat().st_mtime)
        elif base.is_dir():
            for path in base.rglob("*"):
                if path.is_file():
                    newest = max(newest, path.stat().st_mtime)
    return newest


def ensure_ui(force: bool) -> None:
    index = DIST / "index.html"
    sources = [
        APP / "src",
        APP / "index.html",
        APP / "package.json",
        APP / "vite.config.ts",
        APP / "tsconfig.json",
    ]
    stale = not index.exists() or _newest_mtime(sources) > index.stat().st_mtime
    if not (force or stale):
        return
    npm = shutil.which("npm")
    if npm is None:
        if index.exists():
            log.warning("npm not found; serving the existing UI build, which may be out of date")
            return
        sys.exit("The UI is not built and npm was not found. Install Node.js 20.19+ and retry.")
    if not (APP / "node_modules").exists():
        log.info("Installing UI dependencies")
        subprocess.run([npm, "install", "--no-audit", "--no-fund"], cwd=APP, check=True)
    log.info("Building the UI")
    subprocess.run([npm, "run", "build"], cwd=APP, check=True)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Clinura: engine, API and UI on one port.")
    parser.add_argument("--host", default=os.environ.get("CLINURA_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("CLINURA_PORT", "3000")))
    parser.add_argument("--camera", type=int, default=int(os.environ.get("CLINURA_CAMERA", "0")))
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-mirror", action="store_true", help="disable selfie mirroring")
    parser.add_argument("--record-dir", type=Path, default=None, help="record sessions here")
    parser.add_argument("--skip-build", action="store_true", help="do not rebuild the UI")
    parser.add_argument("--rebuild", action="store_true", help="force a UI rebuild")
    parser.add_argument(
        "--no-install", action="store_true", help="do not pip install missing packages"
    )
    parser.add_argument("--open", action="store_true", help="open the browser when ready")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    load_env(ROOT / ".env")
    ensure_python_packages(install=not args.no_install)
    ensure_models()
    if not args.skip_build:
        ensure_ui(force=args.rebuild)

    sys.path.insert(0, str(ENGINE))
    import uvicorn

    from clinura.config import CaptureConfig, ServerConfig
    from clinura.server import Runtime, create_app

    runtime = Runtime(
        capture=CaptureConfig(
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            mirror=not args.no_mirror,
        ),
        server=ServerConfig(host=args.host, port=args.port),
        record_dir=args.record_dir,
    )
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    log.info("Clinura is starting at %s", url)
    if not os.environ.get("DEEPSEEK_API_KEY"):
        log.info("DEEPSEEK_API_KEY is not set; AI handover and copilot answers are disabled")
    if args.open:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(runtime), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
