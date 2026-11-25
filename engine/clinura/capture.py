from __future__ import annotations

import logging
import platform
import queue
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .config import CaptureConfig

logger = logging.getLogger(__name__)


def _camera_backends() -> list[int]:
    """Capture backends to try, in order, for this platform.

    Windows needs more than one: DirectShow is the low-latency default, but many
    USB/UVC devices (including Android "webcam" mode) are only exposed through
    Media Foundation. Worse, DirectShow can *open* such a device and then deliver
    zero frames, so a backend is only accepted once it produces a real frame.
    """
    system = platform.system()
    if system == "Darwin":
        return [cv2.CAP_AVFOUNDATION]
    if system == "Windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    if system == "Linux":
        return [cv2.CAP_V4L2]
    return [cv2.CAP_ANY]


def _camera_backend() -> int:
    """Primary backend for this platform (kept for callers that need one)."""
    return _camera_backends()[0]


class CaptureError(RuntimeError):
    pass


class BlackFrameError(CaptureError):
    pass


def frame_brightness(frame: np.ndarray) -> float:
    return float(frame[::16, ::16].mean())


class CameraSource:
    def __init__(self, config: CaptureConfig | None = None) -> None:
        self.config = config or CaptureConfig()
        self._cap: cv2.VideoCapture | None = None
        self._queue: queue.Queue[tuple[float, np.ndarray]] = queue.Queue(maxsize=1)
        self._latest_lock = threading.Lock()
        self._latest: tuple[float, np.ndarray] | None = None
        self._stop = threading.Event()
        self._failed = threading.Event()
        self._thread: threading.Thread | None = None
        self._times_lock = threading.Lock()
        self._frame_times: deque[float] = deque()
        self._actual_width = 0
        self._actual_height = 0
        self._actual_fps = 0.0

    def _open_backend(self, backend: int) -> cv2.VideoCapture | None:
        """Open the configured index with one backend and apply capture settings."""
        try:
            cap = cv2.VideoCapture(self.config.camera_index, backend)
        except Exception:
            logger.debug("Backend %s raised while opening camera", backend, exc_info=True)
            return None
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.config.manual_exposure:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure_value)
        return cap

    def _first_frame(self, cap: cv2.VideoCapture, timeout: float) -> np.ndarray | None:
        """Read one frame directly, tolerating the warm-up frames some drivers drop."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                return frame
            time.sleep(0.02)
        return None

    def start(self) -> None:
        # Try every backend for this platform and accept the first one that both
        # opens *and* delivers a real frame. On Windows this is what makes USB/UVC
        # cameras work: DirectShow may open the device but return no frames, so we
        # must fall through to Media Foundation instead of giving up.
        backends = _camera_backends()
        errors: list[str] = []
        frame: np.ndarray | None = None
        for backend in backends:
            cap = self._open_backend(backend)
            if cap is None:
                errors.append(f"backend {backend}: could not open")
                continue
            probe = self._first_frame(cap, timeout=max(2.0, self.config.warmup_s + 1.5))
            if probe is None:
                errors.append(f"backend {backend}: opened but delivered no frames")
                try:
                    cap.release()
                except Exception:
                    logger.debug("Probe release failed", exc_info=True)
                continue
            self._cap = cap
            frame = probe
            logger.info("Camera %s opened with backend %s", self.config.camera_index, backend)
            break

        if self._cap is None or frame is None:
            self._release()
            if platform.system() == "Darwin":
                detail = "Check that no other app is using it and allow camera access in System Settings > Privacy & Security > Camera."
            elif platform.system() == "Windows":
                detail = (
                    "Check Windows Settings > Privacy & security > Camera, enable camera access "
                    "for desktop apps, and close apps using the camera. For USB/UVC cameras, "
                    "confirm the device appears in the Windows Camera app first."
                )
            else:
                detail = "Check camera permissions and close other apps using the camera."
            tried = "; ".join(errors) if errors else "no backends available"
            raise CaptureError(
                f"Unable to open camera {self.config.camera_index}. {detail} (tried: {tried})"
            )

        self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        logger.info(
            "Camera %s: %dx%d @ %.1f fps requested; actual %dx%d @ %.1f fps",
            self.config.camera_index,
            self.config.width,
            self.config.height,
            self.config.fps,
            self._actual_width,
            self._actual_height,
            self._actual_fps,
        )
        self._stop.clear()
        self._failed.clear()
        self._thread = threading.Thread(target=self._run, name="clinura-camera", daemon=True)
        self._thread.start()
        self._actual_height, self._actual_width = frame.shape[:2]
        deadline = time.monotonic() + self.config.black_frame_timeout_s
        while frame_brightness(frame) < self.config.black_level:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop()
                raise BlackFrameError(
                    f"Camera {self.config.camera_index} is only sending black frames."
                )
            try:
                _, frame = self.read(timeout=min(1.0, remaining))
            except queue.Empty:
                continue

    def _run(self) -> None:
        assert self._cap is not None
        last_ok = time.perf_counter()
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if time.perf_counter() - last_ok > self.config.stall_timeout_s:
                    logger.error("Camera stopped delivering frames")
                    self._failed.set()
                    return
                time.sleep(0.01)
                continue
            timestamp = time.perf_counter()
            last_ok = timestamp
            if self.config.mirror:
                frame = cv2.flip(frame, 1)
            with self._latest_lock:
                self._latest = (timestamp, frame)
            with self._times_lock:
                self._frame_times.append(timestamp)
                cutoff = timestamp - 2.0
                while self._frame_times and self._frame_times[0] < cutoff:
                    self._frame_times.popleft()
            try:
                self._queue.put_nowait((timestamp, frame))
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait((timestamp, frame))
                except queue.Full:
                    pass

    def read(self, timeout: float = 2.0) -> tuple[float, np.ndarray]:
        return self._queue.get(timeout=timeout)

    def latest(self) -> tuple[float, np.ndarray] | None:
        with self._latest_lock:
            return self._latest

    @property
    def failed(self) -> bool:
        return self._failed.is_set()

    @property
    def measured_fps(self) -> float:
        with self._times_lock:
            if len(self._frame_times) < 2:
                return self._actual_fps
            span = self._frame_times[-1] - self._frame_times[0]
            count = len(self._frame_times)
        if span <= 0:
            return self._actual_fps
        return (count - 1) / span

    @property
    def size(self) -> tuple[int, int]:
        return (self._actual_width, self._actual_height)

    @property
    def requested_fps(self) -> float:
        return float(self._actual_fps or self.config.fps)

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                logger.debug("Camera release failed", exc_info=True)
            self._cap = None

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with self._latest_lock:
            self._latest = None
        self._release()


class VideoFileSource:
    def __init__(self, path: str | Path, timestamps: list[float] | None = None) -> None:
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise CaptureError(f"Unable to open video: {self.path}")
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self._timestamps = timestamps
        self._index = 0

    def read(self) -> tuple[float, np.ndarray] | None:
        ok, frame = self._cap.read()
        if not ok:
            return None
        if self._timestamps is not None and self._index < len(self._timestamps):
            timestamp = float(self._timestamps[self._index])
        else:
            timestamp = self._index / self._fps
        self._index += 1
        return timestamp, frame

    @property
    def measured_fps(self) -> float:
        return self._fps

    def stop(self) -> None:
        self._cap.release()
