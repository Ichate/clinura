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

    def start(self) -> None:
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
        try:
            self._cap = cv2.VideoCapture(self.config.camera_index, backend)
        except Exception as exc:
            raise CaptureError(f"Unable to open camera {self.config.camera_index}: {exc}") from exc
        if self._cap is None or not self._cap.isOpened():
            self._release()
            raise CaptureError(
                f"Unable to open camera {self.config.camera_index}. Check that no other app "
                "is using it and that your terminal or IDE has camera access (System "
                "Settings > Privacy & Security > Camera)."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.config.manual_exposure:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self._cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure_value)
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
        try:
            _, frame = self.read(timeout=max(2.0, self.config.warmup_s + 1.5))
        except queue.Empty as exc:
            self.stop()
            raise CaptureError(
                "The camera opened but delivered no frames. It may be in use by another app."
            ) from exc
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
