"""
Capture Daemon — Motion-triggered clip recording.

Runs as a standalone process. Continuously reads from a USB webcam,
detects motion via frame differencing, records 5-second MP4 clips,
registers them in SQLite, and signals the processing engine via ZeroMQ.

Usage:
    python -m src.capture
"""

import cv2
import time
import numpy as np
from datetime import datetime
from pathlib import Path
import structlog

from src.config import load_config
from src.database import Database
from src.zmq_bridge import ZmqPusher

logger = structlog.get_logger()


class MotionDetector:
    """Frame-difference based motion detection with Gaussian blur."""

    def __init__(self, threshold: int, min_area_pct: float):
        self.threshold = threshold
        self.min_area_pct = min_area_pct
        self.prev_gray = None

    def detect(self, frame: np.ndarray) -> bool:
        """
        Compare current frame against previous frame.
        Returns True if motion area exceeds min_area_pct of total frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False

        delta = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray

        thresh = cv2.threshold(
            delta, self.threshold, 255, cv2.THRESH_BINARY
        )[1]
        motion_pct = (np.count_nonzero(thresh) / thresh.size) * 100

        return motion_pct >= self.min_area_pct


class ClipWriter:
    """Writes fixed-duration MP4 clips from buffered video frames."""

    def __init__(self, clip_dir: str, fps: int, resolution: tuple[int, int]):
        self.clip_dir = Path(clip_dir)
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.resolution = resolution
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    def generate_filename(self) -> tuple[str, str]:
        """Generate a timestamped filename and ISO timestamp."""
        ts = datetime.now()
        filename = f"clip_{ts.strftime('%Y%m%d_%H%M%S')}.mp4"
        filepath = str(self.clip_dir / filename)
        timestamp = ts.isoformat()
        return filepath, timestamp

    def write_clip(self, frames: list[np.ndarray]) -> tuple[str, str]:
        """
        Write buffered frames to an MP4 file.
        Returns (filepath, timestamp).
        """
        filepath, timestamp = self.generate_filename()
        writer = cv2.VideoWriter(
            filepath, self.fourcc, self.fps,
            (self.resolution[0], self.resolution[1])
        )
        for frame in frames:
            writer.write(frame)
        writer.release()
        return filepath, timestamp


def run_capture_daemon(config_path: str = "config.yaml"):
    """
    Main capture loop — runs until interrupted with Ctrl+C.

    Flow:
    1. Read frames continuously from camera
    2. When motion detected → buffer frames for clip_duration_sec
    3. Write clip to disk → insert metadata in SQLite → signal via ZMQ
    4. Resume monitoring
    """
    cfg = load_config(config_path)

    # Initialize components
    db = Database(cfg.storage.database_path, cfg.model.embedding_dim)
    db.connect()

    detector = MotionDetector(
        cfg.recording.motion_threshold,
        cfg.recording.min_motion_area_pct,
    )
    writer = ClipWriter(
        cfg.storage.clip_dir, cfg.camera.fps, tuple(cfg.camera.resolution)
    )
    pusher = ZmqPusher(cfg.zmq_ipc_path)

    # Open camera
    cap = cv2.VideoCapture(cfg.camera.source)
    if isinstance(cfg.camera.resolution, list) and len(cfg.camera.resolution) == 2:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.resolution[1])
    cap.set(cv2.CAP_PROP_FPS, cfg.camera.fps)

    if not cap.isOpened():
        logger.error("camera_open_failed", source=cfg.camera.source)
        return

    logger.info(
        "capture_daemon_started",
        source=cfg.camera.source,
        resolution=cfg.camera.resolution,
        fps=cfg.camera.fps,
        clip_duration=cfg.recording.clip_duration_sec,
        motion_threshold=cfg.recording.motion_threshold,
        min_motion_area_pct=cfg.recording.min_motion_area_pct,
    )

    frames_per_clip = cfg.camera.fps * cfg.recording.clip_duration_sec
    frame_delay = 1.0 / cfg.camera.fps

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("frame_read_failed")
                time.sleep(0.1)
                continue

            if detector.detect(frame):
                logger.info("motion_detected")

                # Buffer frames for the full clip duration
                clip_frames = [frame]
                for _ in range(frames_per_clip - 1):
                    ret, frame = cap.read()
                    if ret:
                        clip_frames.append(frame)
                    time.sleep(frame_delay)

                # Write clip to disk
                filepath, timestamp = writer.write_clip(clip_frames)
                duration = len(clip_frames) / cfg.camera.fps

                # Register in database (processed=0)
                clip_id = db.insert_clip(filepath, timestamp, duration)
                logger.info(
                    "clip_saved",
                    clip_id=clip_id,
                    path=filepath,
                    frames=len(clip_frames),
                    duration=f"{duration:.1f}s",
                )

                # Signal processing engine with the specific clip ID (non-blocking)
                pusher.send(str(clip_id))

            else:
                time.sleep(frame_delay)

    except KeyboardInterrupt:
        logger.info("capture_daemon_stopped")
    finally:
        cap.release()
        db.close()
        pusher.close()


if __name__ == "__main__":
    run_capture_daemon()
