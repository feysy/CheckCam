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

        # Reset prev_gray if not initialized or if frame resolution changed dynamically
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
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

    def __init__(self, clip_dir: str, fps: int, resolution: tuple[int, int] | None = None):
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
        Uses actual frame dimensions directly to prevent VideoWriter/codec geometry mismatch.
        Returns (filepath, timestamp).
        """
        if not frames:
            return "", ""

        filepath, timestamp = self.generate_filename()
        # Derive geometry directly from actual frames in buffer
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(filepath, self.fourcc, self.fps, (w, h))

        if not writer.isOpened():
            logger.error(
                "videowriter_open_failed",
                filepath=filepath,
                width=w,
                height=h,
                fps=self.fps
            )
            return "", ""

        for frame in frames:
            # Guarantee frame size matches writer geometry
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
        writer.release()
        return filepath, timestamp


class BaseCameraSource:
    """Abstract interface for video capture sources."""

    def is_opened(self) -> bool:
        raise NotImplementedError

    def read(self) -> tuple[bool, np.ndarray | None]:
        raise NotImplementedError

    def release(self):
        raise NotImplementedError


class OpenCVCameraSource(BaseCameraSource):
    """Standard USB webcam, RTSP stream, or video file via OpenCV VideoCapture."""

    def __init__(self, source: int | str, resolution: tuple[int, int], fps: int):
        self.source = source
        # Handle GStreamer pipeline string if libcamerasrc is specified directly
        if isinstance(source, str) and source.startswith("libcamerasrc"):
            self.cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(source)

        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            # On Linux V4L2 USB webcams, request MJPG for resolutions above 640x480.
            # USB 2.0 uncompressed YUYV does not have enough bus bandwidth for 720p/1080p,
            # so the Linux UVC driver will silently reject high resolutions unless MJPG is requested.
            if resolution[0] > 640 or resolution[1] > 480:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "opencv_camera_opened",
            source=source,
            requested_resolution=resolution,
            actual_resolution=(actual_w, actual_h),
            fps=actual_fps,
        )

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.cap.read()

    def release(self):
        self.cap.release()


class RPiCameraSource(BaseCameraSource):
    """Raspberry Pi CSI Camera Module (Module 2, 3, HQ, Global Shutter) via Picamera2."""

    def __init__(self, resolution: tuple[int, int], fps: int):
        try:
            from picamera2 import Picamera2
        except ImportError:
            raise ImportError(
                "picamera2 library not found. To use the Raspberry Pi Camera Module, "
                "install it via apt: 'sudo apt install python3-picamera2' or pip."
            )

        self.picam2 = Picamera2()
        width, height = resolution[0], resolution[1]

        # Configure Picamera2 with BGR888 format for direct OpenCV compatibility
        video_config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "BGR888"},
            controls={"FrameRate": fps}
        )
        self.picam2.configure(video_config)
        self.picam2.start()
        self._running = True
        logger.info("picamera2_started", resolution=resolution, fps=fps)

    def is_opened(self) -> bool:
        return self._running

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._running:
            return False, None
        try:
            frame = self.picam2.capture_array()
            return True, frame
        except Exception as e:
            logger.error("picamera2_read_failed", error=str(e))
            return False, None

    def release(self):
        if self._running:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self._running = False
            logger.info("picamera2_closed")


def create_camera_source(source: int | str, resolution: tuple[int, int], fps: int) -> BaseCameraSource:
    """
    Factory to instantiate either a standard OpenCV camera or Raspberry Pi Picamera2.

    Args:
        source: Integer device index (e.g. 0), RTSP string, or 'picam'/'rpi'/'csi'.
        resolution: (width, height) tuple.
        fps: Target framerate.
    """
    if isinstance(source, str) and source.lower() in ("picam", "picamera", "rpi", "csi", "picamera2"):
        return RPiCameraSource(resolution, fps)

    return OpenCVCameraSource(source, resolution, fps)


def run_capture_daemon(config_path: str = "config.yaml"):
    """
    Main capture loop — runs until interrupted.

    Continuously monitors camera stream for motion. When motion is detected,
    buffers frames for clip_duration_sec, writes MP4 to disk, inserts clip
    record into SQLite with processed=0, and sends clip_id via ZeroMQ.
    """
    cfg = load_config(config_path)

    # Initialize components
    db = Database(cfg.storage.database_path, cfg.model.embedding_dim)
    db.connect()

    current_camera_source = cfg.camera.source
    current_camera_fps = cfg.camera.fps
    current_camera_res = tuple(cfg.camera.resolution)
    clip_duration_sec = cfg.recording.clip_duration_sec

    detector = MotionDetector(
        cfg.recording.motion_threshold,
        cfg.recording.min_motion_area_pct,
    )
    writer = ClipWriter(
        cfg.storage.clip_dir, current_camera_fps, current_camera_res
    )
    pusher = ZmqPusher(cfg.zmq_ipc_path)

    # Open camera (USB webcam, RTSP stream, or Raspberry Pi Camera)
    try:
        cap = create_camera_source(
            current_camera_source, current_camera_res, current_camera_fps
        )
    except Exception as e:
        logger.error("camera_init_failed", error=str(e), source=current_camera_source)
        db.close()
        pusher.close()
        return

    if not cap.is_opened():
        logger.error("camera_open_failed", source=current_camera_source)
        db.close()
        pusher.close()
        return

    logger.info(
        "capture_daemon_started",
        source=current_camera_source,
        resolution=current_camera_res,
        fps=current_camera_fps,
        clip_duration=clip_duration_sec,
        motion_threshold=cfg.recording.motion_threshold,
        min_motion_area_pct=cfg.recording.min_motion_area_pct,
    )

    frames_per_clip = current_camera_fps * clip_duration_sec
    frame_delay = 1.0 / max(current_camera_fps, 1)
    last_settings_check = time.time()

    try:
        while True:
            # Periodically reload dynamic settings from database without restarting daemon (every 10s)
            now = time.time()
            if now - last_settings_check > 10.0:
                last_settings_check = now
                try:
                    cur_settings = db.get_settings()
                    rec = cur_settings.get("recording", {})
                    st = cur_settings.get("storage", {})
                    cam = cur_settings.get("camera", {})

                    detector.threshold = rec.get("motion_threshold", detector.threshold)
                    detector.min_area_pct = rec.get("min_motion_area_pct", detector.min_area_pct)
                    clip_duration_sec = min(rec.get("clip_duration_sec", clip_duration_sec), 10)

                    if "clip_dir" in st and st["clip_dir"]:
                        writer.clip_dir = Path(st["clip_dir"])
                        writer.clip_dir.mkdir(parents=True, exist_ok=True)

                    # Dynamic camera reconfiguration if resolution, fps, or source changed in UI settings
                    cam_source = cam.get("source", current_camera_source)
                    cam_fps = int(cam.get("fps", current_camera_fps))
                    raw_res = cam.get("resolution", current_camera_res)
                    cam_res = tuple(raw_res) if isinstance(raw_res, (list, tuple)) else current_camera_res

                    if (cam_source != current_camera_source or
                        cam_fps != current_camera_fps or
                        cam_res != current_camera_res):
                        logger.info(
                            "reconfiguring_camera",
                            source=cam_source,
                            resolution=cam_res,
                            fps=cam_fps
                        )
                        cap.release()
                        try:
                            cap = create_camera_source(cam_source, cam_res, cam_fps)
                            current_camera_source = cam_source
                            current_camera_fps = cam_fps
                            current_camera_res = cam_res
                            writer.fps = cam_fps
                            writer.resolution = cam_res
                            detector.prev_gray = None
                        except Exception as cam_err:
                            logger.error("camera_reconfigure_failed", error=str(cam_err))

                    frames_per_clip = current_camera_fps * clip_duration_sec
                    frame_delay = 1.0 / max(current_camera_fps, 1)
                except Exception as e:
                    logger.debug("capture_settings_sync_error", error=str(e))

            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("frame_read_failed")
                time.sleep(0.5)
                continue

            if detector.detect(frame):
                logger.info("motion_detected")

                # Buffer frames for the full clip duration
                clip_frames = [frame]
                for _ in range(frames_per_clip - 1):
                    ret, f = cap.read()
                    if ret and f is not None:
                        clip_frames.append(f)
                    time.sleep(frame_delay)

                # Write clip to disk
                filepath, timestamp = writer.write_clip(clip_frames)
                if not filepath:
                    logger.error("failed_to_write_clip")
                    continue

                duration = len(clip_frames) / max(current_camera_fps, 1)

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

