"""
Transcoder worker — Dedicated background thread for video conversion.

Receives clip IDs over a dedicated ZeroMQ PULL socket, transcodes raw/lean
recordings to browser-compatible H.264 (yuv420p + faststart moov atom) using
hardware acceleration (e.g. h264_v4l2m2m on Raspberry Pi) or fast software
encoders (libopenh264 / libx264), and atomically replaces the file on disk.
"""

import subprocess
import shutil
import threading
from pathlib import Path
import structlog

from src.zmq_bridge import ZmqPuller
from src.database import Database

logger = structlog.get_logger()

_CACHED_ENCODER: str | None = None
_ENCODER_CANDIDATES = [
    "h264_v4l2m2m",  # Raspberry Pi 4 hardware encoder via V4L2 M2M
    "libx264",       # Standard high-performance software encoder
    "libopenh264",   # Cisco OpenH264 software encoder
    "h264",          # General fallback
]


def detect_h264_encoder() -> str:
    """
    Probe ffmpeg for the optimal working H.264 encoder on this hardware.
    Caches the result after the initial probe.
    """
    global _CACHED_ENCODER
    if _CACHED_ENCODER is not None:
        return _CACHED_ENCODER

    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg_not_found_on_system")
        _CACHED_ENCODER = "libopenh264"
        return _CACHED_ENCODER

    try:
        res = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True,
            text=True,
            check=True
        )
        encoders_output = res.stdout
    except Exception as e:
        logger.error("failed_to_probe_ffmpeg_encoders", error=str(e))
        _CACHED_ENCODER = "libopenh264"
        return _CACHED_ENCODER

    for enc in _ENCODER_CANDIDATES:
        if enc in encoders_output:
            # Test if the encoder actually initializes and runs
            test_cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
                "-c:v", enc,
                "-f", "null", "-"
            ]
            try:
                test_res = subprocess.run(test_cmd, capture_output=True, timeout=3)
                if test_res.returncode == 0:
                    logger.info("h264_encoder_detected", encoder=enc)
                    _CACHED_ENCODER = enc
                    return enc
            except Exception:
                continue

    logger.warning("no_specialized_encoder_verified_fallback_to_libopenh264")
    _CACHED_ENCODER = "libopenh264"
    return _CACHED_ENCODER


def transcode_to_h264(
    input_path: str,
    output_path: str | None = None,
    encoder: str | None = None
) -> bool:
    """
    Transcode video file to browser-compatible H.264 (avc1 + yuv420p + faststart).

    If output_path is None, writes to a temporary file in the same directory
    and atomically replaces input_path.
    """
    if encoder is None:
        encoder = detect_h264_encoder()

    src_p = Path(input_path)
    if not src_p.exists():
        logger.warning("transcode_source_not_found", path=input_path)
        return False

    target_p = Path(output_path) if output_path else src_p.with_suffix(".tmp.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_p),
        "-c:v", encoder,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(target_p),
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if res.returncode != 0:
            logger.error("ffmpeg_transcode_failed", error=res.stderr, path=input_path)
            if target_p.exists():
                target_p.unlink()
            return False

        if output_path is None:
            # Atomic replace on same filesystem
            target_p.replace(src_p)
            logger.info("clip_transcoded_to_h264", path=input_path, encoder=encoder)
        else:
            logger.info("clip_transcoded_to_h264", path=str(target_p), encoder=encoder)

        return True

    except Exception as e:
        logger.error("transcode_exception", error=str(e), path=input_path)
        if target_p.exists():
            target_p.unlink()
        return False


class TranscoderWorker(threading.Thread):
    """
    Independent worker thread dedicated to H.264 video transcoding.

    Listens on a ZeroMQ PULL socket for clip IDs and transcodes saved/alerted
    clips in the background without blocking AI inference or HTTP serving.
    """

    def __init__(self, db_path: str, embedding_dim: int, zmq_ipc_path: str):
        super().__init__(name="TranscoderWorkerThread", daemon=True)
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.zmq_ipc_path = zmq_ipc_path
        self._stop_event = threading.Event()
        self.encoder = detect_h264_encoder()

    def run(self):
        """Worker thread entrypoint."""
        logger.info("transcoder_worker_started", ipc=self.zmq_ipc_path, encoder=self.encoder)

        # Thread-local database connection for clean concurrency
        db = Database(self.db_path, self.embedding_dim)
        db.connect()

        puller = ZmqPuller(self.zmq_ipc_path)

        try:
            while not self._stop_event.is_set():
                msg = puller.receive(timeout_ms=1000)
                if not msg:
                    continue

                msg_str = msg.strip()
                logger.info("transcode_request_received", message=msg_str)

                if msg_str.isdigit():
                    clip_id = int(msg_str)
                    clip = db.get_clip_by_id(clip_id)
                    if not clip:
                        logger.warning("transcode_clip_not_found", clip_id=clip_id)
                        continue

                    file_path = clip.get("file_path")
                    if file_path and Path(file_path).exists():
                        transcode_to_h264(file_path, encoder=self.encoder)
                    else:
                        logger.warning("transcode_file_missing", clip_id=clip_id, path=file_path)

        except Exception as e:
            logger.error("transcoder_worker_loop_error", error=str(e))
        finally:
            puller.close()
            db.close()
            logger.info("transcoder_worker_stopped")

    def stop(self):
        """Signal the worker thread to terminate."""
        self._stop_event.set()
