"""Configuration loader — reads config.yaml into typed dataclasses."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    source: int | str
    fps: int
    resolution: list[int]


@dataclass
class RecordingConfig:
    clip_duration_sec: int
    motion_threshold: int
    min_motion_area_pct: float


@dataclass
class StorageConfig:
    clip_dir: str
    database_path: str
    model_dir: str


@dataclass
class ModelConfig:
    repo_id: str
    vision_model: str
    text_model: str
    tokenizer_id: str
    embedding_dim: int
    image_size: int
    max_text_length: int
    keyframes_per_clip: int


from src.motion_attention import MotionAttentionConfig


@dataclass
class RulesConfig:
    confidence_threshold: float = 0.05
    discard: list[str] = field(default_factory=list)
    save: list[str] = field(default_factory=list)
    alert: list[str] = field(default_factory=list)


@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: str
    enabled: bool

@dataclass
class DebugConfig:
    save_keyframes: bool = False
    max_debug_clips: int = 10
    debug_dir: str = "./data/debug_keyframes"

@dataclass
class AppConfig:
    camera: CameraConfig
    recording: RecordingConfig
    storage: StorageConfig
    model: ModelConfig
    rules: RulesConfig
    telegram: TelegramConfig
    motion_attention: MotionAttentionConfig
    zmq_ipc_path: str
    zmq_transcode_ipc_path: str
    log_level: str
    debug: DebugConfig = field(default_factory=DebugConfig)


def load_config(path: str = "config.yaml", use_db: bool = True) -> AppConfig:
    """Load config.yaml with dynamic setting overrides from SQLite database."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    db_path = raw["storage"]["database_path"]
    embedding_dim = raw["model"]["embedding_dim"]

    # Base configs from YAML
    camera_data = dict(raw["camera"])
    recording_data = dict(raw["recording"])
    storage_data = dict(raw["storage"])
    model_data = dict(raw["model"])
    rules_data = dict(raw.get("rules", {}))
    telegram_data = dict(raw.get("telegram", {}))
    motion_attention_data = dict(raw.get("motion_attention", {}))
    debug_data = dict(raw.get("debug", {}))


    # Overlay from SQLite if database exists and use_db is True
    if use_db and Path(db_path).exists():
        try:
            from src.database import Database
            db = Database(db_path, embedding_dim)
            db.connect()
            db_settings = db.get_settings()
            db.close()

            if "camera" in db_settings:
                camera_data.update(db_settings["camera"])
            if "recording" in db_settings:
                recording_data.update(db_settings["recording"])
            if "storage" in db_settings and "clip_dir" in db_settings["storage"]:
                storage_data["clip_dir"] = db_settings["storage"]["clip_dir"]
            if "model" in db_settings and "keyframes_per_clip" in db_settings["model"]:
                model_data["keyframes_per_clip"] = db_settings["model"]["keyframes_per_clip"]
            if "rules" in db_settings and "confidence_threshold" in db_settings["rules"]:
                rules_data["confidence_threshold"] = db_settings["rules"]["confidence_threshold"]
            if "telegram" in db_settings:
                telegram_data.update(db_settings["telegram"])
            if "motion_attention" in db_settings:
                motion_attention_data.update(db_settings["motion_attention"])
        except Exception:
            pass

    return AppConfig(
        camera=CameraConfig(**camera_data),
        recording=RecordingConfig(**recording_data),
        storage=StorageConfig(**storage_data),
        model=ModelConfig(**model_data),
        rules=RulesConfig(**rules_data),
        telegram=TelegramConfig(**telegram_data),
        motion_attention=MotionAttentionConfig(**motion_attention_data),
        zmq_ipc_path=raw["zmq"]["ipc_path"],
        zmq_transcode_ipc_path=raw["zmq"].get(
            "transcode_ipc_path", "ipc:///tmp/checkcam_transcode.sock"
        ),
        log_level=raw["logging"]["level"],
        debug=DebugConfig(**debug_data),
    )
