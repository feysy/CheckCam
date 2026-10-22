"""Configuration loader — reads config.yaml into typed dataclasses."""

import yaml
from pathlib import Path
from dataclasses import dataclass


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


from dataclasses import dataclass, field


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
class AppConfig:
    camera: CameraConfig
    recording: RecordingConfig
    storage: StorageConfig
    model: ModelConfig
    rules: RulesConfig
    telegram: TelegramConfig
    zmq_ipc_path: str
    zmq_transcode_ipc_path: str
    log_level: str


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load and validate config.yaml into typed dataclass."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return AppConfig(
        camera=CameraConfig(**raw["camera"]),
        recording=RecordingConfig(**raw["recording"]),
        storage=StorageConfig(**raw["storage"]),
        model=ModelConfig(**raw["model"]),
        rules=RulesConfig(**raw["rules"]),
        telegram=TelegramConfig(**raw["telegram"]),
        zmq_ipc_path=raw["zmq"]["ipc_path"],
        zmq_transcode_ipc_path=raw["zmq"].get(
            "transcode_ipc_path", "ipc:///tmp/checkcam_transcode.sock"
        ),
        log_level=raw["logging"]["level"],
    )
