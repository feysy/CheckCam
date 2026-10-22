"""
FastAPI application — semantic search API + multi-threaded background workers.

Architecture:
- Thread 1: FastAPI asyncio loop (serving HTTP search, streaming, and UI)
- Thread 2: ProcessorWorker (ZeroMQ PULL -> SigLIP ONNX inference -> SQLite)
- Thread 3: TranscoderWorker (ZeroMQ PULL -> FFmpeg H.264 encoding with faststart)

Start with: uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

import subprocess
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import structlog

from src.config import load_config
from src.database import Database
from src.embedding import SigLIPEmbedder
from src.processor import ProcessorWorker
from src.transcoder import TranscoderWorker
from src.alerting import TelegramAlerter

logger = structlog.get_logger()

# --- Global state (initialized in lifespan) ---
db: Database = None
embedder: SigLIPEmbedder = None
processor_worker: ProcessorWorker = None
transcoder_worker: TranscoderWorker = None


def _reload_processor_rules():
    """Notify the processor thread to reload rule embeddings from SQLite."""
    if processor_worker and processor_worker.processor:
        processor_worker.processor.reload_rules()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model, connect DB, start processor and transcoder worker threads."""
    global db, embedder, processor_worker, transcoder_worker

    cfg = load_config()

    # 1. Database for API queries
    db = Database(cfg.storage.database_path, cfg.model.embedding_dim)
    db.connect()
    logger.info("database_connected", path=cfg.storage.database_path)

    # 2. Embedding engine (resident in memory for natural language search)
    embedder = SigLIPEmbedder(cfg.storage.model_dir, cfg.model)
    logger.info("embedding_engine_loaded")

    # 3. Telegram alerter
    alerter = None
    if cfg.telegram.enabled and cfg.telegram.bot_token:
        alerter = TelegramAlerter(cfg.telegram.bot_token, cfg.telegram.chat_id)
        logger.info("telegram_alerter_enabled")

    # 4. Transcoder Worker Thread (dedicated H.264 video conversion)
    transcoder_worker = TranscoderWorker(
        db_path=cfg.storage.database_path,
        embedding_dim=cfg.model.embedding_dim,
        zmq_ipc_path=cfg.zmq_transcode_ipc_path,
    )
    transcoder_worker.start()
    logger.info("transcoder_thread_initialized", ipc=cfg.zmq_transcode_ipc_path)

    # 5. Processor Worker Thread (dedicated AI inference and vector indexing)
    processor_worker = ProcessorWorker(
        db_path=cfg.storage.database_path,
        embedding_dim=cfg.model.embedding_dim,
        zmq_pipeline_ipc=cfg.zmq_ipc_path,
        zmq_transcode_ipc=cfg.zmq_transcode_ipc_path,
        embedder=embedder,
        alerter=alerter,
        confidence_threshold=cfg.rules.confidence_threshold,
        keyframes_per_clip=cfg.model.keyframes_per_clip,
    )
    processor_worker.start()
    logger.info("processor_thread_initialized", ipc=cfg.zmq_ipc_path)

    yield  # --- App is running ---

    # Shutdown threads cleanly
    if processor_worker:
        processor_worker.stop()
    if transcoder_worker:
        transcoder_worker.stop()

    if processor_worker:
        processor_worker.join(timeout=3)
    if transcoder_worker:
        transcoder_worker.join(timeout=3)

    if db:
        db.close()
    logger.info("shutdown_complete")


# --- FastAPI App ---

app = FastAPI(
    title="CheckCam API",
    description="Local-first smart security camera — semantic search API",
    version="0.2.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str
    limit: int = 10


class SearchResult(BaseModel):
    clip_id: int
    file_path: str
    timestamp: str
    action: str | None
    distance: float
    duration_sec: float | None = 5.0
    thumbnail_url: str | None = None
    video_url: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


@app.post("/search", response_model=SearchResponse)
async def search_clips(req: SearchRequest):
    """Search video clips using natural language."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_embedding = embedder.encode_texts([req.query])[0]
    results = db.search_similar(query_embedding, limit=req.limit)

    return SearchResponse(
        query=req.query,
        results=[
            SearchResult(
                clip_id=r["clip_id"],
                file_path=r["file_path"],
                timestamp=r["timestamp"],
                action=r["action"],
                distance=r["distance"],
                duration_sec=r.get("duration_sec", 5.0),
                thumbnail_url=f"/thumbnails/thumb_{r['clip_id']}.jpg" if (r.get("thumbnail_path") and Path(r["thumbnail_path"]).exists()) or Path(f"data/thumbnails/thumb_{r['clip_id']}.jpg").exists() else None,
                video_url=f"/api/clips/{r['clip_id']}/video",
            )
            for r in results
        ],
    )


@app.get("/api/clips/recent")
async def get_recent_clips(limit: int = 30):
    """Fetch recent saved/alerted clips with thumbnails."""
    clips = db.get_recent_clips(limit=limit)
    return [
        {
            "clip_id": c["clip_id"],
            "file_path": c["file_path"],
            "timestamp": c["timestamp"],
            "duration_sec": c.get("duration_sec", 5.0),
            "action": c["action"],
            "thumbnail_url": f"/thumbnails/thumb_{c['clip_id']}.jpg" if (c.get("thumbnail_path") and Path(c["thumbnail_path"]).exists()) or Path(f"data/thumbnails/thumb_{c['clip_id']}.jpg").exists() else None,
            "video_url": f"/api/clips/{c['clip_id']}/video",
        }
        for c in clips
    ]


@app.api_route("/api/clips/{clip_id}/video", methods=["GET", "HEAD"])
async def get_clip_video(clip_id: int):
    """Stream clip MP4 video with browser-compatible H.264 headers."""
    clip = db.get_clip_by_id(clip_id)
    if not clip or not Path(clip["file_path"]).exists():
        raise HTTPException(status_code=404, detail="Clip video file not found")
    return FileResponse(clip["file_path"], media_type="video/mp4")


@app.get("/health")
async def health_check():
    """Basic health check for API server, models, and database."""
    return {
        "status": "ok",
        "model_loaded": embedder is not None,
        "db_connected": db is not None,
        "processor_alive": processor_worker.is_alive() if processor_worker else False,
        "transcoder_alive": transcoder_worker.is_alive() if transcoder_worker else False,
    }


@app.get("/api/capture/status")
async def get_capture_status():
    """
    Check if the camera capture daemon is actively watching.
    Queries systemctl checkcam-daemon.service (with manual process fallback).
    If inactive, the system is simply on standby / disabled, not in failure.
    """
    is_watching = False
    service_status = "inactive"

    # 1. Check systemctl is-active
    try:
        res = subprocess.run(
            ["systemctl", "is-active", "checkcam-daemon.service"],
            capture_output=True,
            text=True,
            timeout=2
        )
        service_status = res.stdout.strip()
        if service_status == "active":
            is_watching = True
    except Exception:
        pass

    # 2. Check standalone process fallback
    if not is_watching:
        try:
            res = subprocess.run(
                ["pgrep", "-f", "src.capture"],
                capture_output=True,
                text=True,
                timeout=1
            )
            if res.returncode == 0 and res.stdout.strip():
                is_watching = True
                service_status = "running"
        except Exception:
            pass

    return {
        "watching": is_watching,
        "status": service_status,
        "service": "checkcam-daemon.service",
    }


@app.get("/clips/pending")
async def get_pending_clips():
    """List clips awaiting processing."""
    return db.get_unprocessed_clips()


# --- Rules & Actions Management Endpoints ---

class CreateRuleRequest(BaseModel):
    description: str
    action_id: int


@app.get("/actions")
async def list_actions():
    """List all available actions (discard, save, alert)."""
    return db.get_actions()


@app.get("/rules")
async def list_rules():
    """List all zero-shot classification rules stored in SQLite."""
    return db.get_rules()


@app.post("/rules")
async def add_rule(req: CreateRuleRequest):
    """Add a new classification rule and immediately reload in-memory embeddings."""
    if not req.description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty")

    actions = db.get_actions()
    valid_ids = {a["id"] for a in actions}
    if req.action_id not in valid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_id. Must be one of: {sorted(list(valid_ids))}"
        )

    try:
        rule_id = db.add_rule(req.description.strip(), req.action_id)
        _reload_processor_rules()
        return {
            "status": "created",
            "rule_id": rule_id,
            "description": req.description.strip(),
            "action_id": req.action_id,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/rules/{rule_id}")
async def delete_rule(rule_id: int):
    """Delete a classification rule and reload in-memory embeddings."""
    deleted = db.delete_rule(rule_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Rule not found")

    _reload_processor_rules()
    return {"status": "deleted", "rule_id": rule_id}


@app.post("/rules/reload")
async def reload_rules():
    """Explicitly trigger in-memory recomputation of all rule embeddings from SQLite."""
    _reload_processor_rules()
    return {"status": "reloaded", "rule_count": len(db.get_rules())}


# --- Dynamic Settings Endpoints ---

@app.get("/api/settings")
async def get_settings():
    """Fetch current dynamic settings stored in SQLite."""
    return db.get_settings()


@app.post("/api/settings")
async def update_settings(payload: dict):
    """
    Update dynamic settings in SQLite and apply runtime updates to workers.
    Enforces clip_duration_sec max 10.
    """
    updated = db.update_settings(payload)
    if processor_worker:
        processor_worker.update_dynamic_settings(updated)
    return {
        "status": "updated",
        "settings": updated
    }


class TestTelegramRequest(BaseModel):
    bot_token: str
    chat_id: str


@app.post("/api/telegram/test")
async def test_telegram_connection(req: TestTelegramRequest):
    """Send a test message to verify Telegram bot credentials."""
    if not req.bot_token.strip() or not req.chat_id.strip():
        raise HTTPException(status_code=400, detail="Bot token and chat ID are required")
    try:
        from telegram import Bot
        bot = Bot(token=req.bot_token.strip())
        await bot.send_message(
            chat_id=req.chat_id.strip(),
            text="🔔 *CheckCam Test Alert*\n\nYour Telegram alert connection is verified and functioning properly!",
            parse_mode="Markdown"
        )
        return {"status": "ok", "message": "Test alert sent successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Telegram test failed: {str(e)}")


# --- Static Mounts ---
Path("data/thumbnails").mkdir(parents=True, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory="data/thumbnails"), name="thumbnails")

frontend_dist = Path("frontend/dist")
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
