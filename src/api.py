"""
FastAPI application — search API + ZeroMQ background processor.

Start with: uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
import asyncio
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
from src.processor import ClipProcessor
from src.alerting import TelegramAlerter
from src.zmq_bridge import ZmqPuller

logger = structlog.get_logger()

# --- Global state (initialized in lifespan) ---
db: Database = None
embedder: SigLIPEmbedder = None
processor: ClipProcessor = None
zmq_puller: ZmqPuller = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model, connect DB, start ZMQ listener."""
    global db, embedder, processor, zmq_puller

    cfg = load_config()

    # 1. Database
    db = Database(cfg.storage.database_path, cfg.model.embedding_dim)
    db.connect()
    logger.info("database_connected", path=cfg.storage.database_path)

    # 2. Embedding engine (resident in memory)
    embedder = SigLIPEmbedder(cfg.storage.model_dir, cfg.model)
    logger.info("embedding_engine_loaded")

    # 3. Telegram alerter
    alerter = None
    if cfg.telegram.enabled and cfg.telegram.bot_token:
        alerter = TelegramAlerter(cfg.telegram.bot_token, cfg.telegram.chat_id)
        logger.info("telegram_alerter_enabled")

    # 4. Clip processor (loads classification rules from SQLite)
    processor = ClipProcessor(
        embedder=embedder, db=db, alerter=alerter,
        confidence_threshold=cfg.rules.confidence_threshold,
        keyframes_per_clip=cfg.model.keyframes_per_clip
    )

    # 5. ZeroMQ listener background task
    zmq_puller = ZmqPuller(cfg.zmq_ipc_path)
    zmq_task = asyncio.create_task(_zmq_listener_loop())
    logger.info("zmq_listener_started", ipc=cfg.zmq_ipc_path)

    yield  # --- App is running ---

    # Shutdown
    zmq_task.cancel()
    zmq_puller.close()
    db.close()
    logger.info("shutdown_complete")


async def _zmq_listener_loop():
    """Background loop: listen for ZMQ signals, process targeted or pending clips."""
    while True:
        try:
            msg = await asyncio.to_thread(zmq_puller.receive, 1000)
            if msg:
                logger.info("zmq_signal_received", message=msg)
                msg_str = msg.strip()
                if msg_str.isdigit():
                    await _process_single_clip(int(msg_str))
                else:
                    await _process_pending_clips()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("zmq_listener_error", error=str(e))
        await asyncio.sleep(0.1)


async def _process_single_clip(clip_id: int):
    """Process a single specific clip signaled by ID."""
    clip = db.get_clip_by_id(clip_id)
    if not clip:
        logger.warning("clip_not_found_for_id", clip_id=clip_id)
        return

    if clip["processed"]:
        logger.debug("clip_already_processed", clip_id=clip_id)
        return

    logger.info("processing_targeted_clip", clip_id=clip_id)
    try:
        await processor.process_clip(clip)
    except Exception as e:
        logger.error("clip_processing_error",
                     clip_id=clip["id"], error=str(e))


async def _process_pending_clips():
    """Query all unprocessed clips and process them (fallback / batch)."""
    pending = db.get_unprocessed_clips()
    if not pending:
        return

    logger.info("processing_pending_clips", count=len(pending))
    for clip in pending:
        try:
            await processor.process_clip(clip)
        except Exception as e:
            logger.error("clip_processing_error",
                         clip_id=clip["id"], error=str(e))


# --- FastAPI App ---

app = FastAPI(
    title="CheckCam API",
    description="Local-first smart security camera — semantic search API",
    version="0.1.0",
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


@app.get("/api/clips/{clip_id}/video")
async def get_clip_video(clip_id: int):
    """Stream clip MP4 video."""
    clip = db.get_clip_by_id(clip_id)
    if not clip or not Path(clip["file_path"]).exists():
        raise HTTPException(status_code=404, detail="Clip video file not found")
    return FileResponse(clip["file_path"], media_type="video/mp4")


@app.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "ok",
        "model_loaded": embedder is not None,
        "db_connected": db is not None,
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
        processor.reload_rules()
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

    processor.reload_rules()
    return {"status": "deleted", "rule_id": rule_id}


@app.post("/rules/reload")
async def reload_rules():
    """Explicitly trigger in-memory recomputation of all rule embeddings from SQLite."""
    processor.reload_rules()
    return {"status": "reloaded", "rule_count": len(db.get_rules())}


# --- Static Mounts ---
Path("data/thumbnails").mkdir(parents=True, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory="data/thumbnails"), name="thumbnails")

frontend_dist = Path("frontend/dist")
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")


