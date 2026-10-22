"""
Clip processing pipeline.

Takes unprocessed video clips, extracts keyframes, encodes them via SigLIP,
evaluates against user-defined natural language rules, and routes actions
(DISCARD / SAVE / ALERT).
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import structlog

from src.embedding import SigLIPEmbedder
from src.database import Database
from src.alerting import TelegramAlerter
from src.config import RulesConfig

logger = structlog.get_logger()


class ClipProcessor:
    """Processes video clips: extract keyframes → embed → classify → act."""

    def __init__(
        self,
        embedder: SigLIPEmbedder,
        db: Database,
        alerter: TelegramAlerter | None,
        confidence_threshold: float = 0.05,
        keyframes_per_clip: int = 3,
    ):
        self.embedder = embedder
        self.db = db
        self.alerter = alerter
        self.confidence_threshold = confidence_threshold
        self.keyframes_per_clip = keyframes_per_clip

        # Load rules from DB and compute text embeddings
        self.reload_rules()

    def reload_rules(self):
        """Fetch all rules from the database and compute/cache rule embeddings in memory."""
        rules = self.db.get_rules()
        self.rule_labels: list[tuple[str, str]] = []  # (action_command, prompt_description)
        all_prompts: list[str] = []

        for r in rules:
            self.rule_labels.append((r["action_command"], r["description"]))
            all_prompts.append(r["description"])

        if not all_prompts:
            logger.warning("no_rules_defined_in_database")
            self.rule_embeddings = np.empty((0, self.embedder.cfg.embedding_dim))
            return

        self.rule_embeddings = self.embedder.encode_texts(all_prompts)
        logger.info(
            "rule_embeddings_loaded_from_db",
            count=len(all_prompts),
            rules=[f"[{cmd}] {desc}" for cmd, desc in self.rule_labels]
        )

    def extract_keyframes(self, video_path: str) -> list[Image.Image]:
        """
        Extract uniformly spaced keyframes from a video clip.

        For a clip with N total frames and K desired keyframes,
        samples at indices: [0, N/(K-1), 2*N/(K-1), ..., N-1]
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            logger.warning("empty_video", path=video_path)
            cap.release()
            return []

        # Calculate frame indices for uniform sampling
        n = min(self.keyframes_per_clip, total_frames)
        if n == 1:
            indices = [0]
        else:
            indices = [
                int(i * (total_frames - 1) / (n - 1)) for i in range(n)
            ]

        keyframes = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                keyframes.append(Image.fromarray(rgb))

        cap.release()
        logger.debug("keyframes_extracted",
                      path=video_path, count=len(keyframes),
                      indices=indices)
        return keyframes

    def classify_clip(
        self, image_embeddings: np.ndarray
    ) -> tuple[str, str, float, int]:
        """
        Classify a clip against all rule prompts.

        Computes similarity for each keyframe and takes the max similarity
        across keyframes for each prompt so that brief motion events aren't
        diluted by static/empty frames before or after.

        Returns:
            (action, matched_prompt, confidence_score, best_keyframe_index)
        """
        if len(self.rule_labels) == 0:
            return "SAVED", "no_rules_defined", 0.0, 0

        # Similarity matrix: (num_keyframes, num_prompts)
        frame_similarities = np.dot(image_embeddings, self.rule_embeddings.T)

        # Max similarity across all keyframes for each rule prompt
        clip_similarities = frame_similarities.max(axis=0)

        best_idx = int(np.argmax(clip_similarities))
        best_score = float(clip_similarities[best_idx])
        best_category, best_prompt = self.rule_labels[best_idx]

        # Best keyframe index for the matched prompt
        best_kf_idx = int(np.argmax(frame_similarities[:, best_idx]))

        # Check against minimum confidence threshold
        if best_score < self.confidence_threshold:
            return (
                "DISCARDED",
                f"below_threshold (best: {best_prompt}, score: {best_score:.3f})",
                best_score,
                best_kf_idx,
            )

        return best_category, best_prompt, best_score, best_kf_idx

    async def process_clip(self, clip: dict):
        """
        Full processing pipeline for a single clip.

        1. Extract keyframes from video file
        2. Encode keyframes into SigLIP image embeddings
        3. Classify against rule prompts (max across keyframes)
        4. Execute action: DISCARD (delete file), SAVE (index vectors),
           or ALERT (index vectors + send Telegram notification)
        """
        clip_id = clip["id"]
        file_path = clip["file_path"]

        logger.info("processing_clip", clip_id=clip_id, path=file_path)

        # 1. Extract keyframes
        keyframes = self.extract_keyframes(file_path)
        if not keyframes:
            logger.warning("no_keyframes_extracted", clip_id=clip_id)
            self.db.mark_processed(clip_id, "DISCARDED")
            return

        # 2. Generate image embeddings
        image_embeddings = self.embedder.encode_images(keyframes)

        # 3. Classify against rules
        action, matched_prompt, score, best_kf_idx = self.classify_clip(image_embeddings)
        logger.info(
            "clip_classified",
            clip_id=clip_id,
            action=action,
            prompt=matched_prompt,
            score=f"{score:.3f}",
        )

        # 4. Execute action
        if action in ("DISCARD", "DISCARDED"):
            # Delete the physical file from disk
            try:
                Path(file_path).unlink()
                logger.info("clip_file_deleted", clip_id=clip_id)
            except FileNotFoundError:
                pass
            self.db.mark_processed(clip_id, "DISCARDED")

        elif action in ("SAVE", "SAVED", "ALERT", "ALERTED"):
            # Mean-pool embeddings and store in vector DB
            clip_vec = image_embeddings.mean(axis=0)
            clip_vec = clip_vec / np.maximum(
                np.linalg.norm(clip_vec), 1e-12
            )
            self.db.insert_vector(clip_id, clip_vec)

            # Save best keyframe as thumbnail for UI
            thumb_dir = Path("data/thumbnails")
            thumb_dir.mkdir(parents=True, exist_ok=True)
            thumb_path = str(thumb_dir / f"thumb_{clip_id}.jpg")
            try:
                keyframes[best_kf_idx].save(thumb_path, format="JPEG", quality=85)
            except Exception as e:
                logger.error("thumbnail_save_failed", clip_id=clip_id, error=str(e))
                thumb_path = None

            action_label = "ALERTED" if "ALERT" in action.upper() else "SAVED"
            self.db.mark_processed(clip_id, action_label, thumbnail_path=thumb_path)
            logger.info("clip_indexed", clip_id=clip_id, action=action_label, thumbnail=thumb_path)

            # Send Telegram alert if action is ALERT
            if "ALERT" in action.upper() and self.alerter:
                snapshot = keyframes[best_kf_idx]  # Keyframe that matched best
                await self.alerter.send_alert(
                    clip_id=clip_id,
                    file_path=file_path,
                    timestamp=clip["timestamp"],
                    matched_prompt=matched_prompt,
                    confidence=score,
                    snapshot=snapshot,
                )
        else:
            logger.warning("unrecognized_action_fallback", clip_id=clip_id, action=action)
            self.db.mark_processed(clip_id, f"UNKNOWN_{action}")
