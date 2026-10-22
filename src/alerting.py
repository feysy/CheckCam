"""
Telegram alert sender.

Sends notification messages with a snapshot frame when an ALERT
action is triggered by the processing pipeline.
"""

import io
from PIL import Image
from telegram import Bot
import structlog

logger = structlog.get_logger()


class TelegramAlerter:
    """Sends alert notifications via Telegram bot API."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
        logger.info("telegram_alerter_initialized", chat_id=chat_id)

    async def send_alert(
        self,
        clip_id: int,
        file_path: str,
        timestamp: str,
        matched_prompt: str,
        confidence: float,
        snapshot: Image.Image,
    ):
        """
        Send a Telegram message with a snapshot image and event details.

        Args:
            clip_id: Database ID of the clip.
            file_path: Path to the saved .mp4 file.
            timestamp: ISO timestamp of the clip.
            matched_prompt: The rule prompt that triggered the alert.
            confidence: Cosine similarity score.
            snapshot: PIL Image of the middle keyframe.
        """
        caption = (
            f"🚨 *CheckCam Alert*\n\n"
            f"*Event:* {matched_prompt}\n"
            f"*Confidence:* {confidence:.1%}\n"
            f"*Time:* {timestamp}\n"
            f"*Clip:* `{file_path}`\n"
            f"*ID:* {clip_id}"
        )

        # Convert PIL Image to JPEG bytes
        buf = io.BytesIO()
        snapshot.save(buf, format="JPEG", quality=85)
        buf.seek(0)

        try:
            await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=buf,
                caption=caption,
                parse_mode="Markdown",
            )
            logger.info("telegram_alert_sent", clip_id=clip_id)
        except Exception as e:
            logger.error("telegram_alert_failed",
                         error=str(e), clip_id=clip_id)
