"""
Test Telegram bot connectivity.

Verifies that the bot token and chat ID in config.yaml are valid
by sending a test message.

Usage:
    python scripts/test_telegram.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config


async def main():
    cfg = load_config()

    if not cfg.telegram.bot_token or not cfg.telegram.chat_id:
        print("❌ ERROR: Set telegram.bot_token and telegram.chat_id in config.yaml")
        sys.exit(1)

    from telegram import Bot

    bot = Bot(token=cfg.telegram.bot_token)

    print("Testing Telegram bot connection...")
    me = await bot.get_me()
    print(f"  ✓ Bot connected: @{me.username} ({me.first_name})")

    await bot.send_message(
        chat_id=cfg.telegram.chat_id,
        text="✅ CheckCam Telegram integration test successful!",
    )
    print(f"  ✓ Test message sent to chat_id: {cfg.telegram.chat_id}")
    print("\n✅ Telegram integration is working.")


if __name__ == "__main__":
    asyncio.run(main())
