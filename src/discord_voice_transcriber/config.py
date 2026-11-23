"""Environment configuration for the bot."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    discord_token: str
    stt_provider: str = "openai"
    stt_model: str | None = None
    stt_language: str | None = None
    discord_guild_id: int | None = None

    @classmethod
    def from_env(cls) -> Settings:
        token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
        if not token:
            raise ValueError("DISCORD_BOT_TOKEN is required")
        guild = os.getenv("DISCORD_GUILD_ID", "").strip()
        return cls(
            discord_token=token,
            stt_provider=os.getenv("STT_PROVIDER", "openai").strip().lower(),
            stt_model=os.getenv("STT_MODEL") or None,
            stt_language=os.getenv("STT_LANGUAGE") or None,
            discord_guild_id=int(guild) if guild else None,
        )
