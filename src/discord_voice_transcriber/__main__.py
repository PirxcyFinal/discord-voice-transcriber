from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from .bot import VoiceTranscriberBot
from .config import Settings
from .transcription import Transcriber


def build_bot(settings: Settings) -> VoiceTranscriberBot:
    transcriber = Transcriber(
        provider=settings.stt_provider,
        model=settings.stt_model,
        language=settings.stt_language,
    )
    return VoiceTranscriberBot(transcriber, guild_id=settings.discord_guild_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discord per-speaker voice transcriber")
    parser.add_argument("--check", action="store_true", help="Validate configuration without logging in")
    args = parser.parse_args()
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = Settings.from_env()
    bot = build_bot(settings)
    if args.check:
        print(
            f"Configuration valid: provider={bot.transcriber.provider} "
            f"(requested={settings.stt_provider}), "
            f"model={bot.transcriber.model}, guild={settings.discord_guild_id or 'global'}"
        )
        return
    bot.run(settings.discord_token, log_handler=None)


if __name__ == "__main__":
    main()
