from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from .bot import VoiceTranscriberBot
from .config import Settings
from .transcription import Transcriber, is_english_only_model


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
    try:
        settings = Settings.from_env()
        bot = build_bot(settings)
    except ValueError as error:
        # Configuration problems are user errors, not crashes: report them in one
        # actionable line instead of a traceback.
        raise SystemExit(f"Configuration error: {error}") from error
    if args.check:
        print(
            f"Configuration valid: provider={bot.transcriber.provider} "
            f"(requested={settings.stt_provider}), "
            f"model={bot.transcriber.model}, guild={settings.discord_guild_id or 'global'}"
        )
        if is_english_only_model(bot.transcriber.model):
            # Not an error: the default is English-only on purpose because it is the
            # most accurate lightweight option. But it transcribes other languages as
            # confident English nonsense rather than failing, so say so out loud.
            print(
                "Note: this model is English-only and will mistranscribe other "
                "languages instead of erroring. Set STT_MODEL=base (or another "
                "multilingual model) for non-English voice channels."
            )
        return
    bot.run(settings.discord_token, log_handler=None)


if __name__ == "__main__":
    main()
