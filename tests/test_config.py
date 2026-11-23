from __future__ import annotations

import pytest

from discord_voice_transcriber.config import Settings


def test_defaults_to_hermes_style_provider_autodetect(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "placeholder-token")
    monkeypatch.delenv("STT_PROVIDER", raising=False)
    monkeypatch.delenv("STT_MODEL", raising=False)
    monkeypatch.delenv("STT_LANGUAGE", raising=False)
    monkeypatch.delenv("DISCORD_GUILD_ID", raising=False)

    settings = Settings.from_env()

    assert settings.stt_provider == "auto"
    assert settings.stt_model is None
    assert settings.stt_language is None
    assert settings.discord_guild_id is None


def test_environment_overrides_are_applied(monkeypatch):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "placeholder-token")
    monkeypatch.setenv("STT_PROVIDER", "OpenAI")
    monkeypatch.setenv("STT_MODEL", "gpt-4o-transcribe")
    monkeypatch.setenv("STT_LANGUAGE", "en")
    monkeypatch.setenv("DISCORD_GUILD_ID", "12345")

    settings = Settings.from_env()

    assert settings.stt_provider == "openai"
    assert settings.stt_model == "gpt-4o-transcribe"
    assert settings.stt_language == "en"
    assert settings.discord_guild_id == 12345


def test_missing_token_is_rejected(monkeypatch):
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)

    with pytest.raises(ValueError, match="DISCORD_BOT_TOKEN is required"):
        Settings.from_env()
