from __future__ import annotations

from pathlib import Path

import pytest

from discord_voice_transcriber import transcription
from discord_voice_transcriber.transcription import (
    HERMES_DEFAULT_MODELS,
    Transcriber,
    resolve_provider,
)


class FakeTranscriptions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return "hello from the channel"


class FakeAudio:
    def __init__(self):
        self.transcriptions = FakeTranscriptions()


class FakeClient:
    def __init__(self):
        self.audio = FakeAudio()


@pytest.mark.asyncio
async def test_openai_transcriber_returns_text_and_removes_temp_audio(tmp_path: Path):
    client = FakeClient()
    transcriber = Transcriber(provider="openai", openai_client=client, temp_dir=tmp_path)

    text = await transcriber.transcribe_pcm(b"\x00\x00" * 48000 * 2, user_id=42)

    assert text == "hello from the channel"
    assert client.audio.transcriptions.calls[0]["model"] == "whisper-1"
    assert list(tmp_path.iterdir()) == []


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unsupported STT provider"):
        Transcriber(provider="mystery")


def test_default_models_match_hermes_stt_catalog():
    """Hermes pins stt.openai.model=whisper-1 and stt.local.model=base."""
    assert HERMES_DEFAULT_MODELS == {"local": "base", "openai": "whisper-1"}


def test_openai_default_model_matches_hermes():
    assert Transcriber(provider="openai", openai_client=FakeClient()).model == "whisper-1"


def test_local_default_model_matches_hermes():
    assert Transcriber(provider="local").model == "base"


def test_explicit_model_overrides_hermes_default():
    transcriber = Transcriber(provider="openai", model="gpt-4o-transcribe", openai_client=FakeClient())
    assert transcriber.model == "gpt-4o-transcribe"


def test_auto_provider_prefers_local_like_hermes(monkeypatch):
    """Hermes auto-detects local first, then cloud providers."""
    monkeypatch.setattr(transcription, "_local_backend_available", lambda: True)
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")

    assert resolve_provider("auto") == "local"
    assert Transcriber(provider="auto").model == "base"


def test_auto_provider_falls_back_to_openai_without_local_backend(monkeypatch):
    monkeypatch.setattr(transcription, "_local_backend_available", lambda: False)
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")

    assert resolve_provider("auto") == "openai"
    assert Transcriber(provider="auto", openai_client=FakeClient()).model == "whisper-1"


def test_auto_provider_without_any_backend_raises(monkeypatch):
    monkeypatch.setattr(transcription, "_local_backend_available", lambda: False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="No STT backend available"):
        resolve_provider("auto")


def test_explicit_provider_is_honoured_over_autodetect(monkeypatch):
    monkeypatch.setattr(transcription, "_local_backend_available", lambda: True)

    assert resolve_provider("openai") == "openai"
