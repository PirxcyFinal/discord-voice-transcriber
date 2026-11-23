from __future__ import annotations

from pathlib import Path

import pytest

from discord_voice_transcriber.transcription import Transcriber


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
    assert client.audio.transcriptions.calls[0]["model"] == "gpt-4o-mini-transcribe"
    assert list(tmp_path.iterdir()) == []


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unsupported STT provider"):
        Transcriber(provider="mystery")
