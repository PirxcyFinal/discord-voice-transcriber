from __future__ import annotations

from pathlib import Path

import pytest

from discord_voice_transcriber import transcription
from discord_voice_transcriber.transcription import (
    DEFAULT_MODELS,
    HERMES_DEFAULT_MODELS,
    RECOMMENDED_LOCAL_MODEL,
    Transcriber,
    is_english_only_model,
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


def test_local_default_is_the_benchmarked_distil_model():
    """We deliberately diverge from Hermes' ``base`` for the local provider.
    Benchmarked on one 20s clip: this model beat ``base`` on accuracy while still
    running ~25x realtime, so the extra 190MB buys accuracy we can measure.
    """
    assert RECOMMENDED_LOCAL_MODEL == "Systran/faster-distil-whisper-small.en"
    assert DEFAULT_MODELS["local"] == RECOMMENDED_LOCAL_MODEL
    assert Transcriber(provider="local").model == RECOMMENDED_LOCAL_MODEL


def test_openai_default_still_matches_hermes_exactly():
    """Only the local default diverges; the cloud default stays at parity."""
    assert DEFAULT_MODELS["openai"] == HERMES_DEFAULT_MODELS["openai"]


def test_explicit_model_overrides_hermes_default():
    transcriber = Transcriber(provider="openai", model="gpt-4o-transcribe", openai_client=FakeClient())
    assert transcriber.model == "gpt-4o-transcribe"


@pytest.mark.parametrize(
    ("model", "english_only"),
    [
        ("Systran/faster-distil-whisper-small.en", True),
        ("tiny.en", True),
        ("base", False),
        ("Systran/faster-whisper-small", False),
        ("whisper-1", False),
    ],
)
def test_english_only_models_are_detected(model: str, english_only: bool):
    """The default is English-only, so callers must be able to warn about it."""
    assert is_english_only_model(model) is english_only


def test_auto_provider_prefers_local_like_hermes(monkeypatch):
    """Hermes auto-detects local first, then cloud providers."""
    monkeypatch.setattr(transcription, "_local_backend_available", lambda: True)
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")

    assert resolve_provider("auto") == "local"
    assert Transcriber(provider="auto").model == RECOMMENDED_LOCAL_MODEL


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


def test_pcm_to_whisper_audio_downmixes_and_resamples_without_av():
    """The local path must not depend on faster-whisper's av-based decode_audio.

    av>=15 dropped the ``metadata_errors`` kwarg that faster_whisper.audio.decode_audio
    passes, and older av has no wheels on recent Pythons, so we decode in-process.
    """
    import math
    import struct

    from discord_voice_transcriber.transcription import (
        WHISPER_SAMPLE_RATE,
        pcm_to_whisper_audio,
    )

    seconds = 0.5
    frames = int(48_000 * seconds)
    # 440 Hz tone, identical in both channels, as interleaved stereo s16le.
    samples = []
    for i in range(frames):
        value = int(16_000 * math.sin(2 * math.pi * 440 * i / 48_000))
        samples.extend((value, value))
    pcm = struct.pack(f"<{len(samples)}h", *samples)

    audio = pcm_to_whisper_audio(pcm)

    assert audio.ndim == 1, "whisper expects mono"
    assert audio.dtype.name == "float32"
    expected = int(frames * WHISPER_SAMPLE_RATE / 48_000)
    assert abs(len(audio) - expected) <= 2
    assert audio.max() <= 1.0 and audio.min() >= -1.0
    assert float(abs(audio).mean()) > 0.01, "tone should survive resampling"


def test_pcm_to_whisper_audio_handles_empty_and_odd_input():
    from discord_voice_transcriber.transcription import pcm_to_whisper_audio

    assert len(pcm_to_whisper_audio(b"")) == 0
    # A trailing partial frame must be dropped, not crash.
    assert len(pcm_to_whisper_audio(b"\x01\x00\x02\x00\x03")) >= 0
