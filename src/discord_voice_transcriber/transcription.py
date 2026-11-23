"""Speech-to-text providers with automatic temporary-audio cleanup."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import tempfile
from pathlib import Path
from typing import Any

from .receiver import VoiceReceiver

# Model parity with Hermes Agent's STT configuration: ``stt.local.model`` defaults to
# ``base`` (faster-whisper) and ``stt.openai.model`` to ``whisper-1`` -- the same value as
# Hermes' ``DEFAULT_STT_MODEL`` constant in ``tools/transcription_common.py``.
HERMES_DEFAULT_MODELS = {"local": "base", "openai": "whisper-1"}
SUPPORTED_PROVIDERS = frozenset(HERMES_DEFAULT_MODELS)


def _local_backend_available() -> bool:
    """True when faster-whisper can be imported, matching Hermes' local-backend probe."""
    try:
        return importlib.util.find_spec("faster_whisper") is not None
    except (ImportError, ValueError):
        return False


def resolve_provider(provider: str | None = None) -> str:
    """Resolve an STT provider the way Hermes does.

    An explicit provider is honoured with no silent cloud fallback; otherwise the
    backend is auto-detected local-first (Hermes order: ``local`` before the cloud
    providers it supports).
    """
    choice = (provider or "auto").strip().lower()
    if choice and choice != "auto":
        if choice not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported STT provider: {provider}")
        return choice
    if _local_backend_available():
        return "local"
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    raise ValueError(
        "No STT backend available: install the local backend with "
        "`uv sync --extra local`, or set OPENAI_API_KEY, or pin STT_PROVIDER explicitly."
    )


class Transcriber:
    def __init__(
        self,
        provider: str = "auto",
        *,
        model: str | None = None,
        language: str | None = None,
        openai_client: Any | None = None,
        temp_dir: str | Path | None = None,
    ) -> None:
        self.provider = resolve_provider(provider)
        self.model = model or HERMES_DEFAULT_MODELS[self.provider]
        self.language = language
        self.temp_dir = Path(temp_dir) if temp_dir else None
        self._openai_client = openai_client
        self._local_model: Any | None = None

    async def transcribe_pcm(self, pcm_data: bytes, *, user_id: int) -> str:
        directory = str(self.temp_dir) if self.temp_dir else None
        fd, filename = tempfile.mkstemp(prefix=f"speaker-{user_id}-", suffix=".wav", dir=directory)
        os.close(fd)
        path = Path(filename)
        try:
            VoiceReceiver.pcm_to_wav(pcm_data, path)
            if self.provider == "openai":
                return await self._transcribe_openai(path)
            return await asyncio.to_thread(self._transcribe_local, path)
        finally:
            path.unlink(missing_ok=True)

    async def _transcribe_openai(self, path: Path) -> str:
        if self._openai_client is None:
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI()
        kwargs: dict[str, Any] = {"model": self.model}
        if self.language:
            kwargs["language"] = self.language
        with path.open("rb") as audio:
            result = await self._openai_client.audio.transcriptions.create(file=audio, **kwargs)
        text = result if isinstance(result, str) else getattr(result, "text", "")
        return str(text).strip()

    def _transcribe_local(self, path: Path) -> str:
        if self._local_model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as error:
                raise RuntimeError(
                    "Local STT requires: uv sync --extra local"
                ) from error
            self._local_model = WhisperModel(self.model, device="auto", compute_type="auto")
        options = {"vad_filter": True}
        if self.language:
            options["language"] = self.language
        segments, _ = self._local_model.transcribe(str(path), **options)
        return " ".join(segment.text.strip() for segment in segments).strip()
