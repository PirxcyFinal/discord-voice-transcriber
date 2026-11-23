"""Speech-to-text providers with automatic temporary-audio cleanup."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any

from .receiver import VoiceReceiver


class Transcriber:
    def __init__(
        self,
        provider: str = "openai",
        *,
        model: str | None = None,
        language: str | None = None,
        openai_client: Any | None = None,
        temp_dir: str | Path | None = None,
    ) -> None:
        self.provider = provider.strip().lower()
        if self.provider not in {"openai", "local"}:
            raise ValueError(f"Unsupported STT provider: {provider}")
        self.model = model or (
            "gpt-4o-mini-transcribe" if self.provider == "openai" else "small"
        )
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
