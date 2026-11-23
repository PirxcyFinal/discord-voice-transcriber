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

# Whisper models always operate on 16 kHz mono audio.
WHISPER_SAMPLE_RATE = 16_000


def pcm_to_whisper_audio(
    pcm_data: bytes,
    *,
    sample_rate: int = 48_000,
    channels: int = 2,
):
    """Convert interleaved 16-bit PCM into a mono float32 array at 16 kHz.

    This deliberately replaces ``faster_whisper.audio.decode_audio``. That helper calls
    ``av.open(..., metadata_errors=...)``, a keyword PyAV removed in v15, while
    faster-whisper still only declares ``av>=11``; pinning av backwards is not an option
    because older releases ship no wheels for current Python versions. Decoding here keeps
    the local provider working regardless of the installed av version -- and is cheaper,
    since the receiver already hands us raw PCM.
    """
    import numpy as np

    frame_width = 2 * channels
    usable = len(pcm_data) - (len(pcm_data) % frame_width)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)

    interleaved = np.frombuffer(pcm_data[:usable], dtype="<i2")
    # Downmix to mono in float32 so the channel average cannot overflow.
    mono = interleaved.reshape(-1, channels).astype(np.float32).mean(axis=1) / 32768.0

    if sample_rate != WHISPER_SAMPLE_RATE and mono.size:
        # Low-pass before decimating, otherwise content above the new Nyquist
        # frequency aliases back into the speech band and hurts accuracy.
        decimation = sample_rate / WHISPER_SAMPLE_RATE
        taps = int(max(4.0, decimation * 8.0)) | 1
        window = np.hamming(taps).astype(np.float32)
        cutoff = 0.5 / decimation
        offsets = np.arange(taps, dtype=np.float32) - (taps - 1) / 2.0
        kernel = window * 2.0 * cutoff * np.sinc(2.0 * cutoff * offsets)
        kernel /= kernel.sum()
        filtered = np.convolve(mono, kernel, mode="same")

        target = round(mono.size / decimation)
        positions = np.arange(target, dtype=np.float32) * decimation
        mono = np.interp(positions, np.arange(filtered.size, dtype=np.float32), filtered)

    return np.clip(mono, -1.0, 1.0).astype(np.float32)


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
        if self.provider == "local":
            # No temporary file needed: decode in-process and feed whisper directly.
            return await asyncio.to_thread(self._transcribe_local, pcm_data)
        directory = str(self.temp_dir) if self.temp_dir else None
        fd, filename = tempfile.mkstemp(prefix=f"speaker-{user_id}-", suffix=".wav", dir=directory)
        os.close(fd)
        path = Path(filename)
        try:
            VoiceReceiver.pcm_to_wav(pcm_data, path)
            return await self._transcribe_openai(path)
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

    def _transcribe_local(self, pcm_data: bytes) -> str:
        if self._local_model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as error:
                raise RuntimeError(
                    "Local STT requires: uv sync --extra local"
                ) from error
            self._local_model = WhisperModel(self.model, device="auto", compute_type="auto")
        audio = pcm_to_whisper_audio(pcm_data)
        if not audio.size:
            return ""
        options = {"vad_filter": True}
        if self.language:
            options["language"] = self.language
        segments, _ = self._local_model.transcribe(audio, **options)
        return " ".join(segment.text.strip() for segment in segments).strip()
