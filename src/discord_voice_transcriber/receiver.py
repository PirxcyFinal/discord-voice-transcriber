"""Discord RTP receiver adapted from Hermes Agent's Discord adapter.

The receiver hooks discord.py's voice socket, maps RTP SSRC values to Discord
user IDs, decrypts transport/DAVE payloads, and buffers decoded PCM separately
for every speaker.
"""

from __future__ import annotations

import ctypes.util
import logging
import os
import struct
import sys
import threading
import time
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

import discord
from nacl.exceptions import CryptoError

logger = logging.getLogger(__name__)

_OPUS_PATHS = (
    "/opt/homebrew/lib/libopus.dylib",
    "/opt/homebrew/opt/opus/lib/libopus.dylib",
    "/usr/local/lib/libopus.dylib",
    "/usr/lib/x86_64-linux-gnu/libopus.so.0",
    "/usr/lib/aarch64-linux-gnu/libopus.so.0",
)


def load_opus_codec() -> None:
    """Load libopus, including Homebrew paths missed by find_library on macOS."""
    if discord.opus.is_loaded():
        return
    candidates = [path for path in (ctypes.util.find_library("opus"), *_OPUS_PATHS) if path]
    for path in candidates:
        if os.path.isabs(path) and not os.path.isfile(path):
            continue
        try:
            discord.opus.load_opus(path)
        except Exception:
            logger.debug("Could not load Opus from %s", path, exc_info=True)
        if discord.opus.is_loaded():
            return
    hint = " Install it with `brew install opus`." if sys.platform == "darwin" else " Install libopus."
    raise RuntimeError("Discord voice reception requires the Opus codec." + hint)


class VoiceReceiver:
    """Capture and segment Discord voice audio while preserving speaker IDs."""

    SILENCE_THRESHOLD = 1.5
    MIN_SPEECH_DURATION = 0.5
    SAMPLE_RATE = 48_000
    CHANNELS = 2
    REKEY_FAILURE_STREAK = 25

    def __init__(self, voice_client: discord.VoiceClient, allowed_user_ids: set[str] | None = None):
        self._vc = voice_client
        self._allowed_user_ids = allowed_user_ids or set()
        self._running = False
        self._creds: tuple[bytes, Any, int, bool] = (b"", None, 0, False)
        self._bot_ssrc = 0
        self._dave_passthrough_until = 0.0
        self._ssrc_to_user: dict[int, int] = {}
        self._lock = threading.Lock()
        self._buffers: dict[int, bytearray] = defaultdict(bytearray)
        self._last_packet_time: dict[int, float] = {}
        self._decoders: dict[int, Any] = {}
        self._paused = False
        self._packet_debug_count = 0
        self._decode_ok = 0
        self._decode_failed = 0
        self._dave_unmapped_dropped = 0
        self._nacl_fail_streak = 0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def _dave_downgraded(self) -> bool:
        return self._creds[3]

    @property
    def _dave_protocol_version(self) -> int:
        return self._creds[2]

    def start(self) -> None:
        load_opus_codec()
        conn = self._vc._connection
        self._resolve_credentials(conn)
        self._install_speaking_hook(conn)
        conn.add_socket_listener(self._on_packet)
        self._running = True
        logger.info("Voice receiver started (bot SSRC %d)", self._bot_ssrc)

    def stop(self) -> None:
        self._running = False
        try:
            self._vc._connection.remove_socket_listener(self._on_packet)
        except Exception:
            logger.debug("Voice socket listener was already gone", exc_info=True)
        with self._lock:
            self._buffers.clear()
            self._last_packet_time.clear()
            self._decoders.clear()
            self._ssrc_to_user.clear()
        logger.info(
            "Voice receiver stopped (frames=%d, decrypt failures=%d, unmapped DAVE=%d)",
            self._decode_ok,
            self._decode_failed,
            self._dave_unmapped_dropped,
        )

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def _resolve_credentials(self, conn: Any) -> None:
        self._creds = (
            bytes(conn.secret_key),
            conn.dave_session,
            int(getattr(conn, "dave_protocol_version", 0) or 0),
            bool(getattr(conn, "dave_downgraded", False)),
        )
        self._bot_ssrc = conn.ssrc
        pending = getattr(conn, "dave_pending_transitions", None) or {}
        if 0 in pending.values() and time.monotonic() >= self._dave_passthrough_until:
            self.note_dave_passthrough_window(120.0)

    def refresh_credentials(self, reason: str) -> None:
        if not self._running:
            return
        try:
            self._resolve_credentials(self._vc._connection)
            self._nacl_fail_streak = 0
            logger.info("Voice credentials refreshed: %s", reason)
        except Exception:
            logger.warning("Voice credential refresh failed: %s", reason, exc_info=True)

    def note_dave_passthrough_window(self, seconds: float) -> None:
        self._dave_passthrough_until = time.monotonic() + seconds

    def map_ssrc(self, ssrc: int, user_id: int) -> None:
        with self._lock:
            self._ssrc_to_user[ssrc] = user_id

    def _install_speaking_hook(self, conn: Any) -> None:
        original_hook = conn.hook

        async def wrapped_hook(ws: Any, msg: Any) -> None:
            if isinstance(msg, dict) and msg.get("op") == 5:
                data = msg.get("d", {})
                if data.get("ssrc") and data.get("user_id"):
                    self.map_ssrc(int(data["ssrc"]), int(data["user_id"]))
            if original_hook:
                await original_hook(ws, msg)
            if not isinstance(msg, dict):
                return
            op = msg.get("op")
            if op in (4, 24):
                self.refresh_credentials(f"voice websocket op {op}")
            elif op == 21 and (msg.get("d") or {}).get("protocol_version") == 0:
                self.note_dave_passthrough_window(120.0)
            elif op == 22:
                was_downgraded = self._dave_downgraded
                self.refresh_credentials("voice websocket op 22")
                if was_downgraded and not self._dave_downgraded and self._dave_protocol_version > 0:
                    self.note_dave_passthrough_window(10.0)

        conn.hook = wrapped_hook
        try:
            from discord.utils import MISSING

            if hasattr(conn, "ws") and conn.ws is not MISSING:
                conn.ws._hook = wrapped_hook
        except Exception:
            logger.warning("Could not install speaking hook on live websocket", exc_info=True)

    def _on_packet(self, data: bytes) -> None:
        if not self._running or self._paused or len(data) < 16:
            return
        secret_key, dave_session, dave_version, _ = self._creds
        self._packet_debug_count += 1
        if (data[0] >> 6) != 2 or (data[1] & 0x7F) != 0x78:
            return

        first_byte = data[0]
        _, _, _seq, _timestamp, ssrc = struct.unpack_from(">BBHII", data, 0)
        if ssrc == self._bot_ssrc:
            return

        csrc_count = first_byte & 0x0F
        has_extension = bool(first_byte & 0x10)
        has_padding = bool(first_byte & 0x20)
        header_size = 12 + (4 * csrc_count) + (4 if has_extension else 0)
        if len(data) < header_size + 4:
            return
        extension_length = 0
        if has_extension:
            offset = 12 + (4 * csrc_count)
            extension_length = struct.unpack_from(">H", data, offset + 2)[0] * 4

        header = bytes(data[:header_size])
        payload_with_nonce = data[header_size:]
        nonce = bytearray(24)
        nonce[:4] = payload_with_nonce[-4:]
        encrypted = bytes(payload_with_nonce[:-4])
        try:
            import nacl.secret

            decrypted = nacl.secret.Aead(secret_key).decrypt(encrypted, header, bytes(nonce))
            self._nacl_fail_streak = 0
        except (CryptoError, ValueError):
            self._decode_failed += 1
            self._nacl_fail_streak += 1
            if self._nacl_fail_streak >= self.REKEY_FAILURE_STREAK:
                self.refresh_credentials("decrypt-failure streak")
            return

        if extension_length and len(decrypted) > extension_length:
            decrypted = decrypted[extension_length:]
        if has_padding:
            if not decrypted:
                return
            padding = decrypted[-1]
            if padding == 0 or padding > len(decrypted):
                return
            decrypted = decrypted[:-padding]
            if not decrypted:
                return

        if dave_session:
            with self._lock:
                user_id = self._ssrc_to_user.get(ssrc, 0) or self._infer_user_for_ssrc(ssrc)
            if user_id:
                try:
                    import davey

                    decrypted = dave_session.decrypt(user_id, davey.MediaType.audio, decrypted)
                except Exception as error:  # noqa: BLE001 - davey exposes native exception types
                    if "Unencrypted" not in str(error):
                        return
            elif dave_version > 0 and time.monotonic() >= self._dave_passthrough_until:
                self._dave_unmapped_dropped += 1
                return

        try:
            if ssrc not in self._decoders:
                self._decoders[ssrc] = discord.opus.Decoder()
            pcm = self._decoders[ssrc].decode(decrypted)
            self._decode_ok += 1
            with self._lock:
                self._buffers[ssrc].extend(pcm)
                self._last_packet_time[ssrc] = time.monotonic()
        except Exception:
            with self._lock:
                self._decoders.pop(ssrc, None)
            logger.debug("Opus decode failed for SSRC %s", ssrc, exc_info=True)

    def _infer_user_for_ssrc(self, ssrc: int) -> int:
        try:
            channel = self._vc.channel
            bot_id = self._vc.user.id if self._vc.user else 0
            candidates = [
                member.id
                for member in channel.members
                if member.id != bot_id
                and (not self._allowed_user_ids or str(member.id) in self._allowed_user_ids)
            ]
            if len(candidates) == 1:
                self._ssrc_to_user[ssrc] = candidates[0]
                return candidates[0]
        except (AttributeError, TypeError):
            logger.debug("Could not infer a user for SSRC %s", ssrc, exc_info=True)
        return 0

    def check_silence(self) -> list[tuple[int, bytes]]:
        now = time.monotonic()
        completed: list[tuple[int, bytes]] = []
        with self._lock:
            mapping = dict(self._ssrc_to_user)
            for ssrc in list(self._buffers):
                silence = now - self._last_packet_time.get(ssrc, now)
                buffer = self._buffers[ssrc]
                duration = len(buffer) / (self.SAMPLE_RATE * self.CHANNELS * 2)
                if silence >= self.SILENCE_THRESHOLD and duration >= self.MIN_SPEECH_DURATION:
                    user_id = mapping.get(ssrc, 0) or self._infer_user_for_ssrc(ssrc)
                    if user_id:
                        completed.append((user_id, bytes(buffer)))
                    self._buffers[ssrc] = bytearray()
                    self._last_packet_time.pop(ssrc, None)
                elif silence >= self.SILENCE_THRESHOLD * 2:
                    self._buffers.pop(ssrc, None)
                    self._last_packet_time.pop(ssrc, None)
        return completed

    def flush_pending(self) -> list[tuple[int, bytes]]:
        completed: list[tuple[int, bytes]] = []
        with self._lock:
            mapping = dict(self._ssrc_to_user)
            for ssrc, buffer in list(self._buffers.items()):
                duration = len(buffer) / (self.SAMPLE_RATE * self.CHANNELS * 2)
                if duration >= self.MIN_SPEECH_DURATION:
                    user_id = mapping.get(ssrc, 0) or self._infer_user_for_ssrc(ssrc)
                    if user_id:
                        completed.append((user_id, bytes(buffer)))
                self._buffers.pop(ssrc, None)
                self._last_packet_time.pop(ssrc, None)
        return completed

    def discard_pending(self) -> None:
        with self._lock:
            self._buffers.clear()
            self._last_packet_time.clear()

    @classmethod
    def pcm_to_wav(cls, pcm_data: bytes, output_path: str | Path) -> None:
        """Write native Discord PCM as a standards-compliant WAV file."""
        with wave.open(str(output_path), "wb") as output:
            output.setnchannels(cls.CHANNELS)
            output.setsampwidth(2)
            output.setframerate(cls.SAMPLE_RATE)
            output.writeframes(pcm_data)
