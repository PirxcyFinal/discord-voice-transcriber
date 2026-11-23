from __future__ import annotations

import time
from types import SimpleNamespace

from discord_voice_transcriber.receiver import VoiceReceiver, load_opus_codec


class DummyConnection:
    secret_key = bytes(32)
    dave_session = None
    dave_protocol_version = 0
    dave_downgraded = False
    ssrc = 999
    hook = None

    def __init__(self):
        self.dave_pending_transitions = {}

    def add_socket_listener(self, callback):
        self.listener = callback

    def remove_socket_listener(self, callback):
        self.listener = None


class DummyVoiceClient:
    def __init__(self):
        self._connection = DummyConnection()
        self.channel = SimpleNamespace(members=[])
        self.user = SimpleNamespace(id=999)


def pcm(seconds: float) -> bytes:
    return b"\x01\x00" * int(VoiceReceiver.SAMPLE_RATE * VoiceReceiver.CHANNELS * seconds)


def test_opus_loader_finds_homebrew_library(monkeypatch, tmp_path):
    fake_library = tmp_path / "libopus.dylib"
    fake_library.touch()
    loaded = []
    monkeypatch.setattr("discord_voice_transcriber.receiver.ctypes.util.find_library", lambda _name: None)
    monkeypatch.setattr("discord_voice_transcriber.receiver.sys.platform", "darwin")
    monkeypatch.setattr("discord_voice_transcriber.receiver._OPUS_PATHS", (str(fake_library),))
    monkeypatch.setattr("discord_voice_transcriber.receiver.discord.opus.is_loaded", lambda: bool(loaded))
    monkeypatch.setattr("discord_voice_transcriber.receiver.discord.opus.load_opus", loaded.append)

    load_opus_codec()

    assert loaded == [str(fake_library)]


def test_completed_utterances_stay_separated_by_speaker(monkeypatch):
    receiver = VoiceReceiver(DummyVoiceClient())
    receiver.map_ssrc(101, 11)
    receiver.map_ssrc(202, 22)
    now = time.monotonic()
    receiver._buffers[101].extend(pcm(0.75))
    receiver._buffers[202].extend(pcm(1.0))
    receiver._last_packet_time[101] = now - 2.0
    receiver._last_packet_time[202] = now - 2.0

    completed = receiver.check_silence()

    assert [(user_id, len(audio)) for user_id, audio in completed] == [
        (11, len(pcm(0.75))),
        (22, len(pcm(1.0))),
    ]


def test_short_noise_is_discarded_after_extended_silence():
    receiver = VoiceReceiver(DummyVoiceClient())
    receiver.map_ssrc(101, 11)
    receiver._buffers[101].extend(pcm(0.1))
    receiver._last_packet_time[101] = time.monotonic() - 4.0

    assert receiver.check_silence() == []
    assert 101 not in receiver._buffers


def test_flush_pending_preserves_speaker_identity():
    receiver = VoiceReceiver(DummyVoiceClient())
    receiver.map_ssrc(303, 33)
    receiver._buffers[303].extend(pcm(0.6))

    assert receiver.flush_pending() == [(33, pcm(0.6))]
    assert receiver._buffers == {}
