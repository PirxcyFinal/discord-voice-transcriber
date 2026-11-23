from __future__ import annotations

from types import SimpleNamespace

import pytest

from discord_voice_transcriber.bot import VoiceTranscriberBot


@pytest.mark.asyncio
async def test_voice_channel_is_taken_from_invoking_member():
    bot = VoiceTranscriberBot.__new__(VoiceTranscriberBot)
    channel = object()
    interaction = SimpleNamespace(user=SimpleNamespace(voice=SimpleNamespace(channel=channel)))

    assert bot._voice_channel_for(interaction) is channel


@pytest.mark.asyncio
async def test_user_outside_voice_channel_is_rejected():
    bot = VoiceTranscriberBot.__new__(VoiceTranscriberBot)
    interaction = SimpleNamespace(user=SimpleNamespace(voice=None))

    with pytest.raises(ValueError, match="voice channel"):
        bot._voice_channel_for(interaction)


def test_transcript_line_includes_speaker():
    bot = VoiceTranscriberBot.__new__(VoiceTranscriberBot)
    member = SimpleNamespace(display_name="Ada")

    assert bot._format_transcript(member, "Testing one two") == "**Ada:** Testing one two"
