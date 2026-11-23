"""Focused Discord bot: join, receive per-speaker audio, transcribe, leave."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import discord
from discord import app_commands

from .receiver import VoiceReceiver
from .transcription import Transcriber

logger = logging.getLogger(__name__)


@dataclass
class VoiceSession:
    voice_client: discord.VoiceClient
    receiver: VoiceReceiver
    text_channel: Any
    poll_task: asyncio.Task | None = None
    transcript_tasks: set[asyncio.Task] = field(default_factory=set)


class VoiceTranscriberBot(discord.Client):
    def __init__(self, transcriber: Transcriber, *, guild_id: int | None = None):
        intents = discord.Intents.default()
        intents.guilds = True
        intents.voice_states = True
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.transcriber = transcriber
        self.guild_id = guild_id
        self.sessions: dict[int, VoiceSession] = {}
        self.tree.command(name="listen", description="Join your voice channel and transcribe speakers")(
            self.listen
        )
        self.tree.command(name="leave", description="Stop transcription and leave voice")(
            self.leave
        )
        self.tree.command(name="voice_status", description="Show voice transcription status")(
            self.voice_status
        )

    async def setup_hook(self) -> None:
        if self.guild_id:
            guild = discord.Object(id=self.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        else:
            await self.tree.sync()

    def _voice_channel_for(self, interaction: discord.Interaction) -> Any:
        voice_state = getattr(interaction.user, "voice", None)
        channel = getattr(voice_state, "channel", None)
        if channel is None:
            raise ValueError("Join a voice channel before using /listen.")
        return channel

    @staticmethod
    def _format_transcript(member: Any, text: str) -> str:
        return f"**{member.display_name}:** {text}"

    async def listen(self, interaction: discord.Interaction, acknowledge_consent: bool) -> None:
        if interaction.guild is None:
            await interaction.response.send_message("Use this command in a server.", ephemeral=True)
            return
        if not acknowledge_consent:
            await interaction.response.send_message(
                "I won't record silently. Confirm everyone knows transcription is active, then run "
                "`/listen acknowledge_consent:true`.",
                ephemeral=True,
            )
            return
        try:
            voice_channel = self._voice_channel_for(interaction)
        except ValueError as error:
            await interaction.response.send_message(str(error), ephemeral=True)
            return

        guild_id = interaction.guild.id
        if guild_id in self.sessions:
            await interaction.response.send_message(
                "I'm already transcribing a voice channel here. Use `/leave` first.", ephemeral=True
            )
            return

        await interaction.response.defer(ephemeral=True, thinking=True)
        voice_client = await voice_channel.connect(self_deaf=False)
        receiver = VoiceReceiver(voice_client)
        session = VoiceSession(voice_client, receiver, interaction.channel)
        self.sessions[guild_id] = session
        try:
            receiver.start()
            session.poll_task = asyncio.create_task(
                self._poll_session(interaction.guild, session),
                name=f"voice-transcriber-{guild_id}",
            )
        except Exception:
            self.sessions.pop(guild_id, None)
            await voice_client.disconnect(force=True)
            raise

        await interaction.followup.send(
            f"Joined **{voice_channel.name}**. Transcription is active and raw audio is not retained.",
            ephemeral=True,
        )
        if interaction.channel:
            await interaction.channel.send(
                f"Voice transcription started in **{voice_channel.name}** by {interaction.user.mention}. "
                "Use `/leave` to stop it."
            )

    async def leave(self, interaction: discord.Interaction) -> None:
        guild_id = interaction.guild.id if interaction.guild else 0
        session = self.sessions.pop(guild_id, None)
        if session is None:
            await interaction.response.send_message("I'm not transcribing voice here.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        await self._stop_session(interaction.guild, session)
        await interaction.followup.send("Transcription stopped and I left the voice channel.", ephemeral=True)

    async def voice_status(self, interaction: discord.Interaction) -> None:
        guild_id = interaction.guild.id if interaction.guild else 0
        session = self.sessions.get(guild_id)
        if session is None:
            message = "Voice transcription is stopped."
        else:
            message = f"Transcribing **{session.voice_client.channel.name}** with per-speaker tracks."
        await interaction.response.send_message(message, ephemeral=True)

    async def _poll_session(self, guild: discord.Guild, session: VoiceSession) -> None:
        try:
            while session.receiver.running:
                await asyncio.sleep(0.25)
                for user_id, pcm in session.receiver.check_silence():
                    task = asyncio.create_task(self._publish_utterance(guild, session, user_id, pcm))
                    session.transcript_tasks.add(task)
                    task.add_done_callback(session.transcript_tasks.discard)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Voice polling failed in guild %s", guild.id)

    async def _publish_utterance(
        self, guild: discord.Guild, session: VoiceSession, user_id: int, pcm: bytes
    ) -> None:
        try:
            text = await self.transcriber.transcribe_pcm(pcm, user_id=user_id)
            if not text:
                return
            member = guild.get_member(user_id)
            speaker = member or type("Speaker", (), {"display_name": f"User {user_id}"})()
            await session.text_channel.send(self._format_transcript(speaker, text)[:2000])
        except Exception:
            logger.exception("Transcription failed for user %s", user_id)

    async def _stop_session(self, guild: discord.Guild | None, session: VoiceSession) -> None:
        pending = session.receiver.flush_pending()
        session.receiver.stop()
        if session.poll_task:
            session.poll_task.cancel()
            await asyncio.gather(session.poll_task, return_exceptions=True)
        if guild:
            for user_id, pcm in pending:
                task = asyncio.create_task(self._publish_utterance(guild, session, user_id, pcm))
                session.transcript_tasks.add(task)
        if session.transcript_tasks:
            await asyncio.gather(*session.transcript_tasks, return_exceptions=True)
        await session.voice_client.disconnect(force=True)

    async def on_voice_state_update(
        self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState
    ) -> None:
        session = self.sessions.get(member.guild.id)
        if session and (before.channel == session.voice_client.channel or after.channel == session.voice_client.channel):
            session.receiver.refresh_credentials("voice-state membership change")

    async def close(self) -> None:
        sessions = list(self.sessions.items())
        self.sessions.clear()
        for guild_id, session in sessions:
            guild = self.get_guild(guild_id)
            await self._stop_session(guild, session)
        await super().close()
