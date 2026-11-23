# Discord Voice Transcriber

A focused Discord bot that joins the voice channel occupied by the person who
runs `/listen`, receives each participant as a separate audio stream, transcribes
speech, and posts speaker-attributed lines to the command's text channel.

The Discord RTP/DAVE/Opus receiving path is adapted from
[Hermes Agent](https://github.com/NousResearch/hermes-agent). See
[`NOTICE.md`](NOTICE.md) for provenance.

## What it does

- `/listen acknowledge_consent:true` joins **your current voice channel**.
- Maps Discord RTP SSRCs to user IDs.
- Handles Discord transport encryption, DAVE transitions, and Opus decoding.
- Buffers audio separately for every speaker; tracks are never mixed.
- Splits speech after 1.5 seconds of silence.
- Uses OpenAI transcription by default, or optional local `faster-whisper`.
- Posts transcript lines as `**Display name:** what they said`.
- `/leave` flushes pending utterances, disconnects, and cleans up.
- Deletes temporary WAV files after each transcription. Raw audio is not retained.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- Discord bot token
- `libopus` (`brew install opus` on macOS; your distribution's Opus package on Linux)
- OpenAI API key for the default provider, **or** the optional local model extra

## Discord application setup

1. Create an application and bot in the
   [Discord Developer Portal](https://discord.com/developers/applications).
2. Under OAuth2 URL Generator, select scopes:
   - `bot`
   - `applications.commands`
3. Grant only these bot permissions:
   - View Channels
   - Connect
   - Send Messages
   - Use Application Commands
4. Invite the bot to the server.

The bot uses the `Guilds` and `Voice States` intents. It does not require the
privileged Message Content intent.

## Install

```bash
git clone <repository-url>
cd discord-voice-transcriber
uv sync
cp .env.example .env
```

Edit `.env`:

```dotenv
DISCORD_BOT_TOKEN=your-discord-bot-token
OPENAI_API_KEY=your-openai-api-key
STT_PROVIDER=openai
# Optional: faster slash-command sync while developing
DISCORD_GUILD_ID=your-server-id
```

Validate without connecting to Discord:

```bash
uv run discord-voice-transcriber --check
```

Start the bot:

```bash
uv run discord-voice-transcriber
```

Then join a voice channel and run:

```text
/listen acknowledge_consent:true
```

Stop with:

```text
/leave
```

Use `/voice_status` to check whether the bot is active.

## Local transcription

Install the optional model dependency:

```bash
uv sync --extra local
```

Set:

```dotenv
STT_PROVIDER=local
STT_MODEL=small
```

The first run downloads the selected faster-whisper model. Local inference can
be CPU-intensive. OpenAI is the lighter default for the bot host.

## Privacy and consent

Voice transcription affects everyone in the channel. Obtain consent from all
participants and comply with applicable recording and privacy laws.

The bot requires the explicit slash-command argument
`acknowledge_consent:true`, announces that transcription started in the text
channel, and posts all transcripts there. This is a safety reminder, not a
substitute for consent.

Audio handling:

- PCM remains in memory per speaker until an utterance is segmented.
- Each utterance is written to a temporary WAV for STT.
- The WAV is deleted in a `finally` block after success or failure.
- No raw audio archive or database is included.
- Transcript messages remain in Discord according to that server's retention.
- With OpenAI STT, audio is sent to OpenAI. Use the local provider to avoid that.

## Configuration

- `DISCORD_BOT_TOKEN` — required Discord bot token.
- `DISCORD_GUILD_ID` — optional development server for immediate command sync.
- `STT_PROVIDER` — `openai` (default) or `local`.
- `STT_MODEL` — optional model override. Defaults to
  `gpt-4o-mini-transcribe` or local `small`.
- `STT_LANGUAGE` — optional ISO-639-1 language code such as `en`.
- `OPENAI_API_KEY` — required only for OpenAI STT.

Secrets belong in `.env`, which is gitignored. Never commit tokens or API keys.

## Development

```bash
uv sync --extra test
uv run pytest
uvx ruff check .
uv build
```

Project layout:

```text
src/discord_voice_transcriber/
  bot.py            Discord commands and session lifecycle
  receiver.py       RTP, encryption, DAVE, Opus, per-speaker buffers
  transcription.py Temporary WAV and STT providers
  config.py         Environment configuration
  __main__.py       CLI entry point
```

## Known boundaries

- Discord's receive path is not a high-level public `discord.py` API. This bot
  intentionally uses connection/socket internals, as Hermes Agent does, so new
  Discord or `discord.py` voice protocol changes may require maintenance.
- Speaker identity depends on Discord's SSRC speaking events. An unmapped DAVE
  stream is dropped rather than risk decrypting or attributing it incorrectly.
- One active transcription session is allowed per server.

## License

MIT. See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).
