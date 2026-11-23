# Discord Voice Transcriber

A focused Discord bot that joins the voice channel occupied by the person who
runs `/listen`, receives each participant as a separate audio stream, transcribes
speech, and posts speaker-attributed lines to the command's text channel.

The Discord RTP/DAVE/Opus receiving path is adapted from
[Hermes Agent](https://github.com/NousResearch/hermes-agent). See
[`NOTICE.md`](NOTICE.md) for provenance.

## For AI agents and LLM tools

If you are an AI assistant reading, modifying, or reimplementing this project,
read **[`AGENTS.md`](AGENTS.md)** first. It is written for you and covers:

- the full audio pipeline, from RTP packet to posted transcript
- which module owns which responsibility, and the tunable constants
- exactly which keys are required, which are optional, and what "keyless" means
- the pitfalls that have already broken this code, with the real error messages
- a step-by-step guide to building your own bot like this one
- how to verify a change honestly, and what the test suite does *not* cover

Short version of the architecture: Discord sends RTP packets tagged with an
**SSRC**, not a user ID. The SSRC-to-user mapping arrives separately on the
voice websocket (opcode 5), so `receiver.py` hooks that websocket to build the
mapping, then keeps a **separate buffer and Opus decoder per SSRC**. Each
speaker's audio is segmented on silence and transcribed independently, which is
what makes speaker attribution possible. Getting that mapping wrong is the usual
cause of unattributed or misattributed transcripts.

## What it does

- `/listen acknowledge_consent:true` joins **your current voice channel**.
- Maps Discord RTP SSRCs to user IDs.
- Handles Discord transport encryption, DAVE transitions, and Opus decoding.
- Buffers audio separately for every speaker; tracks are never mixed.
- Splits speech after 1.5 seconds of silence.
- Resolves speech-to-text local-first like Hermes Agent: `whisper-1` (OpenAI) or
  a local `faster-whisper` model, defaulting to the benchmarked
  `Systran/faster-distil-whisper-small.en`.
- Posts transcript lines as `**Display name:** what they said`.
- `/leave` flushes pending utterances, disconnects, and cleans up.
- Deletes temporary WAV files after each transcription. Raw audio is not retained.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- Discord bot token
- `libopus` (`brew install opus` on macOS; your distribution's Opus package on Linux)
- An OpenAI API key, **or** the optional local model extra (`faster-whisper`)

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
STT_PROVIDER=auto
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
# Optional; this is the default when STT_MODEL is blank.
STT_MODEL=Systran/faster-distil-whisper-small.en
```

The first run downloads the selected faster-whisper model. Local inference can
be CPU-intensive; OpenAI is lighter on the bot host. With `STT_PROVIDER=auto`
the local backend is preferred whenever `faster-whisper` is installed, which is
the same order Hermes Agent uses.

### Choosing a local model

`STT_MODEL` accepts any faster-whisper size name or Hugging Face CTranslate2
repository id. Measured on this project, on one ~20 s clip of deliberately
awkward speech (file paths, a line number, a Discord user ID, technical jargon),
on Apple Silicon CPU:

| Model | Size | Speed | Accuracy rank |
| --- | --- | --- | --- |
| `Systran/faster-distil-whisper-small.en` **(default)** | 336 MB | 24.9x realtime | **best** |
| `Systran/faster-whisper-small.en` | 486 MB | 18.6x realtime | 2nd |
| `Zoont/faster-whisper-large-v3-turbo-int8-ct2` | 818 MB | 4.9x realtime | 2nd (tied) |
| `base` | 148 MB | 54.2x realtime | 4th |
| `rhasspy/faster-whisper-base-int8` | 80 MB | 51.5x realtime | 4th (tied) |
| `Systran/faster-whisper-tiny.en` | 78 MB | 69.8x realtime | worst |

Read that as a ranking, not as absolute error rates: it is one clip from one
voice, and the scoring penalised correct digit formatting. Reproduce it with
your own audio before trusting the ordering for your channel.

What the numbers are good for:

- **Distillation beats size here.** The 336 MB distilled model was more accurate
  than the 818 MB turbo model while running roughly 5x faster. Reaching for the
  biggest checkpoint is not an upgrade on this workload.
- **`rhasspy/faster-whisper-base-int8` is free compression.** Identical accuracy
  to plain `base` at half the disk; a good pick for small containers.
- **Avoid `tiny.en`.** It mangled the acronyms that matter here (`SSRC`, `DAVE`),
  and `base` is already far faster than live voice requires.

> **The default is English-only.** Models with an `.en` suffix do not reject
> other languages — they transcribe them as confident English nonsense, which is
> much harder to spot than an outright error. For non-English voice channels set
> `STT_MODEL=base` (or another multilingual model). `--check` prints a reminder
> whenever the resolved model is English-only.

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
- `STT_PROVIDER` — `auto` (default), `openai`, or `local`. `auto` resolves
  local-first and then OpenAI, matching Hermes Agent's STT provider detection.
  An explicit value is always honoured with no silent cloud fallback.
- `STT_MODEL` — optional model override. Defaults to `whisper-1` for OpenAI and
  `Systran/faster-distil-whisper-small.en` for local. See
  [Choosing a local model](#choosing-a-local-model).
- `STT_LANGUAGE` — optional ISO-639-1 language code such as `en`.
- `OPENAI_API_KEY` — required only for OpenAI STT.

Secrets belong in `.env`, which is gitignored. Never commit tokens or API keys.

### Relationship to Hermes Agent's defaults

The provider order and the cloud model come from Hermes Agent's own STT
configuration rather than being chosen independently:

- `stt.openai.model` → `whisper-1`, matched exactly (also Hermes'
  `DEFAULT_STT_MODEL` constant in `tools/transcription_common.py`)
- Provider order → explicit selection wins, otherwise local before cloud

**The local model deliberately diverges.** Hermes uses `base`
(`DEFAULT_LOCAL_MODEL`); this project defaults to
`Systran/faster-distil-whisper-small.en` because it measured more accurate at
ample speed for live voice — see the table above. Hermes' value is still
recorded in code as `HERMES_DEFAULT_MODELS` and is one env var away:
`STT_MODEL=base` restores full parity and multilingual support.

Other models Hermes accepts for OpenAI (`gpt-4o-mini-transcribe`,
`gpt-4o-transcribe`) still work via `STT_MODEL` if you prefer them.

### Running keyless (no API key)

Yes, transcription can run with no API key at all:

```bash
uv sync --extra local
DISCORD_BOT_TOKEN=... uv run discord-voice-transcriber
```

With the `local` extra installed, `STT_PROVIDER=auto` resolves to `local` and
runs faster-whisper entirely on your machine. No `OPENAI_API_KEY` is
read, and no audio leaves the host. Model weights are open and downloaded once
from Hugging Face; there is no per-minute cost.

A Discord bot token is still required — that is Discord's own gateway
authentication, not a speech-to-text key. So the bot is keyless for *transcription*,
never for connecting to Discord.

Notes on the local path:

- Audio is decoded in-process from the receiver's raw PCM straight into a 16 kHz
  mono float32 array, so no temporary WAV is written and no API call is made.
- This bypasses `faster_whisper.audio.decode_audio` on purpose. It calls
  `av.open(..., metadata_errors=...)`, a keyword PyAV removed in v15, while
  faster-whisper still only declares `av>=11`. Pinning av backwards is not
  viable because older releases ship no wheels for current Python versions.
- On Apple Silicon, ctranslate2 may suggest its MPS backend for extra speed.

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
  transcription.py  Provider resolution, PCM conversion, STT backends
  config.py         Environment configuration
  __main__.py       CLI entry point
```

Contributor and agent notes, including the pitfalls behind several of the design
choices above, live in [`AGENTS.md`](AGENTS.md).

## Known boundaries

- Discord's receive path is not a high-level public `discord.py` API. This bot
  intentionally uses connection/socket internals, as Hermes Agent does, so new
  Discord or `discord.py` voice protocol changes may require maintenance.
- Speaker identity depends on Discord's SSRC speaking events. An unmapped DAVE
  stream is dropped rather than risk decrypting or attributing it incorrectly.
- One active transcription session is allowed per server.

## License

MIT. See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).
