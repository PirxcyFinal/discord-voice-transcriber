# AGENTS.md

Guidance for AI coding agents and LLM-driven tools working on — or rebuilding —
this project. Humans are welcome to read it too; it is simply more detailed and
more blunt about failure modes than [`README.md`](README.md).

If you are an agent asked to "build a Discord voice transcription bot", read
[Rebuilding this from scratch](#rebuilding-this-from-scratch) before writing any
code. The hard parts of this project are not the parts that look hard.

---

## 1. What this repository is

A single-purpose Discord bot. One user runs `/listen`, the bot joins **that
user's** voice channel, captures every participant as a **separate** audio
stream, transcribes each utterance, and posts `**Name:** text` to the text
channel the command came from.

It is deliberately **not** a framework, not a general assistant, and not a
recording archive. There is no database and no retained audio.

The Discord receive path is adapted from
[Hermes Agent](https://github.com/NousResearch/hermes-agent); see
[`NOTICE.md`](NOTICE.md).

### Scope boundaries

| In scope | Out of scope |
| --- | --- |
| Joining the invoker's voice channel | Playing audio / TTS |
| Per-speaker capture and attribution | Mixing speakers into one track |
| Speech-to-text and posting transcripts | Storing audio or transcripts |
| Consent gating and cleanup | Moderation, summarisation, agent replies |

---

## 2. Keys and credentials

There are exactly **two** possible secrets. Both live in `.env`, which is
gitignored. Never commit either, and never print them in logs.

### `DISCORD_BOT_TOKEN` — always required

This is Discord's gateway authentication. There is no way to connect without it.
Get it from the [Discord Developer Portal](https://discord.com/developers/applications)
→ your application → Bot → Reset Token.

### `OPENAI_API_KEY` — only for cloud STT

Required **only** when the resolved provider is `openai`. If you run the local
provider, this variable is never read.

### Is it keyless?

For transcription: **yes.**

```bash
uv sync --extra local
DISCORD_BOT_TOKEN=... uv run discord-voice-transcriber
```

`faster-whisper` runs on your own hardware, no API key, no audio leaving the
host. The weights are open and fetched once from Hugging Face, so there is no
per-minute cost either. But the **Discord bot token is still mandatory** — that
is platform auth, not a speech key. Do not tell a user the project is "fully
keyless"; it is keyless for *speech-to-text only*.

If neither a local backend nor a key is available, startup fails loudly with an
actionable message. It does not silently degrade.

### Required Discord configuration

- OAuth2 scopes: `bot`, `applications.commands`
- Bot permissions: View Channels, Connect, Send Messages, Use Application Commands
- Gateway intents: `Guilds`, `Voice States`

The privileged **Message Content** intent is *not* needed. Do not enable it; it
triggers Discord verification requirements for no benefit here.

---

## 3. How it works

### Data flow

```text
Discord voice UDP socket
  │  raw RTP packets
  ▼
receiver.py  _on_packet()
  │  validate RTP header → skip own SSRC → parse CSRC/extension/padding
  │  decrypt (xchacha20poly1305 AEAD via PyNaCl)
  │  optional DAVE end-to-end decrypt (davey), keyed BY USER ID
  │  Opus decode — one decoder instance PER SSRC
  ▼
per-SSRC PCM buffers (48 kHz, stereo, s16le)
  │  check_silence() every 250 ms
  │  emit when silence ≥ 1.5 s AND speech ≥ 0.5 s
  ▼
bot.py  _publish_utterance()
  │
  ├─ provider "local"  → decode PCM in-process → 16 kHz mono float32 → faster-whisper
  └─ provider "openai" → temp WAV → transcriptions API → unlink in `finally`
  ▼
"**Display name:** text"  → the invoking text channel
```

### The key insight: SSRC is identity

Discord does **not** label audio packets with user IDs. Each RTP packet carries
an **SSRC** (a synchronisation source number). The mapping from SSRC → user ID
arrives separately, over the voice **websocket**, as opcode 5 ("speaking").

So `receiver.py` wraps `conn.hook` to intercept op 5 and build
`_ssrc_to_user`. Everything about speaker attribution depends on this. Keep
per-SSRC buffers and per-SSRC Opus decoders — a shared decoder corrupts output
because Opus is a stateful codec.

### Module map

| File | Responsibility |
| --- | --- |
| `receiver.py` | RTP parsing, decryption, DAVE, Opus, per-speaker buffers, silence segmentation |
| `transcription.py` | Provider resolution, PCM→whisper conversion, OpenAI + local backends |
| `bot.py` | Slash commands, `VoiceSession` lifecycle, polling, publishing, cleanup |
| `config.py` | `Settings.from_env()` — environment only, no file parsing |
| `__main__.py` | CLI entry point and `--check` validation |

### Tunable constants (`VoiceReceiver`)

```python
SILENCE_THRESHOLD   = 1.5     # seconds of quiet that ends an utterance
MIN_SPEECH_DURATION = 0.5     # shorter buffers are discarded as noise
SAMPLE_RATE         = 48_000  # Discord is always 48 kHz
CHANNELS            = 2       # Discord is always stereo
REKEY_FAILURE_STREAK = 25     # consecutive decrypt failures → refresh keys
```

`SAMPLE_RATE` and `CHANNELS` are dictated by Discord. Do not "optimise" them.

### Model selection

Two constants in `transcription.py`, and the difference between them matters:

| Constant | Value | Meaning |
| --- | --- | --- |
| `HERMES_DEFAULT_MODELS` | `{"local": "base", "openai": "whisper-1"}` | A **record of what Hermes uses**. Do not change it to express a preference; it documents upstream. |
| `DEFAULT_MODELS` | `{"local": RECOMMENDED_LOCAL_MODEL, "openai": "whisper-1"}` | What **this project** actually defaults to. `Transcriber` reads this one. |

`RECOMMENDED_LOCAL_MODEL` is `Systran/faster-distil-whisper-small.en`. The cloud
default stays at Hermes parity; only the local default diverges, and it diverges
because it was measured, not assumed. Provider order is unchanged: explicit
choice wins, otherwise **local before cloud**.

If you are changing the default model, benchmark it first — accuracy *and* speed,
on speech resembling the real workload (acronyms, names, numbers, jargon), not
on a clean sentence like "the quick brown fox", which every model gets right and
which therefore tells you nothing. Measured results are in the README table and
in the comment above `RECOMMENDED_LOCAL_MODEL`. Two findings worth keeping:
distillation beat raw size (336 MB more accurate than 818 MB, and 5x faster),
and `tiny.en` mangles exactly the technical acronyms this project deals in.

**English-only defaults are a trap, so the code names it.** Checkpoints ending
`.en` do not refuse other languages; they emit confident English nonsense over
them, which no exception will reveal. `is_english_only_model()` detects the
suffix and `--check` prints a reminder. If you add a model-selection feature,
preserve that warning — silence here produces plausible, wrong transcripts.
`STT_MODEL=base` restores both Hermes parity and multilingual support.

If asked to "use the same model as Hermes", resolve it from Hermes itself rather
than from memory or documentation:

```bash
~/.hermes/hermes-agent/.venv/bin/python -c \
  "import tools.transcription_tools as t; c=t._load_stt_config(); \
   print(t._get_provider(c), c['local']['model'], c['openai']['model'])"
```

---

## 4. Pitfalls that have actually bitten this code

Each of these was a real bug, not a hypothetical.

### `faster_whisper.audio.decode_audio` is broken on modern installs

It calls `av.open(..., metadata_errors=...)`. PyAV removed that keyword in v15,
but faster-whisper still only declares `av>=11`, so a fresh install pulls av 19
and the library breaks itself:

```text
TypeError: open() got an unexpected keyword argument 'metadata_errors'
```

Pinning av backwards does **not** work — older releases ship no wheels for
current Python versions and fail to build from source.

**Fix used here:** never call `decode_audio`. The receiver already has raw PCM,
so `pcm_to_whisper_audio()` downmixes to mono and resamples 48 k → 16 k itself,
then passes the float32 `ndarray` straight to `model.transcribe()`, which skips
`decode_audio` entirely. Low-pass **before** decimating or everything above the
new Nyquist frequency aliases into the speech band and accuracy drops.

### Passing tests do not prove the pipeline runs

A green suite and a successful `--check` both missed the av failure above,
because neither executed the model. Verify runtime claims by running the thing:

```bash
say -o /tmp/t.aiff "the quick brown fox jumps over the lazy dog"
ffmpeg -loglevel error -y -i /tmp/t.aiff -ar 48000 -ac 2 -f s16le /tmp/t.pcm
env -u OPENAI_API_KEY uv run --all-extras python -c "
import asyncio, pathlib
from discord_voice_transcriber.transcription import Transcriber
t = Transcriber(provider='auto')
print(t.provider, t.model)
print(asyncio.run(t.transcribe_pcm(pathlib.Path('/tmp/t.pcm').read_bytes(), user_id=1)))
"
```

Note `env -u OPENAI_API_KEY`: that is what makes a "keyless" claim real.

### `find_library("opus")` returns `None` on macOS Homebrew

`discord.py` cannot find libopus even when it is installed. `load_opus_codec()`
probes explicit Homebrew and Linux paths as a fallback. Opus loading must happen
before the first `discord.opus.Decoder()`.

### Voice credentials rotate mid-session

Websocket ops 4/22 mean rekeying; op 21 with `protocol_version: 0` means a DAVE
downgrade. Cache the secret key and *refresh it* on those ops, plus on a streak
of decrypt failures. Do not read `conn.secret_key` on the hot path and do not
assume it is stable.

### Unmapped DAVE streams are dropped on purpose

If DAVE is active and an SSRC has no known user ID, the packet is discarded.
Guessing produces misattributed transcripts, which is worse than a gap. The one
exception is single-occupant inference in `_infer_user_for_ssrc()`, which is
unambiguous.

### `discord.py` has no public receive API

This bot touches `voice_client._connection`, `conn.hook`, and
`add_socket_listener`. That is intentional and mirrors Hermes. It also means
Discord or `discord.py` updates can break reception, so changes here need real
voice-channel testing, not just unit tests.

---

## 5. Working on this repository

```bash
uv sync --extra test --extra local   # install dev + local model deps
uv run pytest                        # tests
uvx ruff check .                     # lint
uv build                             # wheel + sdist
uv run discord-voice-transcriber --check   # config validation, no Discord connection
```

`--check` prints the **resolved** provider alongside the requested one, so
`auto` cannot hide which backend will actually run.

### Conventions

- Test-first. The STT model-parity change and the PCM-conversion change were
  both written as failing tests first.
- Keep Ruff clean; do not leave `# noqa` without an explanatory comment.
- Never log audio, tokens, or API keys.
- Clean up in `finally`. Temp files, tasks, sockets, and voice connections must
  release on both success and failure paths.
- Do not add features that retain audio without an explicit request — the
  privacy posture is a feature.

### Verifying a change honestly

State what you actually ran. If you could not test the live Discord voice path
(usually true in CI and for agents), say so explicitly rather than implying
end-to-end coverage. Multi-user RTP, DAVE transitions, and SSRC churn across
rejoins are **not** covered by the unit suite.

---

## 6. Rebuilding this from scratch

A realistic order of work, with the traps marked.

**1. Discord application.** Create app + bot, copy the token, invite with `bot`
and `applications.commands`, enable Guilds + Voice States intents only.

**2. Minimal client.** `discord.Client` + `app_commands.CommandTree`. Add
`/listen` with a required `acknowledge_consent: bool`. Use `DISCORD_GUILD_ID` for
instant command sync while developing — global sync is slow.

**3. Join the invoker's channel.** Read `interaction.user.voice.channel`. Refuse
politely when it is `None`. Connect with `self_deaf=False`, or you receive
silence.

**4. Install libopus.** `brew install opus`. Expect `find_library` to fail and
write explicit path fallbacks.

**5. Receive packets.** `voice_client._connection.add_socket_listener(cb)`.
Validate `(data[0] >> 6) == 2` and `(data[1] & 0x7F) == 0x78`, then unpack
`>BBHII` for the SSRC. Skip your own SSRC or you transcribe the bot.

**6. Decrypt.** `nacl.secret.Aead(secret_key).decrypt(payload, header, nonce)`,
where the nonce is the trailing 4 bytes padded to 24. Strip the RTP extension
and padding *after* decrypting, not before.

**7. Map SSRC → user.** Wrap `conn.hook`, watch websocket op 5, record
`ssrc → user_id`. **This is the step people skip, and without it you have audio
nobody can attribute.**

**8. Decode Opus per SSRC.** One `discord.opus.Decoder()` per SSRC, cached.
Append PCM into per-SSRC buffers.

**9. Segment.** Poll every ~250 ms; emit a buffer once it has been quiet for
1.5 s and holds ≥ 0.5 s of speech. Drop stale buffers.

**10. Transcribe.** Prefer handing whisper a numpy array over writing files —
see §4. Do each transcription in a worker thread or task so the socket callback
never blocks.

**11. Publish and clean up.** Post `**display_name:** text`, truncate to
Discord's 2000-character limit, and flush pending utterances on `/leave` and in
`close()`.

**12. Consent and privacy.** Gate on an explicit acknowledgement, announce in
channel that transcription is live, delete temporary audio, and document the law
is the operator's responsibility.

### If you only remember three things

1. **SSRC → user ID comes from websocket op 5**, not from the audio packets.
2. **Per-SSRC Opus decoders and buffers**, or speakers corrupt each other.
3. **Run it with the key unset** before claiming anything is keyless.

---

## 7. Licence and attribution

MIT. See [`LICENSE`](LICENSE). The Discord receive path derives from Hermes
Agent — preserve the attribution in [`NOTICE.md`](NOTICE.md) in any fork.
