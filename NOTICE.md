# Third-party notices

Portions of `src/discord_voice_transcriber/receiver.py` are adapted from the
Discord voice receiver in Hermes Agent:

- Upstream: https://github.com/NousResearch/hermes-agent
- Source path: `plugins/platforms/discord/adapter.py`
- Upstream commit inspected: `c62a6d4609fd447f05e4ac1ca33c1d4511147864`
- Copyright: Nous Research
- License: MIT (the upstream notice is preserved in `LICENSE`)

The extracted code was reduced to voice packet reception, RTP/XSalsa20-Poly1305
transport decryption, Discord DAVE handling, Opus decoding, per-SSRC buffering,
and speech segmentation. Hermes gateway, tool, chat, and agent functionality is
not included.
