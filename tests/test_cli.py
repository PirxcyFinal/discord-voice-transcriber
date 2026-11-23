from __future__ import annotations

import pytest

from discord_voice_transcriber import __main__ as cli
from discord_voice_transcriber import transcription


@pytest.fixture(autouse=True)
def _isolate_cli(monkeypatch):
    """Keep the CLI away from a developer's real .env and ambient credentials."""
    monkeypatch.setattr(cli, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("STT_PROVIDER", raising=False)
    monkeypatch.delenv("STT_MODEL", raising=False)
    monkeypatch.delenv("STT_LANGUAGE", raising=False)
    monkeypatch.delenv("DISCORD_GUILD_ID", raising=False)


def test_missing_stt_backend_exits_with_actionable_message(monkeypatch, capsys):
    """No key and no local model must fail loudly, not raise a raw traceback."""
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "placeholder-token")
    monkeypatch.setattr(transcription, "_local_backend_available", lambda: False)
    monkeypatch.setattr("sys.argv", ["discord-voice-transcriber", "--check"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    message = str(exit_info.value)
    assert exit_info.value.code != 0
    assert "uv sync --extra local" in message
    assert "OPENAI_API_KEY" in message
    assert "STT_PROVIDER" in message
    assert "Traceback" not in capsys.readouterr().err


def test_missing_token_exits_with_actionable_message(monkeypatch):
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.setattr("sys.argv", ["discord-voice-transcriber", "--check"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert "DISCORD_BOT_TOKEN" in str(exit_info.value)


def test_check_reports_resolved_and_requested_provider(monkeypatch, capsys):
    """`auto` must never hide which backend will actually run."""
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "placeholder-token")
    monkeypatch.setattr(transcription, "_local_backend_available", lambda: True)
    monkeypatch.setattr("sys.argv", ["discord-voice-transcriber", "--check"])

    cli.main()

    output = capsys.readouterr().out
    assert "provider=local" in output
    assert "requested=auto" in output
    assert "model=base" in output
