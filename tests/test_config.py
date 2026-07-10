"""Configuration tests: sane defaults and environment overrides."""

from pathlib import Path

from agentic_rag.config import Settings, get_settings


def test_defaults_are_sane() -> None:
    settings = Settings(_env_file=None)  # ignore any local .env for a reproducible test
    assert settings.chunk_size > settings.chunk_overlap
    assert settings.top_k_final <= settings.top_k_semantic
    assert settings.data_dir == Path("data")


def test_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "800")
    monkeypatch.setenv("CHAT_MODEL", "gpt-4o")
    settings = Settings(_env_file=None)
    assert settings.chunk_size == 800
    assert settings.chat_model == "gpt-4o"


def test_get_settings_is_singleton() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
