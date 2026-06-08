from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is installed in normal runs
    load_dotenv = None


@dataclass(slots=True)
class BotConfig:
    discord_token: str
    database_path: Path
    bot_admin_ids: frozenset[int]
    ai_provider: str
    openai_api_key: str | None
    openai_model: str
    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: float
    ai_scan_all: bool
    sync_guild_id: int | None
    log_level: str
    enable_members_intent: bool


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_set_env(name: str) -> frozenset[int]:
    value = os.getenv(name, "")
    ids: set[int] = set()
    for part in value.replace(",", " ").split():
        part = part.strip()
        if part.isdigit():
            ids.add(int(part))
    return frozenset(ids)


def load_config() -> BotConfig:
    if load_dotenv:
        load_dotenv()

    token = os.getenv("DISCORD_TOKEN", "").strip()
    if not token:
        raise RuntimeError("DISCORD_TOKEN is required. Copy .env.example to .env and set it.")

    guild_raw = os.getenv("DISCORD_TEST_GUILD_ID", "").strip()
    return BotConfig(
        discord_token=token,
        database_path=Path(os.getenv("DATABASE_PATH", "data/prettywords.sqlite3")),
        bot_admin_ids=_int_set_env("BOT_ADMIN_IDS"),
        ai_provider=os.getenv("AI_PROVIDER", "auto").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5-nano"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:4b").strip(),
        ollama_timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "30")),
        ai_scan_all=_bool_env("AI_SCAN_ALL", False),
        sync_guild_id=int(guild_raw) if guild_raw else None,
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        enable_members_intent=_bool_env("ENABLE_MEMBERS_INTENT", False),
    )
