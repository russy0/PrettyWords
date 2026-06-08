import asyncio
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

from prettywords.storage import ModerationStore


def test_ai_settings_persist_and_reset():
    async def run():
        with TemporaryDirectory() as tmp:
            store = ModerationStore(Path(tmp) / "prettywords.sqlite3")
            await store.connect()

            settings = await store.update_settings(
                1,
                ai_provider="ollama",
                ai_model="qwen3:4b",
                ai_scan_all=False,
                health_log_channel_id=777,
            )
            assert settings.ai_provider == "ollama"
            assert settings.ai_model == "qwen3:4b"
            assert settings.ai_scan_all is False
            assert settings.health_log_channel_id == 777

            settings = await store.update_settings(1, ai_provider="", ai_model="", ai_scan_all=None)
            assert settings.ai_provider == ""
            assert settings.ai_model == ""
            assert settings.ai_scan_all is None

            await store.close()

    asyncio.run(run())


def test_existing_database_gets_ai_columns():
    async def run():
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.sqlite3"
            conn = sqlite3.connect(path)
            conn.execute(
                """
                CREATE TABLE guild_settings (
                    guild_id INTEGER PRIMARY KEY,
                    paused INTEGER NOT NULL DEFAULT 0,
                    log_channel_id INTEGER,
                    timeout_minutes INTEGER NOT NULL DEFAULT 10,
                    confidence_threshold REAL NOT NULL DEFAULT 0.78,
                    delete_messages INTEGER NOT NULL DEFAULT 1,
                    dm_users INTEGER NOT NULL DEFAULT 1,
                    dry_run INTEGER NOT NULL DEFAULT 0,
                    escalate INTEGER NOT NULL DEFAULT 1,
                    ai_enabled INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.commit()
            conn.close()

            store = ModerationStore(path)
            await store.connect()
            settings = await store.get_settings(1)
            assert settings.ai_provider == ""
            assert settings.ai_model == ""
            assert settings.ai_scan_all is None
            assert settings.health_log_enabled is True
            assert settings.health_log_channel_id is None
            await store.close()

    asyncio.run(run())


def test_blocked_terms_and_learning_categories_persist():
    async def run():
        with TemporaryDirectory() as tmp:
            store = ModerationStore(Path(tmp) / "prettywords.sqlite3")
            await store.connect()

            await store.add_blocked_term(1, "bad", 2, 999, category="sexual")
            terms = await store.list_blocked_terms(1)
            assert terms[0].term == "bad"
            assert terms[0].category == "sexual"

            await store.add_learning_event(
                guild_id=1,
                label="confirmed_bad",
                source_type="message",
                source_id=123,
                content="very bad",
                term="bad",
                category="sexual",
                created_by=999,
            )
            assert await store.learning_examples(1, "confirmed_bad") == ["[category=sexual term=bad] very bad"]
            await store.close()

    asyncio.run(run())


def test_config_admins_persist():
    async def run():
        with TemporaryDirectory() as tmp:
            store = ModerationStore(Path(tmp) / "prettywords.sqlite3")
            await store.connect()

            assert await store.has_config_admins(1) is False
            await store.add_config_admin(1, 123, 999)
            assert await store.has_config_admins(1) is True
            assert await store.is_config_admin(1, 123) is True
            assert await store.list_config_admins(1) == [123]
            assert await store.remove_config_admin(1, 123) == 1
            assert await store.has_config_admins(1) is False

            await store.close()

    asyncio.run(run())
