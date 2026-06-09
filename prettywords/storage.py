from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .filtering import ModerationDecision, ModerationTerm


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class GuildSettings:
    guild_id: int
    paused: bool = False
    log_channel_id: int | None = None
    health_log_channel_id: int | None = None
    timeout_minutes: int = 10
    confidence_threshold: float = 0.78
    delete_messages: bool = True
    dm_users: bool = True
    dry_run: bool = False
    escalate: bool = True
    ai_enabled: bool = True
    ai_provider: str = ""
    ai_model: str = ""
    ai_scan_all: bool | None = None
    health_log_enabled: bool = True


@dataclass(slots=True)
class InfractionRecord:
    id: int
    guild_id: int
    channel_id: int
    message_id: int
    user_id: int
    content: str
    normalized_hash: str
    decision_json: str


class ModerationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        await self._migrate()

    async def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def _require_conn(self) -> sqlite3.Connection:
        if not self.conn:
            raise RuntimeError("ModerationStore is not connected")
        return self.conn

    async def _migrate(self) -> None:
        conn = self._require_conn()
        async with self._lock:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA foreign_keys=ON;

                CREATE TABLE IF NOT EXISTS guild_settings (
                    guild_id INTEGER PRIMARY KEY,
                    paused INTEGER NOT NULL DEFAULT 0,
                    log_channel_id INTEGER,
                    health_log_channel_id INTEGER,
                    timeout_minutes INTEGER NOT NULL DEFAULT 10,
                    confidence_threshold REAL NOT NULL DEFAULT 0.78,
                    delete_messages INTEGER NOT NULL DEFAULT 1,
                    dm_users INTEGER NOT NULL DEFAULT 1,
                    dry_run INTEGER NOT NULL DEFAULT 0,
                    escalate INTEGER NOT NULL DEFAULT 1,
                    ai_enabled INTEGER NOT NULL DEFAULT 1,
                    ai_provider TEXT NOT NULL DEFAULT '',
                    ai_model TEXT NOT NULL DEFAULT '',
                    ai_scan_all INTEGER,
                    health_log_enabled INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS disabled_channels (
                    guild_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    disabled_by INTEGER NOT NULL,
                    disabled_at TEXT NOT NULL,
                    PRIMARY KEY (guild_id, channel_id)
                );

                CREATE TABLE IF NOT EXISTS blocked_terms (
                    guild_id INTEGER NOT NULL,
                    term TEXT NOT NULL,
                    severity INTEGER NOT NULL DEFAULT 2,
                    category TEXT NOT NULL DEFAULT 'profanity',
                    notes TEXT NOT NULL DEFAULT '',
                    added_by INTEGER NOT NULL,
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (guild_id, term)
                );

                CREATE TABLE IF NOT EXISTS allowed_terms (
                    guild_id INTEGER NOT NULL,
                    term TEXT NOT NULL,
                    added_by INTEGER NOT NULL,
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (guild_id, term)
                );

                CREATE TABLE IF NOT EXISTS allowed_hashes (
                    guild_id INTEGER NOT NULL,
                    normalized_hash TEXT NOT NULL,
                    added_by INTEGER NOT NULL,
                    added_at TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (guild_id, normalized_hash)
                );

                CREATE TABLE IF NOT EXISTS exempt_roles (
                    guild_id INTEGER NOT NULL,
                    role_id INTEGER NOT NULL,
                    added_by INTEGER NOT NULL,
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (guild_id, role_id)
                );

                CREATE TABLE IF NOT EXISTS exempt_users (
                    guild_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    added_by INTEGER NOT NULL,
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (guild_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS config_admins (
                    guild_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    added_by INTEGER NOT NULL,
                    added_at TEXT NOT NULL,
                    PRIMARY KEY (guild_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS infractions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    message_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    content TEXT NOT NULL,
                    normalized_hash TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    action TEXT NOT NULL,
                    timeout_minutes INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL,
                    reporter_id INTEGER NOT NULL,
                    infraction_id INTEGER,
                    message_id INTEGER,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    resolution TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                );

                CREATE TABLE IF NOT EXISTS learning_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id INTEGER,
                    content TEXT NOT NULL,
                    term TEXT,
                    category TEXT NOT NULL DEFAULT '',
                    created_by INTEGER,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._ensure_column(conn, "guild_settings", "health_log_channel_id", "INTEGER")
            self._ensure_column(conn, "guild_settings", "ai_provider", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "guild_settings", "ai_model", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "guild_settings", "ai_scan_all", "INTEGER")
            self._ensure_column(conn, "guild_settings", "health_log_enabled", "INTEGER NOT NULL DEFAULT 1")
            self._ensure_column(conn, "blocked_terms", "category", "TEXT NOT NULL DEFAULT 'profanity'")
            self._ensure_column(conn, "learning_events", "category", "TEXT NOT NULL DEFAULT ''")
            conn.commit()

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        if column not in {row["name"] for row in rows}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    async def get_settings(self, guild_id: int) -> GuildSettings:
        conn = self._require_conn()
        async with self._lock:
            conn.execute("INSERT OR IGNORE INTO guild_settings (guild_id) VALUES (?)", (guild_id,))
            row = conn.execute("SELECT * FROM guild_settings WHERE guild_id = ?", (guild_id,)).fetchone()
            conn.commit()
        return self._settings_from_row(row)

    async def update_settings(self, guild_id: int, **fields: Any) -> GuildSettings:
        if not fields:
            return await self.get_settings(guild_id)
        allowed = {
            "paused",
            "log_channel_id",
            "health_log_channel_id",
            "timeout_minutes",
            "confidence_threshold",
            "delete_messages",
            "dm_users",
            "dry_run",
            "escalate",
            "ai_enabled",
            "ai_provider",
            "ai_model",
            "ai_scan_all",
            "health_log_enabled",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unknown settings fields: {', '.join(sorted(unknown))}")

        await self.get_settings(guild_id)
        assignments = ", ".join(f"{name} = ?" for name in fields)
        values = [self._sqlite_value(value) for value in fields.values()]
        values.append(guild_id)

        conn = self._require_conn()
        async with self._lock:
            conn.execute(f"UPDATE guild_settings SET {assignments} WHERE guild_id = ?", values)
            conn.commit()
        return await self.get_settings(guild_id)

    def _settings_from_row(self, row: sqlite3.Row) -> GuildSettings:
        return GuildSettings(
            guild_id=int(row["guild_id"]),
            paused=bool(row["paused"]),
            log_channel_id=row["log_channel_id"],
            health_log_channel_id=row["health_log_channel_id"],
            timeout_minutes=int(row["timeout_minutes"]),
            confidence_threshold=float(row["confidence_threshold"]),
            delete_messages=bool(row["delete_messages"]),
            dm_users=bool(row["dm_users"]),
            dry_run=bool(row["dry_run"]),
            escalate=bool(row["escalate"]),
            ai_enabled=bool(row["ai_enabled"]),
            ai_provider=str(row["ai_provider"] or ""),
            ai_model=str(row["ai_model"] or ""),
            ai_scan_all=None if row["ai_scan_all"] is None else bool(row["ai_scan_all"]),
            health_log_enabled=bool(row["health_log_enabled"]),
        )

    def _sqlite_value(self, value: Any) -> Any:
        if isinstance(value, bool):
            return int(value)
        return value

    async def set_channel_disabled(self, guild_id: int, channel_id: int, disabled_by: int, disabled: bool) -> None:
        conn = self._require_conn()
        async with self._lock:
            if disabled:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO disabled_channels
                    (guild_id, channel_id, disabled_by, disabled_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (guild_id, channel_id, disabled_by, utc_now_iso()),
                )
            else:
                conn.execute(
                    "DELETE FROM disabled_channels WHERE guild_id = ? AND channel_id = ?",
                    (guild_id, channel_id),
                )
            conn.commit()

    async def is_channel_disabled(self, guild_id: int, channel_id: int) -> bool:
        conn = self._require_conn()
        async with self._lock:
            row = conn.execute(
                "SELECT 1 FROM disabled_channels WHERE guild_id = ? AND channel_id = ?",
                (guild_id, channel_id),
            ).fetchone()
        return row is not None

    async def list_disabled_channels(self, guild_id: int) -> list[int]:
        conn = self._require_conn()
        async with self._lock:
            rows = conn.execute(
                "SELECT channel_id FROM disabled_channels WHERE guild_id = ? ORDER BY disabled_at DESC",
                (guild_id,),
            ).fetchall()
        return [int(row["channel_id"]) for row in rows]

    async def add_blocked_term(
        self,
        guild_id: int,
        term: str,
        severity: int,
        added_by: int,
        notes: str = "",
        category: str = "profanity",
    ) -> None:
        conn = self._require_conn()
        async with self._lock:
            conn.execute(
                """
                INSERT OR REPLACE INTO blocked_terms
                (guild_id, term, severity, category, notes, added_by, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    guild_id,
                    term.strip(),
                    max(1, min(3, severity)),
                    category[:64],
                    notes[:300],
                    added_by,
                    utc_now_iso(),
                ),
            )
            conn.commit()

    async def remove_blocked_term(self, guild_id: int, term: str) -> int:
        conn = self._require_conn()
        async with self._lock:
            cursor = conn.execute(
                "DELETE FROM blocked_terms WHERE guild_id = ? AND term = ?",
                (guild_id, term.strip()),
            )
            conn.commit()
        return cursor.rowcount

    async def list_blocked_terms(self, guild_id: int) -> list[ModerationTerm]:
        conn = self._require_conn()
        async with self._lock:
            rows = conn.execute(
                "SELECT term, severity, notes, category FROM blocked_terms WHERE guild_id = ? ORDER BY term",
                (guild_id,),
            ).fetchall()
        return [
            ModerationTerm(row["term"], int(row["severity"]), row["notes"], row["category"] or "profanity")
            for row in rows
        ]

    async def add_allowed_term(self, guild_id: int, term: str, added_by: int) -> None:
        conn = self._require_conn()
        async with self._lock:
            conn.execute(
                """
                INSERT OR REPLACE INTO allowed_terms (guild_id, term, added_by, added_at)
                VALUES (?, ?, ?, ?)
                """,
                (guild_id, term.strip(), added_by, utc_now_iso()),
            )
            conn.commit()

    async def remove_allowed_term(self, guild_id: int, term: str) -> int:
        conn = self._require_conn()
        async with self._lock:
            cursor = conn.execute(
                "DELETE FROM allowed_terms WHERE guild_id = ? AND term = ?",
                (guild_id, term.strip()),
            )
            conn.commit()
        return cursor.rowcount

    async def list_allowed_terms(self, guild_id: int) -> list[str]:
        conn = self._require_conn()
        async with self._lock:
            rows = conn.execute(
                "SELECT term FROM allowed_terms WHERE guild_id = ? ORDER BY term",
                (guild_id,),
            ).fetchall()
        return [str(row["term"]) for row in rows]

    async def add_allowed_hash(self, guild_id: int, normalized_hash: str, added_by: int, reason: str = "") -> None:
        conn = self._require_conn()
        async with self._lock:
            conn.execute(
                """
                INSERT OR REPLACE INTO allowed_hashes
                (guild_id, normalized_hash, added_by, added_at, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (guild_id, normalized_hash, added_by, utc_now_iso(), reason[:300]),
            )
            conn.commit()

    async def is_allowed_hash(self, guild_id: int, normalized_hash: str) -> bool:
        conn = self._require_conn()
        async with self._lock:
            row = conn.execute(
                "SELECT 1 FROM allowed_hashes WHERE guild_id = ? AND normalized_hash = ?",
                (guild_id, normalized_hash),
            ).fetchone()
        return row is not None

    async def set_role_exempt(self, guild_id: int, role_id: int, added_by: int, exempt: bool) -> None:
        conn = self._require_conn()
        async with self._lock:
            if exempt:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO exempt_roles (guild_id, role_id, added_by, added_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (guild_id, role_id, added_by, utc_now_iso()),
                )
            else:
                conn.execute(
                    "DELETE FROM exempt_roles WHERE guild_id = ? AND role_id = ?",
                    (guild_id, role_id),
                )
            conn.commit()

    async def set_user_exempt(self, guild_id: int, user_id: int, added_by: int, exempt: bool) -> None:
        conn = self._require_conn()
        async with self._lock:
            if exempt:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO exempt_users (guild_id, user_id, added_by, added_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (guild_id, user_id, added_by, utc_now_iso()),
                )
            else:
                conn.execute(
                    "DELETE FROM exempt_users WHERE guild_id = ? AND user_id = ?",
                    (guild_id, user_id),
                )
            conn.commit()

    async def is_user_exempt(self, guild_id: int, user_id: int, role_ids: list[int]) -> bool:
        conn = self._require_conn()
        async with self._lock:
            user_row = conn.execute(
                "SELECT 1 FROM exempt_users WHERE guild_id = ? AND user_id = ?",
                (guild_id, user_id),
            ).fetchone()
            if user_row:
                return True
            if not role_ids:
                return False
            placeholders = ",".join("?" for _ in role_ids)
            role_row = conn.execute(
                f"SELECT 1 FROM exempt_roles WHERE guild_id = ? AND role_id IN ({placeholders}) LIMIT 1",
                (guild_id, *role_ids),
            ).fetchone()
        return role_row is not None

    async def add_config_admin(self, guild_id: int, user_id: int, added_by: int) -> None:
        conn = self._require_conn()
        async with self._lock:
            conn.execute(
                """
                INSERT OR REPLACE INTO config_admins (guild_id, user_id, added_by, added_at)
                VALUES (?, ?, ?, ?)
                """,
                (guild_id, user_id, added_by, utc_now_iso()),
            )
            conn.commit()

    async def remove_config_admin(self, guild_id: int, user_id: int) -> int:
        conn = self._require_conn()
        async with self._lock:
            cursor = conn.execute(
                "DELETE FROM config_admins WHERE guild_id = ? AND user_id = ?",
                (guild_id, user_id),
            )
            conn.commit()
        return cursor.rowcount

    async def is_config_admin(self, guild_id: int, user_id: int) -> bool:
        conn = self._require_conn()
        async with self._lock:
            row = conn.execute(
                "SELECT 1 FROM config_admins WHERE guild_id = ? AND user_id = ?",
                (guild_id, user_id),
            ).fetchone()
        return row is not None

    async def has_config_admins(self, guild_id: int) -> bool:
        conn = self._require_conn()
        async with self._lock:
            row = conn.execute(
                "SELECT 1 FROM config_admins WHERE guild_id = ? LIMIT 1",
                (guild_id,),
            ).fetchone()
        return row is not None

    async def list_config_admins(self, guild_id: int) -> list[int]:
        conn = self._require_conn()
        async with self._lock:
            rows = conn.execute(
                "SELECT user_id FROM config_admins WHERE guild_id = ? ORDER BY added_at DESC",
                (guild_id,),
            ).fetchall()
        return [int(row["user_id"]) for row in rows]

    async def create_infraction(
        self,
        *,
        guild_id: int,
        channel_id: int,
        message_id: int,
        user_id: int,
        username: str,
        content: str,
        normalized_hash: str,
        decision: ModerationDecision,
        action: str,
        timeout_minutes: int,
    ) -> int:
        conn = self._require_conn()
        async with self._lock:
            cursor = conn.execute(
                """
                INSERT INTO infractions
                (guild_id, channel_id, message_id, user_id, username, content, normalized_hash,
                 decision_json, action, timeout_minutes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    guild_id,
                    channel_id,
                    message_id,
                    user_id,
                    username,
                    content[:1900],
                    normalized_hash,
                    json.dumps(decision.to_dict(), ensure_ascii=False),
                    action,
                    timeout_minutes,
                    utc_now_iso(),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    async def get_infraction(self, guild_id: int, infraction_id: int) -> InfractionRecord | None:
        conn = self._require_conn()
        async with self._lock:
            row = conn.execute(
                "SELECT * FROM infractions WHERE guild_id = ? AND id = ?",
                (guild_id, infraction_id),
            ).fetchone()
        if not row:
            return None
        return InfractionRecord(
            id=int(row["id"]),
            guild_id=int(row["guild_id"]),
            channel_id=int(row["channel_id"]),
            message_id=int(row["message_id"]),
            user_id=int(row["user_id"]),
            content=row["content"],
            normalized_hash=row["normalized_hash"],
            decision_json=row["decision_json"],
        )

    async def count_recent_infractions(self, guild_id: int, user_id: int, days: int = 7) -> int:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conn = self._require_conn()
        async with self._lock:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM infractions
                WHERE guild_id = ? AND user_id = ? AND created_at >= ?
                """,
                (guild_id, user_id, since),
            ).fetchone()
        return int(row["count"] if row else 0)

    async def create_report(
        self,
        guild_id: int,
        reporter_id: int,
        reason: str,
        infraction_id: int | None = None,
        message_id: int | None = None,
    ) -> int:
        conn = self._require_conn()
        async with self._lock:
            cursor = conn.execute(
                """
                INSERT INTO reports
                (guild_id, reporter_id, infraction_id, message_id, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (guild_id, reporter_id, infraction_id, message_id, reason[:1000], utc_now_iso()),
            )
            conn.commit()
            return int(cursor.lastrowid)

    async def resolve_report(self, guild_id: int, report_id: int, resolution: str) -> sqlite3.Row | None:
        conn = self._require_conn()
        async with self._lock:
            # status = 'open' 조건으로 이미 처리된 신고는 재처리 안 함
            row = conn.execute(
                "SELECT * FROM reports WHERE guild_id = ? AND id = ? AND status = 'open'",
                (guild_id, report_id),
            ).fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE reports
                    SET status = 'resolved', resolution = ?, resolved_at = ?
                    WHERE guild_id = ? AND id = ?
                    """,
                    (resolution, utc_now_iso(), guild_id, report_id),
                )
                conn.commit()
        return row

    async def add_learning_event(
        self,
        *,
        guild_id: int,
        label: str,
        source_type: str,
        content: str,
        source_id: int | None = None,
        term: str | None = None,
        category: str | None = None,
        created_by: int | None = None,
    ) -> None:
        conn = self._require_conn()
        async with self._lock:
            conn.execute(
                """
                INSERT INTO learning_events
                (guild_id, label, source_type, source_id, content, term, category, created_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    guild_id,
                    label,
                    source_type,
                    source_id,
                    content[:1200],
                    term,
                    category or "",
                    created_by,
                    utc_now_iso(),
                ),
            )
            conn.commit()

    async def learning_examples(self, guild_id: int, label: str, limit: int = 8) -> list[str]:
        conn = self._require_conn()
        async with self._lock:
            rows = conn.execute(
                """
                SELECT content, term, category FROM learning_events
                WHERE guild_id = ? AND label = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (guild_id, label, limit),
            ).fetchall()
        examples: list[str] = []
        for row in rows:
            tags = []
            if row["category"]:
                tags.append(f"category={row['category']}")
            if row["term"]:
                tags.append(f"term={row['term']}")
            prefix = f"[{' '.join(tags)}] " if tags else ""
            examples.append(f"{prefix}{row['content']}")
        return examples
