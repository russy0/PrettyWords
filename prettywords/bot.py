from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import discord
from discord import app_commands
from discord.ext import commands, tasks

from .ai import AIContext, GroqClassifier, OllamaClassifier, OpenAIClassifier, RateLimitedError
from .config import BotConfig, load_config
from .filtering import (
    CATEGORY_LABELS,
    DEFAULT_BLOCKED_TERMS,
    LocalClassifier,
    ModerationDecision,
    ModerationTerm,
    category_label,
    compact_text,
    combine_decisions,
    message_fingerprint,
    normalize_category,
)
from .storage import GuildSettings, ModerationStore


LOGGER = logging.getLogger(__name__)
MAX_TIMEOUT_MINUTES = 28 * 24 * 60

# ── 길드별 TTL 캐시 ─────────────────────────────────────────────────────────────
# on_message 마다 동일한 DB 쿼리를 반복하지 않도록 짧은 TTL 캐시를 사용합니다.

_CACHE_MISS: object = object()


@dataclass(slots=True)
class _CacheEntry:
    value: Any
    expires_at: datetime


class _GuildCache:
    """길드별 TTL 인메모리 캐시. settings/terms/채널 목록 등 자주 읽히는 값을 캐시합니다."""

    def __init__(self) -> None:
        self._data: dict[tuple, _CacheEntry] = {}

    def get(self, key: tuple) -> Any:
        entry = self._data.get(key)
        if entry is None or datetime.now(timezone.utc) >= entry.expires_at:
            return _CACHE_MISS
        return entry.value

    def put(self, key: tuple, value: Any, ttl_seconds: float) -> None:
        self._data[key] = _CacheEntry(
            value, datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)
        )

    def invalidate_guild(self, guild_id: int) -> None:
        self._data = {k: v for k, v in self._data.items() if k[1] != guild_id}
CATEGORY_CHOICES = [
    app_commands.Choice(name=label, value=value)
    for value, label in CATEGORY_LABELS.items()
]
OUTCOME_LABELS = {
    "false_positive": "오탐",
    "confirmed": "정상 제재",
    "duplicate": "중복",
    "rejected": "기각",
}
ACTION_LABELS = {
    "dry-run": "모의 실행",
    "deleted": "메시지 삭제",
    "already deleted": "이미 삭제됨",
    "delete failed: missing permission": "삭제 실패: 권한 부족",
    "delete failed": "삭제 실패",
    "delete failed: incompatible discord library": "삭제 실패: discord.py 버전 불일치",
    "logged": "로그만 기록",
    "timeout failed: missing permission or role hierarchy": "타임아웃 실패: 권한 또는 역할 순서 문제",
    "timeout failed": "타임아웃 실패",
}
SOURCE_LABELS = {
    "local": "로컬 필터",
    "ai": "AI",
    "local+ai": "로컬+AI",
    "openai": "AI (OpenAI)",
    "ollama": "AI (Ollama)",
    "groq": "AI (Groq)",
}


def _enabled_label(value: bool) -> str:
    return "켜짐" if value else "꺼짐"


def _source_label(value: str) -> str:
    return SOURCE_LABELS.get(value, value)


def _action_label(action: str) -> str:
    labels = []
    for part in (item.strip() for item in action.split(",")):
        if part.startswith("timeout ") and part.endswith("m"):
            labels.append(f"타임아웃 {part.removeprefix('timeout ').removesuffix('m')}분")
        else:
            labels.append(ACTION_LABELS.get(part, part))
    return ", ".join(labels)


def _outcome_label(value: str) -> str:
    return OUTCOME_LABELS.get(value, value)


def _parse_discord_id(value: str) -> int:
    digits = "".join(char for char in value if char.isdigit())
    if not digits:
        raise ValueError("Discord ID must contain digits.")
    return int(digits)


def _is_mod_member(member: discord.Member | discord.User) -> bool:
    if not isinstance(member, discord.Member):
        return False
    perms = member.guild_permissions
    return perms.administrator or perms.manage_guild or perms.moderate_members


async def _can_manage_bot_settings(interaction: discord.Interaction) -> bool:
    if interaction.guild is None or interaction.guild_id is None:
        return False

    bot = interaction.client
    user_id = interaction.user.id

    # 서버 오너 · 전역 봇 관리자 · Discord 관리자 권한은 항상 허용
    if getattr(interaction.guild, "owner_id", None) == user_id:
        return True
    if hasattr(bot, "config") and user_id in bot.config.bot_admin_ids:
        return True
    if isinstance(interaction.user, discord.Member) and interaction.user.guild_permissions.administrator:
        return True

    # 서버별 config_admin 등록 여부 확인
    if hasattr(bot, "store"):
        if await bot.store.is_config_admin(interaction.guild_id, user_id):
            return True
        # config_admin이 한 명이라도 등록됐으면 manage_guild / moderate_members 권한만으로는 접근 불가
        if await bot.store.has_config_admins(interaction.guild_id):
            return False

    # config_admin 미등록 시 Discord mod 권한으로 접근 허용
    return _is_mod_member(interaction.user)


def mod_only() -> app_commands.Check:
    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.guild is None:
            raise app_commands.NoPrivateMessage
        if await _can_manage_bot_settings(interaction):
            return True
        raise app_commands.CheckFailure("PrettyWords config admin permission required.")

    return app_commands.check(predicate)


@dataclass(slots=True)
class _PendingClassification:
    """A message waiting to be classified as part of a Groq batch."""

    message: discord.Message
    settings: GuildSettings
    local: ModerationDecision
    queued_at: datetime
    prior_messages: list[str] = field(default_factory=list)
    matched_terms: list | None = None


async def send_interaction(
    interaction: discord.Interaction,
    content: str | None = None,
    *,
    embed: discord.Embed | None = None,
    ephemeral: bool = True,
) -> None:
    if interaction.response.is_done():
        await interaction.followup.send(content=content, embed=embed, ephemeral=ephemeral)
    else:
        await interaction.response.send_message(content=content, embed=embed, ephemeral=ephemeral)


class PrettyWordsBot(commands.Bot):
    def __init__(self, config: BotConfig) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = config.enable_members_intent

        super().__init__(
            command_prefix=commands.when_mentioned_or("!pw "),
            intents=intents,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self.config = config
        self.store = ModerationStore(config.database_path)
        self.local_classifier = LocalClassifier()
        self._ai_classifier_cache = {}
        self._groq_unavailable_until: datetime | None = None

    def _effective_ai_settings(self, settings: GuildSettings) -> tuple[str, str, bool]:
        provider = (settings.ai_provider or self.config.ai_provider).strip().lower()
        if provider == "auto":
            # settings.ai_model은 길드 전용 오버라이드이므로 ollama를 의미합니다.
            # ollama_model_configured는 OLLAMA_MODEL 환경변수가 명시적으로 설정된 경우에만 True입니다.
            # 기본값("qwen3:4b")만 있는 경우에는 ollama를 자동 선택하지 않습니다.
            if settings.ai_model or self.config.ollama_model_configured:
                provider = "ollama"
            elif self.config.openai_api_key:
                provider = "openai"
            elif self.config.groq_api_keys:
                provider = "groq"
            else:
                provider = "none"

        if provider == "ollama":
            model = settings.ai_model or self.config.ollama_model
        elif provider == "openai":
            model = settings.ai_model or self.config.openai_model
        elif provider == "groq":
            model = settings.ai_model or self.config.groq_model
        else:
            model = settings.ai_model or ""

        scan_all = self.config.ai_scan_all if settings.ai_scan_all is None else settings.ai_scan_all
        return provider, model, scan_all

    def _get_ai_classifier(self, settings: GuildSettings):
        provider, model, _scan_all = self._effective_ai_settings(settings)
        if provider == "none":
            return None
        if provider == "ollama":
            if not model:
                LOGGER.warning("AI_PROVIDER=ollama but OLLAMA_MODEL is empty; AI disabled")
                return None
            return self._cached_ollama_classifier(self.config.ollama_base_url, model, self.config.ollama_timeout_seconds)
        if provider == "openai":
            if not self.config.openai_api_key:
                LOGGER.warning("AI_PROVIDER=openai but OPENAI_API_KEY is empty; AI disabled")
                return None
            key = ("openai", model, self.config.openai_api_key)
            if key not in self._ai_classifier_cache:
                self._ai_classifier_cache[key] = OpenAIClassifier(self.config.openai_api_key, model)
            return self._ai_classifier_cache[key]
        if provider == "groq":
            if not self.config.groq_api_keys:
                LOGGER.warning("AI_PROVIDER=groq but GROQ_API_KEY is empty; AI disabled")
                return None
            if not model:
                LOGGER.warning("AI_PROVIDER=groq but no Groq model configured; AI disabled")
                return None
            key = ("groq", model, self.config.groq_api_keys)
            if key not in self._ai_classifier_cache:
                self._ai_classifier_cache[key] = GroqClassifier(self.config.groq_api_keys, model)
            return self._ai_classifier_cache[key]
        LOGGER.warning("Unknown AI_PROVIDER=%s; AI disabled", provider)
        return None

    def _cached_ollama_classifier(self, base_url: str, model: str, timeout_seconds: float) -> OllamaClassifier:
        key = ("ollama", base_url, model, timeout_seconds)
        if key not in self._ai_classifier_cache:
            self._ai_classifier_cache[key] = OllamaClassifier(base_url, model, timeout_seconds=timeout_seconds)
        return self._ai_classifier_cache[key]

    def _groq_fallback_classifier(self) -> OllamaClassifier | None:
        """Local Ollama classifier used while Groq is in a rate-limit cooldown."""
        if not self.config.ollama_model:
            return None
        return self._cached_ollama_classifier(
            self.config.ollama_base_url, self.config.ollama_model, self.config.ollama_timeout_seconds
        )

    def _groq_in_cooldown(self) -> bool:
        return self._groq_unavailable_until is not None and datetime.now(timezone.utc) < self._groq_unavailable_until

    def _groq_cooldown_remaining(self) -> float:
        if self._groq_unavailable_until is None:
            return 0.0
        remaining = (self._groq_unavailable_until - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, remaining)

    def _enter_groq_cooldown(self, retry_after: float | None = None) -> None:
        seconds = retry_after if retry_after and retry_after > 0 else self.config.groq_rate_limit_cooldown_seconds
        self._groq_unavailable_until = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        LOGGER.warning("Groq rate limited; using local fallback for about %.0fs", seconds)

    def _ai_label(self, settings: GuildSettings) -> str:
        provider, model, _scan_all = self._effective_ai_settings(settings)
        if provider == "none" or not model:
            return "없음"
        return f"{provider}:{model}"

    async def setup_hook(self) -> None:
        await self.store.connect()
        await self.add_cog(ModerationCog(self))

        if self.config.sync_guild_id:
            guild = discord.Object(id=self.config.sync_guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            LOGGER.info("Synced slash commands to guild %s", self.config.sync_guild_id)
        else:
            await self.tree.sync()
            LOGGER.info("Synced global slash commands")

    async def close(self) -> None:
        await self.store.close()
        await super().close()


# ── Discord UI ─────────────────────────────────────────────────────────────────
# custom_id 형식:
#   pw_appeal:{guild_id}:{infraction_id}
#   pw_resolve:{guild_id}:{report_id}:{outcome}
# ModerationCog.on_interaction 에서 처리. 뷰는 표시용으로만 사용하며
# add_view() 등록 없이도 봇 재시작 후 버튼이 동작함.

class _AppealModal(discord.ui.Modal, title="이의제기"):
    """DM 경고의 이의제기 버튼을 눌렀을 때 표시되는 모달."""

    reason: discord.ui.TextInput = discord.ui.TextInput(
        label="이의제기 사유",
        placeholder="이 제재가 오탐지라고 생각하는 이유를 적어주세요.",
        style=discord.TextStyle.paragraph,
        max_length=500,
        required=True,
    )

    def __init__(self, bot: "PrettyWordsBot", guild_id: int, infraction_id: int) -> None:
        super().__init__()
        self._pw_bot = bot
        self._guild_id = guild_id
        self._infraction_id = infraction_id

    async def on_submit(self, interaction: discord.Interaction) -> None:
        report_id = await self._pw_bot.store.create_report(
            self._guild_id,
            interaction.user.id,
            self.reason.value,
            infraction_id=self._infraction_id,
        )
        await interaction.response.send_message(
            f"이의제기가 접수됐습니다 (신고 #{report_id}). 관리자 검토 후 처리됩니다.",
            ephemeral=True,
        )
        guild = self._pw_bot.get_guild(self._guild_id)
        if guild is None:
            return
        settings = await self._pw_bot.store.get_settings(self._guild_id)
        # 이의제기 전용 채널 우선, 없으면 일반 로그 채널로 폴백
        target_ch_id = settings.appeal_channel_id or settings.log_channel_id
        if not target_ch_id:
            return
        target_ch = guild.get_channel(target_ch_id)
        if not isinstance(target_ch, discord.abc.Messageable):
            return

        # ── 이의제기 embed ────────────────────────────────────────────────
        appeal_embed = discord.Embed(
            title=f"✋ 이의제기 #{report_id}",
            description=self.reason.value[:900],
            color=discord.Color.orange(),
        )
        appeal_embed.add_field(
            name="신청자",
            value=f"<@{interaction.user.id}> {interaction.user} (`{interaction.user.id}`)",
            inline=False,
        )
        appeal_embed.add_field(name="제재 기록 ID", value=f"#{self._infraction_id}", inline=True)

        # ── 원본 제재 기록 embed ──────────────────────────────────────────
        infraction = await self._pw_bot.store.get_infraction(self._guild_id, self._infraction_id)
        embeds = [appeal_embed]
        if infraction:
            import json as _json
            try:
                dec = _json.loads(infraction.decision_json)
            except Exception:
                dec = {}
            inf_embed = discord.Embed(
                title=f"📋 원본 제재 기록 #{self._infraction_id}",
                color=discord.Color.red(),
            )
            inf_embed.add_field(
                name="대상 메시지",
                value=f"```{infraction.content[:900]}```",
                inline=False,
            )
            inf_embed.add_field(
                name="AI 판정",
                value=(
                    f"위반: {dec.get('violation', '?')} | "
                    f"확신도: {dec.get('confidence', 0):.2f} | "
                    f"심각도: {dec.get('severity', '?')}"
                ),
                inline=False,
            )
            if dec.get("categories"):
                from .filtering import category_label as _cat_label
                cats = ", ".join(_cat_label(c) for c in dec["categories"])
                inf_embed.add_field(name="카테고리", value=cats, inline=True)
            if dec.get("reason"):
                inf_embed.add_field(name="AI 이유", value=dec["reason"][:300], inline=False)
            ch = guild.get_channel(infraction.channel_id)
            if ch:
                inf_embed.add_field(name="채널", value=ch.mention, inline=True)
            embeds.append(inf_embed)

        view = _ResolveReportView(self._guild_id, report_id)
        appeal_embed.set_footer(text="아래 버튼으로 처리하세요")
        try:
            await target_ch.send(embeds=embeds, view=view)
        except discord.HTTPException:
            LOGGER.exception("Failed to send appeal to channel")

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        LOGGER.exception("AppealModal error: %s", error)
        if not interaction.response.is_done():
            await interaction.response.send_message("처리 중 오류가 발생했습니다.", ephemeral=True)


class _AppealView(discord.ui.View):
    """DM 경고 메시지에 첨부하는 이의제기 버튼."""

    def __init__(self, guild_id: int, infraction_id: int) -> None:
        super().__init__(timeout=None)
        self.add_item(discord.ui.Button(
            label="✋  이의제기",
            style=discord.ButtonStyle.secondary,
            custom_id=f"pw_appeal:{guild_id}:{infraction_id}",
        ))


class _ResolveReportView(discord.ui.View):
    """로그 채널 신고 embed에 첨부하는 처리 버튼들."""

    def __init__(self, guild_id: int, report_id: int) -> None:
        super().__init__(timeout=None)
        for label, outcome, style in (
            ("✅  오탐 처리", "false_positive", discord.ButtonStyle.success),
            ("🔨  정상 제재", "confirmed", discord.ButtonStyle.danger),
            ("❌  기각", "rejected", discord.ButtonStyle.secondary),
        ):
            self.add_item(discord.ui.Button(
                label=label,
                style=style,
                custom_id=f"pw_resolve:{guild_id}:{report_id}:{outcome}",
            ))



def _fast_term_match(text: str, terms: list) -> list:
    """메시지에 포함된 금지어를 빠르게 찾습니다. O(n*k) 단순 포함 탐색."""
    low = text.lower()
    matched = []
    seen: set[str] = set()
    for t in terms:
        w = t.term.lower()
        if w not in seen and w in low:
            matched.append(t)
            seen.add(w)
    return matched


class ModerationCog(commands.Cog):
    filter = app_commands.Group(name="filter", description="AI 비속어 필터 설정")
    admin = app_commands.Group(name="pw", description="PrettyWords 봇 관리")

    def __init__(self, bot: PrettyWordsBot) -> None:
        self.bot = bot
        self._stats: dict[int, dict[str, int | str]] = {}
        self._groq_queues: dict[int, list[_PendingClassification]] = {}
        self._groq_flush_locks: dict[int, asyncio.Lock] = {}
        self._cache = _GuildCache()

    async def cog_load(self) -> None:
        self.health_heartbeat.start()
        self.groq_batch_flush.start()

    async def cog_unload(self) -> None:
        self.health_heartbeat.cancel()
        self.groq_batch_flush.cancel()
        for guild_id in list(self._groq_queues):
            await self._flush_groq_queue(guild_id, force=True)

    def _guild_stats(self, guild_id: int) -> dict[str, int | str]:
        return self._stats.setdefault(
            guild_id,
            {
                "seen": 0,
                "checked": 0,
                "skipped": 0,
                "ai_calls": 0,
                "ai_failures": 0,
                "violations": 0,
                "deleted": 0,
                "delete_failures": 0,
                "timeouts": 0,
                "timeout_failures": 0,
                "last_scan": "아직 없음",
                "last_error": "",
            },
        )

    def _bump(self, guild_id: int, key: str, amount: int = 1) -> None:
        stats = self._guild_stats(guild_id)
        stats[key] = int(stats.get(key, 0)) + amount

    def _mark_scan(self, guild_id: int) -> None:
        self._guild_stats(guild_id)["last_scan"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _set_last_error(self, guild_id: int, error: str) -> None:
        self._guild_stats(guild_id)["last_error"] = error[:240]

    @tasks.loop(minutes=10)
    async def health_heartbeat(self) -> None:
        for guild in self.bot.guilds:
            settings = await self.bot.store.get_settings(guild.id)
            if not settings.health_log_enabled or not (settings.health_log_channel_id or settings.log_channel_id):
                continue
            await self._send_health_log(guild, settings, title="PrettyWords 상태")

    @health_heartbeat.before_loop
    async def before_health_heartbeat(self) -> None:
        await self.bot.wait_until_ready()

    @tasks.loop(seconds=2)
    async def groq_batch_flush(self) -> None:
        """Flush per-guild Groq batches once they hit the size or time threshold (whichever first)."""
        if not self._groq_queues:
            return
        now = datetime.now(timezone.utc)
        window = timedelta(seconds=self.bot.config.groq_batch_window_seconds)
        for guild_id, queue in list(self._groq_queues.items()):
            if not queue:
                continue
            if len(queue) >= self.bot.config.groq_batch_size or (now - queue[0].queued_at) >= window:
                await self._flush_groq_queue(guild_id)

    @groq_batch_flush.before_loop
    async def before_groq_batch_flush(self) -> None:
        await self.bot.wait_until_ready()

    async def _send_health_log(
        self,
        guild: discord.Guild,
        settings: GuildSettings,
        *,
        title: str = "PrettyWords 상태",
    ) -> None:
        channel_id = settings.health_log_channel_id or settings.log_channel_id
        channel = guild.get_channel(channel_id) if channel_id else None
        if not isinstance(channel, discord.abc.Messageable):
            return

        stats = self._guild_stats(guild.id)
        _provider, _model, scan_all = self.bot._effective_ai_settings(settings)
        embed = discord.Embed(
            title=title,
            color=discord.Color.green() if not settings.paused else discord.Color.light_grey(),
            description="메시지 검사가 작동 중입니다." if not settings.paused else "필터가 일시정지되어 있습니다.",
        )
        embed.add_field(name="AI", value=self.bot._ai_label(settings) if settings.ai_enabled else "꺼짐", inline=True)
        embed.add_field(name="전체 AI 검사", value=_enabled_label(scan_all), inline=True)
        embed.add_field(name="확신도 기준", value=f"{settings.confidence_threshold:.2f}", inline=True)
        embed.add_field(name="수신", value=str(stats["seen"]), inline=True)
        embed.add_field(name="검사", value=str(stats["checked"]), inline=True)
        embed.add_field(name="건너뜀", value=str(stats["skipped"]), inline=True)
        embed.add_field(name="AI 호출", value=str(stats["ai_calls"]), inline=True)
        embed.add_field(name="AI 실패", value=str(stats["ai_failures"]), inline=True)
        embed.add_field(name="위반", value=str(stats["violations"]), inline=True)
        embed.add_field(name="삭제", value=str(stats["deleted"]), inline=True)
        embed.add_field(name="타임아웃", value=str(stats["timeouts"]), inline=True)
        embed.add_field(name="마지막 검사", value=str(stats["last_scan"]), inline=False)

        queue_len = len(self._groq_queues.get(guild.id, []))
        if queue_len:
            embed.add_field(name="Groq 대기열", value=str(queue_len), inline=True)
        cooldown = self.bot._groq_cooldown_remaining()
        if cooldown > 0:
            embed.add_field(name="Groq 대기", value=f"{cooldown:.0f}초 (로컬 폴백 사용 중)", inline=True)

        if stats.get("last_error"):
            embed.add_field(name="최근 오류", value=str(stats["last_error"])[:1000], inline=False)

        try:
            await channel.send(embed=embed)
        except discord.HTTPException:
            LOGGER.exception("Failed to send health log")

    # ── 캐시 래퍼 메서드 ────────────────────────────────────────────────────────

    async def _cached_settings(self, guild_id: int) -> GuildSettings:
        key = ("settings", guild_id)
        hit = self._cache.get(key)
        if hit is not _CACHE_MISS:
            return hit  # type: ignore[return-value]
        value = await self.bot.store.get_settings(guild_id)
        self._cache.put(key, value, ttl_seconds=5.0)
        return value

    async def _cached_blocked_terms(self, guild_id: int) -> list[ModerationTerm]:
        key = ("blocked_terms", guild_id)
        hit = self._cache.get(key)
        if hit is not _CACHE_MISS:
            return hit  # type: ignore[return-value]
        value = await self.bot.store.list_blocked_terms(guild_id)
        self._cache.put(key, value, ttl_seconds=30.0)
        return value

    async def _cached_allowed_terms(self, guild_id: int) -> list[str]:
        key = ("allowed_terms", guild_id)
        hit = self._cache.get(key)
        if hit is not _CACHE_MISS:
            return hit  # type: ignore[return-value]
        value = await self.bot.store.list_allowed_terms(guild_id)
        self._cache.put(key, value, ttl_seconds=30.0)
        return value

    async def _cached_disabled_channels(self, guild_id: int) -> list[int]:
        key = ("disabled_channels", guild_id)
        hit = self._cache.get(key)
        if hit is not _CACHE_MISS:
            return hit  # type: ignore[return-value]
        value = await self.bot.store.list_disabled_channels(guild_id)
        self._cache.put(key, value, ttl_seconds=10.0)
        return value

    async def _fetch_prior_messages(self, message: discord.Message) -> list[str]:
        """AI 맥락 파악을 위해 현재 메시지 직전 최대 2개의 비봇 메시지를 가져옵니다."""
        history: list[str] = []
        try:
            async for prior in message.channel.history(limit=14, before=message):
                if prior.content and not prior.author.bot:
                    history.append(prior.content[:300])
                if len(history) >= 10:
                    break
        except (discord.Forbidden, discord.HTTPException):
            pass
        return list(reversed(history))

    async def _build_ai_context(
        self, guild_id: int, blocked_terms, allowed_terms, guild_notes: str = ""
    ) -> AIContext:
        return AIContext(
            blocked_terms=blocked_terms,
            allowed_terms=allowed_terms,
            guild_notes=guild_notes,
            confirmed_examples=await self.bot.store.learning_examples(guild_id, "confirmed_bad"),
            false_positive_examples=await self.bot.store.learning_examples(guild_id, "false_positive"),
            auto_examples=await self.bot.store.learning_examples(guild_id, "auto_flagged"),
        )

    def _resolve_ai_classifier(self, settings: GuildSettings):
        """Pick the classifier to use for this guild right now.

        Returns (classifier, batchable). When the configured provider is Groq
        and Groq is currently in a rate-limit cooldown, this transparently
        substitutes the local Ollama classifier (if configured) so moderation
        keeps working without hammering Groq.
        """
        provider, _model, _scan_all = self.bot._effective_ai_settings(settings)
        if provider == "groq" and self.bot._groq_in_cooldown():
            return self.bot._groq_fallback_classifier(), False
        classifier = self.bot._get_ai_classifier(settings)
        return classifier, isinstance(classifier, GroqClassifier)

    @commands.Cog.listener()
    async def on_interaction(self, interaction: discord.Interaction) -> None:
        """pw_appeal / pw_resolve 버튼 처리."""
        if interaction.type != discord.InteractionType.component:
            return
        cid: str = (interaction.data or {}).get("custom_id", "")
        if cid.startswith("pw_appeal:"):
            await self._handle_appeal_button(interaction, cid)
        elif cid.startswith("pw_resolve:"):
            await self._handle_resolve_button(interaction, cid)

    async def _handle_appeal_button(
        self, interaction: discord.Interaction, custom_id: str
    ) -> None:
        """DM 이의제기 버튼 → 모달 표시."""
        try:
            _, guild_id_s, infraction_id_s = custom_id.split(":", 2)
            guild_id, infraction_id = int(guild_id_s), int(infraction_id_s)
        except ValueError:
            await interaction.response.send_message("잘못된 버튼입니다.", ephemeral=True)
            return
        await interaction.response.send_modal(
            _AppealModal(self.bot, guild_id, infraction_id)
        )

    async def _handle_resolve_button(
        self, interaction: discord.Interaction, custom_id: str
    ) -> None:
        """로그 채널 처리 버튼 → 신고 처리 + 버튼 비활성화."""
        # 관리자 권한 확인 (빠른 응답 전에)
        if interaction.guild is None or not isinstance(interaction.user, discord.Member):
            await interaction.response.send_message("서버에서만 사용 가능합니다.", ephemeral=True)
            return
        if not await _can_manage_bot_settings(interaction):
            await interaction.response.send_message("관리자 권한이 필요합니다.", ephemeral=True)
            return

        try:
            _, guild_id_s, report_id_s, outcome = custom_id.split(":", 3)
            guild_id, report_id = int(guild_id_s), int(report_id_s)
        except ValueError:
            await interaction.response.send_message("잘못된 버튼입니다.", ephemeral=True)
            return

        # 버튼의 guild_id가 현재 인터랙션 서버와 일치하는지 검증합니다.
        # 다른 서버의 유저가 버튼을 누르는 것을 방지합니다.
        if guild_id != interaction.guild_id:
            await interaction.response.send_message("잘못된 버튼입니다.", ephemeral=True)
            return

        # DB 작업 전에 defer — 3초 응답 제한 초과 방지
        await interaction.response.defer(ephemeral=True)

        report = await self.bot.store.resolve_report(guild_id, report_id, outcome)
        if not report:
            await interaction.followup.send("신고를 찾을 수 없거나 이미 처리됐습니다.", ephemeral=True)
            return

        infraction = None
        if report["infraction_id"]:
            infraction = await self.bot.store.get_infraction(guild_id, int(report["infraction_id"]))

        if outcome == "false_positive" and infraction:
            await self.bot.store.add_allowed_hash(
                guild_id, infraction.normalized_hash, interaction.user.id,
                reason=f"신고 #{report_id}",
            )
            await self.bot.store.add_learning_event(
                guild_id=guild_id, label="false_positive", source_type="report",
                source_id=report_id, content=infraction.content,
                category="false_positive", created_by=interaction.user.id,
            )
        elif outcome == "confirmed" and infraction:
            await self.bot.store.add_learning_event(
                guild_id=guild_id, label="confirmed_bad", source_type="report",
                source_id=report_id, content=infraction.content,
                category="profanity", created_by=interaction.user.id,
            )

        # 버튼 비활성화 — ActionRow.children 순회
        if interaction.message:
            disabled_view = discord.ui.View()
            for action_row in (interaction.message.components or []):
                for component in getattr(action_row, "children", []):
                    if not hasattr(component, "custom_id"):
                        continue
                    disabled_view.add_item(discord.ui.Button(
                        label=getattr(component, "label", ""),
                        style=getattr(component, "style", discord.ButtonStyle.secondary),
                        custom_id=component.custom_id,
                        disabled=True,
                    ))
            try:
                await interaction.message.edit(view=disabled_view)
            except discord.HTTPException:
                pass

        outcome_text = _outcome_label(outcome)
        await interaction.followup.send(f"신고 #{report_id} 처리됨: {outcome_text}", ephemeral=True)
        LOGGER.info(
            "[guild:%d] report #%d resolved via button: %s by %s",
            guild_id, report_id, outcome, interaction.user,
        )

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None or message.author.bot or not message.content:
            return

        self._bump(message.guild.id, "seen")
        settings = await self._cached_settings(message.guild.id)
        if settings.paused:
            self._bump(message.guild.id, "skipped")
            return
        disabled_channels = await self._cached_disabled_channels(message.guild.id)
        if message.channel.id in disabled_channels:
            self._bump(message.guild.id, "skipped")
            return

        role_ids = [role.id for role in getattr(message.author, "roles", [])]
        if await self.bot.store.is_user_exempt(message.guild.id, message.author.id, role_ids):
            self._bump(message.guild.id, "skipped")
            return

        fingerprint = message_fingerprint(message.content)
        if await self.bot.store.is_allowed_hash(message.guild.id, fingerprint):
            self._bump(message.guild.id, "skipped")
            return

        self._bump(message.guild.id, "checked")
        self._mark_scan(message.guild.id)

        LOGGER.debug(
            "[%s] checking: %s#%s | %.80r",
            message.guild.name,
            message.author,
            getattr(message.channel, "name", message.channel.id),
            message.content,
        )

        blocked_terms = await self._cached_blocked_terms(message.guild.id)
        allowed_terms = await self._cached_allowed_terms(message.guild.id)

        _provider, _model, _scan_all = self.bot._effective_ai_settings(settings)
        classifier, batchable = self._resolve_ai_classifier(settings)

        # ── 금지어 사전 매칭 ───────────────────────────────────────────────
        # 관리자가 직접 등록한 단어가 포함된 경우 local=violation으로 처리합니다.
        # AI가 맥락상 오탐으로 판단하면 combine_decisions에서 번복됩니다.
        _matched_terms = _fast_term_match(message.content, blocked_terms) if blocked_terms else []
        if _matched_terms:
            _top = _matched_terms[0]
            from .filtering import category_label as _cat_label
            local = ModerationDecision(
                violation=True,
                confidence=0.95,
                severity=_top.severity,
                categories=tuple(dict.fromkeys(t.category for t in _matched_terms)),
                matched_terms=tuple(t.term for t in _matched_terms),
                reason=f"등록된 금지어 포함: {', '.join(t.term for t in _matched_terms[:5])}",
                source="local",
                suggested_action="timeout",
            )
        else:
            local = ModerationDecision(violation=False, confidence=0.0, severity=0, source="local", reason="금지어 없음")

        should_call_ai = bool(settings.ai_enabled and classifier is not None)

        LOGGER.debug(
            "[%s] AI: provider=%s batchable=%s should_call=%s ai_enabled=%s",
            message.guild.name,
            getattr(classifier, "provider_name", "none") if classifier else "none",
            batchable,
            should_call_ai,
            settings.ai_enabled,
        )

        # 대화 맥락: AI 호출 예정인 경우에만 직전 메시지를 가져옵니다.
        prior_messages: list[str] = []
        if should_call_ai:
            prior_messages = await self._fetch_prior_messages(message)

        if should_call_ai and batchable:
            await self._queue_for_batch(
                message, settings, local, prior_messages,
                matched_terms=_matched_terms if _matched_terms else None,
            )
            return

        ai_decision = None
        ai_called = False
        ai_error = ""
        if should_call_ai:
            ai_called = True
            provider_name = getattr(classifier, "provider_name", "?")
            LOGGER.info(
                "[%s] AI call (%s): %s#%s | %.80r",
                message.guild.name,
                provider_name,
                message.author,
                getattr(message.channel, "name", message.channel.id),
                message.content,
            )
            context = await self._build_ai_context(
                message.guild.id,
                _matched_terms if _matched_terms else blocked_terms,
                allowed_terms,
                guild_notes=settings.ai_notes,
            )
            ai_decision = await classifier.classify(message.content, context, prior_messages)
            if ai_decision is None:
                ai_error = getattr(classifier, "last_error", "") or "AI returned no decision"
                LOGGER.warning("[%s] AI failure (%s): %s", message.guild.name, provider_name, ai_error)
            else:
                LOGGER.info(
                    "[%s] AI result (%s): violation=%s conf=%.2f severity=%d reason=%.120s",
                    message.guild.name,
                    provider_name,
                    ai_decision.violation,
                    ai_decision.confidence,
                    ai_decision.severity,
                    ai_decision.reason,
                )

        if ai_called:
            self._bump(message.guild.id, "ai_calls")
        if ai_error:
            self._bump(message.guild.id, "ai_failures")
            self._set_last_error(message.guild.id, ai_error)

        # 반복 위반자 동적 임계값: AI 위반 판정인 경우에만 이력 조회
        if ai_decision is not None and ai_decision.violation:
            _recent = await self.bot.store.count_recent_infractions(message.guild.id, message.author.id)
        else:
            _recent = 0
        _t_adj = (-0.05 if _recent >= 2 else 0.0) + (-0.05 if _recent >= 4 else 0.0)
        _thresh = max(0.50, settings.confidence_threshold + _t_adj)

        decision = combine_decisions(local, ai_decision, _thresh)
        LOGGER.debug(
            "[%s] combined: source=%s violation=%s conf=%.2f threshold=%.2f (adj=%.2f recent=%d)",
            message.guild.name,
            decision.source,
            decision.violation,
            decision.confidence,
            _thresh,
            _t_adj,
            _recent,
        )
        await self._finish_moderation(message, decision, settings, _recent=_recent)

    async def _queue_for_batch(
        self,
        message: discord.Message,
        settings: GuildSettings,
        local: ModerationDecision,
        prior_messages: list[str] | None = None,
        matched_terms: list | None = None,
    ) -> None:
        queue = self._groq_queues.setdefault(message.guild.id, [])
        queue.append(
            _PendingClassification(
                message=message,
                settings=settings,
                local=local,
                queued_at=datetime.now(timezone.utc),
                prior_messages=prior_messages or [],
                matched_terms=matched_terms,
            )
        )
        LOGGER.debug(
            "[%s] queued for Groq batch: depth=%d/%d | %s | %.60r",
            message.guild.name,
            len(queue),
            self.bot.config.groq_batch_size,
            message.author,
            message.content,
        )
        if len(queue) >= self.bot.config.groq_batch_size:
            await self._flush_groq_queue(message.guild.id)

    async def _flush_groq_queue(self, guild_id: int, *, force: bool = False) -> None:
        """Pop up to one batch's worth of queued messages and classify them together.

        `force` is used on shutdown to drain whatever is left, even if it's a
        partial batch that hasn't hit the size/time threshold yet.
        """
        lock = self._groq_flush_locks.setdefault(guild_id, asyncio.Lock())
        async with lock:
            queue = self._groq_queues.get(guild_id)
            if not queue:
                return
            now = datetime.now(timezone.utc)
            window = timedelta(seconds=self.bot.config.groq_batch_window_seconds)
            ready = force or len(queue) >= self.bot.config.groq_batch_size or (now - queue[0].queued_at) >= window
            if not ready:
                return

            batch_size = max(1, self.bot.config.groq_batch_size)
            items, remainder = queue[:batch_size], queue[batch_size:]
            queue[:] = remainder
            if not items:
                return

            blocked_terms = await self._cached_blocked_terms(guild_id)
            allowed_terms = await self._cached_allowed_terms(guild_id)

            # Re-resolve against the latest settings: the provider may have
            # changed, or a Groq cooldown may have just kicked in/expired.
            current_settings = await self._cached_settings(guild_id)
            classifier, batchable = self._resolve_ai_classifier(current_settings)
            provider_name = getattr(classifier, "provider_name", "none") if classifier else "none"
            # 각 메시지마다 matched_terms가 다를 수 있으므로 배치 레벨에서는
            # 전체 blocked_terms를 컨텍스트로 사용합니다 (이미 80개 상한 적용).
            context = await self._build_ai_context(
                guild_id, blocked_terms, allowed_terms, guild_notes=current_settings.ai_notes
            )

            LOGGER.info(
                "[guild:%d] flushing batch: %d messages | provider=%s batchable=%s force=%s",
                guild_id,
                len(items),
                provider_name,
                batchable,
                force,
            )

            if classifier is not None and batchable:
                try:
                    batch_prior_messages = [item.prior_messages for item in items]
                    decisions = await classifier.classify_batch(
                        [item.message.content for item in items], context, batch_prior_messages
                    )
                except RateLimitedError as exc:
                    LOGGER.warning(
                        "[guild:%d] all Groq keys rate limited; entering cooldown and falling back to Ollama",
                        guild_id,
                    )
                    self.bot._enter_groq_cooldown(exc.retry_after)
                    await self._finish_batch_with_fallback(items, context)
                    return
                violations = sum(1 for d in decisions if d and d.violation)
                LOGGER.info(
                    "[guild:%d] Groq batch done: %d decisions | %d violations",
                    guild_id,
                    len(decisions),
                    violations,
                )
                for item, ai_decision in zip(items, decisions):
                    await self._complete_classification(item, ai_decision, classifier)
                return

            # Active classifier isn't (or no longer is) Groq batching — e.g. we
            # just entered a cooldown and fell back to Ollama. Process one by one.
            LOGGER.debug("[guild:%d] batch falling back to one-by-one via %s", guild_id, provider_name)
            for item in items:
                ai_decision = (
                    await classifier.classify(item.message.content, context, item.prior_messages)
                    if classifier else None
                )
                await self._complete_classification(item, ai_decision, classifier)

        # 큐가 비어 있으면 Lock 객체도 제거해 메모리 누수를 방지합니다.
        if not self._groq_queues.get(guild_id):
            self._groq_queues.pop(guild_id, None)
            self._groq_flush_locks.pop(guild_id, None)

    async def _finish_batch_with_fallback(
        self, items: list[_PendingClassification], context: AIContext
    ) -> None:
        fallback = self.bot._groq_fallback_classifier()
        LOGGER.info(
            "Groq rate-limit fallback: processing %d messages via %s",
            len(items),
            getattr(fallback, "provider_name", "none") if fallback else "none (no fallback)",
        )
        for item in items:
            ai_decision = None
            if fallback is not None:
                try:
                    ai_decision = await fallback.classify(item.message.content, context, item.prior_messages)
                except Exception:
                    LOGGER.exception("Groq cooldown fallback classification failed")
            await self._complete_classification(item, ai_decision, fallback)

    async def _complete_classification(
        self,
        item: _PendingClassification,
        ai_decision: ModerationDecision | None,
        classifier,
    ) -> None:
        guild_id = item.message.guild.id if item.message.guild else None
        if guild_id is None:
            return

        ai_called = classifier is not None
        provider_name = getattr(classifier, "provider_name", "?") if classifier else "none"
        if ai_called:
            self._bump(guild_id, "ai_calls")
            if ai_decision is None:
                ai_error = getattr(classifier, "last_error", "") or "AI returned no decision"
                self._bump(guild_id, "ai_failures")
                self._set_last_error(guild_id, ai_error)
                LOGGER.warning(
                    "[guild:%d] AI failure (%s): %s | %.60r",
                    guild_id,
                    provider_name,
                    ai_error,
                    item.message.content,
                )
            else:
                LOGGER.info(
                    "[guild:%d] AI result (%s): violation=%s conf=%.2f severity=%d reason=%.120s",
                    guild_id,
                    provider_name,
                    ai_decision.violation,
                    ai_decision.confidence,
                    ai_decision.severity,
                    ai_decision.reason,
                )

        # 반복 위반자 동적 임계값: AI 위반 판정인 경우에만 이력 조회
        if ai_decision is not None and ai_decision.violation:
            _recent = await self.bot.store.count_recent_infractions(guild_id, item.message.author.id)
        else:
            _recent = 0
        _t_adj = (-0.05 if _recent >= 2 else 0.0) + (-0.05 if _recent >= 4 else 0.0)
        _thresh = max(0.50, item.settings.confidence_threshold + _t_adj)

        decision = combine_decisions(item.local, ai_decision, _thresh)
        LOGGER.debug(
            "[guild:%d] combined: source=%s violation=%s conf=%.2f threshold=%.2f (adj=%.2f recent=%d)",
            guild_id,
            decision.source,
            decision.violation,
            decision.confidence,
            _thresh,
            _t_adj,
            _recent,
        )
        await self._finish_moderation(item.message, decision, item.settings, _recent=_recent)

    async def _finish_moderation(
        self,
        message: discord.Message,
        decision: ModerationDecision,
        settings: GuildSettings,
        _recent: int = -1,
    ) -> None:
        if not decision.violation:
            LOGGER.debug(
                "[%s] pass (no violation): source=%s conf=%.2f | %s | %.60r",
                message.guild.name,
                decision.source,
                decision.confidence,
                message.author,
                message.content,
            )
            return

        # 호출자에서 미리 조회한 경우 재사용, 아니면 직접 조회합니다.
        # (동적 임계값은 이미 combine_decisions 호출 전에 적용됨)
        recent = (
            _recent if _recent >= 0
            else await self.bot.store.count_recent_infractions(message.guild.id, message.author.id)
        )

        fingerprint = message_fingerprint(message.content)
        self._bump(message.guild.id, "violations")
        timeout_minutes = self._effective_timeout(settings, recent)
        action_parts: list[str] = []

        if settings.dry_run:
            action_parts.append("dry-run")
        else:
            if settings.delete_messages:
                try:
                    await message.delete()
                    action_parts.append("deleted")
                    self._bump(message.guild.id, "deleted")
                except discord.NotFound:
                    action_parts.append("already deleted")
                except discord.Forbidden:
                    action_parts.append("delete failed: missing permission")
                    self._bump(message.guild.id, "delete_failures")
                except discord.HTTPException:
                    LOGGER.exception("Failed to delete message %s", message.id)
                    action_parts.append("delete failed")
                    self._bump(message.guild.id, "delete_failures")
                except TypeError:
                    LOGGER.exception("Failed to delete message %s", message.id)
                    action_parts.append("delete failed: incompatible discord library")
                    self._bump(message.guild.id, "delete_failures")

            if timeout_minutes > 0 and isinstance(message.author, discord.Member):
                timeout_action = await self._timeout_member(message.author, timeout_minutes, decision.reason)
                action_parts.append(timeout_action)
                if timeout_action.startswith("timeout "):
                    self._bump(message.guild.id, "timeouts")
                elif timeout_action.startswith("timeout failed"):
                    self._bump(message.guild.id, "timeout_failures")

        action = ", ".join(action_parts) or "logged"
        LOGGER.info(
            "[%s] VIOLATION: %s | source=%s conf=%.2f severity=%d | action=%s | %.80r",
            message.guild.name,
            message.author,
            decision.source,
            decision.confidence,
            decision.severity,
            action,
            message.content,
        )
        infraction_id = await self.bot.store.create_infraction(
            guild_id=message.guild.id,
            channel_id=message.channel.id,
            message_id=message.id,
            user_id=message.author.id,
            username=str(message.author),
            content=message.content,
            normalized_hash=fingerprint,
            decision=decision,
            action=action,
            timeout_minutes=timeout_minutes,
        )

        if decision.source != "local" and decision.confidence >= 0.9:
            await self.bot.store.add_learning_event(
                guild_id=message.guild.id,
                label="auto_flagged",
                source_type="infraction",
                source_id=infraction_id,
                content=message.content,
                created_by=self.bot.user.id if self.bot.user else None,
            )
            await self._auto_register_terms(message.guild.id, decision, infraction_id)

        await self._log_infraction(message, infraction_id, decision, action, timeout_minutes, settings)
        if settings.dm_users and not settings.dry_run:
            await self._dm_warning(message.author, infraction_id, timeout_minutes, decision.reason, guild_id=message.guild.id)

    def _mask_member_names(self, message: discord.Message) -> str:
        """Return message content with all known server member names replaced by spaces.

        This prevents usernames/nicknames that happen to contain blocked words
        from triggering false positives in the local keyword filter.
        """
        if not message.guild:
            return message.content

        # Collect names from cached members + message author + mentioned users.
        members: list = list(message.guild.members) if message.guild.members else []
        members.append(message.author)
        members.extend(message.mentions)

        names: set[str] = set()
        for m in members:
            for attr in ("display_name", "global_name", "name"):
                val = getattr(m, attr, None)
                if val and len(val) >= 2:
                    names.add(val)

        result = message.content
        # Replace longest names first to avoid partial-match issues.
        for name in sorted(names, key=len, reverse=True):
            result = re.sub(re.escape(name), " ", result, flags=re.IGNORECASE)

        if result != message.content:
            LOGGER.debug(
                "[%s] name-masked for local filter: %.80r → %.80r",
                message.guild.name,
                message.content,
                result,
            )
        return result

    async def _auto_register_terms(
        self,
        guild_id: int,
        decision: ModerationDecision,
        infraction_id: int,
    ) -> None:
        """Auto-add AI-confirmed matched terms to the guild's blocked_terms table.

        Only runs when the AI had high confidence (≥ 0.9) and the decision came
        from an AI source.  Terms already in DEFAULT_BLOCKED_TERMS or already
        registered for this guild are skipped.
        """
        if not decision.matched_terms:
            return

        default_terms = {term.lower() for term, _ in DEFAULT_BLOCKED_TERMS}
        existing = {t.term.lower() for t in await self.bot.store.list_blocked_terms(guild_id)}

        added: list[str] = []
        for raw in decision.matched_terms:
            term = raw.strip()
            # ── 유효성 검사 ────────────────────────────────────────────────────
            # 1) 공백 포함: AI가 reason 문장이나 구절을 matched_terms에 넣는 경우 차단
            if " " in term or "." in term:
                LOGGER.debug("[guild:%d] auto-register skip (contains space/period): %r", guild_id, term)
                continue
            # 2) 너무 긴 문자열: 단어가 아닌 문장이 들어온 경우 차단 (최대 20자)
            if len(term) > 20:
                LOGGER.debug("[guild:%d] auto-register skip (too long): %r", guild_id, term)
                continue
            # 3) 2음절 이하 단어는 자동 등록 안 함 — 한국어는 짧은 단어가
            #    동사 어간·어미에 묻혀 오탐이 너무 많음 (예: "자지" → "자지마").
            #    단, 한글 자모만으로 이루어진 표현(ㅅㅂ, ㅂㅅ 등)은 길이 무관 등록 허용.
            _is_jamo_only = bool(term) and all('\u3130' <= ch <= '\u318f' for ch in term)
            if not term or (len(term) < 3 and not _is_jamo_only):
                continue
            # 4) 카테고리명·메타 텍스트 차단 (AI가 카테고리를 그대로 matched_terms에 넣는 경우)
            _meta_terms = {"profanity", "sexual", "family_insult", "harassment", "hate", "threat",
                           "other", "욕설", "패드립", "성적발언", "괴롭힘", "혐오", "위협", "욕설 사용"}
            if term.lower() in _meta_terms:
                LOGGER.debug("[guild:%d] auto-register skip (meta/category name): %r", guild_id, term)
                continue
            if term.lower() in default_terms or term.lower() in existing:
                continue

            severity = min(3, max(1, decision.severity))
            category = decision.categories[0] if decision.categories else "profanity"
            await self.bot.store.add_blocked_term(
                guild_id,
                term,
                severity,
                added_by=self.bot.user.id if self.bot.user else 0,
                notes=f"auto from infraction #{infraction_id} (conf={decision.confidence:.2f})",
                category=category,
            )
            await self.bot.store.add_learning_event(
                guild_id=guild_id,
                label="confirmed_bad",
                source_type="infraction",
                source_id=infraction_id,
                content=term,
                term=term,
                created_by=self.bot.user.id if self.bot.user else None,
            )
            existing.add(term.lower())
            added.append(term)

        if added:
            LOGGER.info(
                "[guild:%d] auto-registered %d term(s) from infraction #%d: %s",
                guild_id,
                len(added),
                infraction_id,
                added,
            )

    def _effective_timeout(self, settings: GuildSettings, recent_count: int) -> int:
        base = max(0, min(MAX_TIMEOUT_MINUTES, settings.timeout_minutes))
        if base == 0:
            return 0
        if not settings.escalate:
            return base
        multiplier = min(16, 2**min(recent_count, 4))
        return min(MAX_TIMEOUT_MINUTES, base * multiplier)

    async def _timeout_member(self, member: discord.Member, minutes: int, reason: str) -> str:
        reason_text = f"PrettyWords 비속어 필터: {reason[:120]}"
        try:
            await member.timeout(timedelta(minutes=minutes), reason=reason_text)
            return f"timeout {minutes}m"
        except AttributeError:
            until = discord.utils.utcnow() + timedelta(minutes=minutes)
            try:
                await member.edit(timed_out_until=until, reason=reason_text)
                return f"timeout {minutes}m"
            except discord.Forbidden:
                return "timeout failed: missing permission or role hierarchy"
            except discord.HTTPException:
                LOGGER.exception("Failed to timeout member %s", member.id)
                return "timeout failed"
        except discord.Forbidden:
            return "timeout failed: missing permission or role hierarchy"
        except discord.HTTPException:
            LOGGER.exception("Failed to timeout member %s", member.id)
            return "timeout failed"

    async def _log_infraction(
        self,
        message: discord.Message,
        infraction_id: int,
        decision: ModerationDecision,
        action: str,
        timeout_minutes: int,
        settings: GuildSettings,
    ) -> None:
        if not settings.log_channel_id:
            return
        channel = message.guild.get_channel(settings.log_channel_id)
        if not isinstance(channel, discord.abc.Messageable):
            return

        embed = discord.Embed(
            title=f"PrettyWords 제재 기록 #{infraction_id}",
            color=discord.Color.orange(),
            description=decision.reason[:350] or "정책 위반 가능성이 감지되었습니다.",
        )
        embed.add_field(
            name="사용자",
            value=f"{message.author.mention} {message.author} (`{message.author.id}`)",
            inline=False,
        )
        embed.add_field(name="채널", value=message.channel.mention, inline=True)
        embed.add_field(name="조치", value=_action_label(action), inline=True)
        embed.add_field(name="타임아웃", value=f"{timeout_minutes}분", inline=True)
        embed.add_field(name="확신도", value=f"{decision.confidence:.2f}", inline=True)
        embed.add_field(name="심각도", value=str(decision.severity), inline=True)
        embed.add_field(name="판정 출처", value=_source_label(decision.source), inline=True)
        embed.add_field(name="메시지 링크", value=f"[바로가기]({message.jump_url})", inline=False)
        if decision.categories:
            embed.add_field(
                name="카테고리",
                value=", ".join(category_label(category) for category in decision.categories)[:250],
                inline=False,
            )
        if decision.matched_terms:
            embed.add_field(name="감지된 표현", value=", ".join(decision.matched_terms)[:250], inline=False)
        embed.add_field(name="메시지", value=message.content[:900] or "(비어 있음)", inline=False)
        embed.set_footer(text="/filter report 또는 /filter resolve-report 로 검토")

        try:
            await channel.send(embed=embed)
        except discord.HTTPException:
            LOGGER.exception("Failed to send infraction log")

    async def _log_admin_event(self, guild: discord.Guild, title: str, body: str) -> None:
        settings = await self.bot.store.get_settings(guild.id)
        if not settings.log_channel_id:
            return
        channel = guild.get_channel(settings.log_channel_id)
        if not isinstance(channel, discord.abc.Messageable):
            return
        embed = discord.Embed(title=title, description=body[:1000], color=discord.Color.blurple())
        try:
            await channel.send(embed=embed)
        except discord.HTTPException:
            LOGGER.exception("Failed to send admin log")

    async def _dm_warning(
        self,
        user: discord.User | discord.Member,
        infraction_id: int,
        timeout_minutes: int,
        reason: str,
        guild_id: int | None = None,
    ) -> None:
        text = (
            f"**PrettyWords 제재 기록 #{infraction_id}**: 서버 규칙 위반 가능성이 감지되었습니다.\n"
            f"타임아웃: **{timeout_minutes}분** | 사유: {reason[:250]}\n"
            "오탐이라고 생각하면 아래 버튼으로 이의제기하세요. 타임아웃 중에도 이 DM에서 이의제기할 수 있습니다."
        )
        try:
            if guild_id:
                await user.send(text, view=_AppealView(guild_id, infraction_id))
            else:
                await user.send(text)
        except discord.HTTPException:
            return

    @filter.command(name="status", description="현재 필터 설정을 봅니다")
    @mod_only()
    async def status(self, interaction: discord.Interaction) -> None:
        settings = await self.bot.store.get_settings(interaction.guild_id)
        disabled = await self.bot.store.list_disabled_channels(interaction.guild_id)
        blocked = await self.bot.store.list_blocked_terms(interaction.guild_id)
        allowed = await self.bot.store.list_allowed_terms(interaction.guild_id)
        config_admins = await self.bot.store.list_config_admins(interaction.guild_id)
        embed = discord.Embed(title="PrettyWords 상태", color=discord.Color.green())
        embed.add_field(name="일시정지", value=_enabled_label(settings.paused), inline=True)
        _provider, _model, scan_all = self.bot._effective_ai_settings(settings)
        embed.add_field(name="AI", value=self.bot._ai_label(settings) if settings.ai_enabled else "꺼짐", inline=True)
        embed.add_field(name="전체 AI 검사", value=_enabled_label(scan_all), inline=True)
        embed.add_field(name="상태 로그", value=_enabled_label(settings.health_log_enabled), inline=True)
        embed.add_field(name="모의 실행", value=_enabled_label(settings.dry_run), inline=True)
        embed.add_field(name="타임아웃", value=f"{settings.timeout_minutes}분", inline=True)
        embed.add_field(name="확신도 기준", value=f"{settings.confidence_threshold:.2f}", inline=True)
        embed.add_field(name="반복 위반 가중", value=_enabled_label(settings.escalate), inline=True)
        embed.add_field(name="제재/신고 로그 채널", value=f"<#{settings.log_channel_id}>" if settings.log_channel_id else "미설정", inline=False)
        health_channel_value = (
            f"<#{settings.health_log_channel_id}>"
            if settings.health_log_channel_id
            else (f"제재/신고 로그 채널 사용 (<#{settings.log_channel_id}>)" if settings.log_channel_id else "미설정")
        )
        embed.add_field(
            name="상태 로그 채널",
            value=health_channel_value,
            inline=False,
        )
        embed.add_field(name="필터 비활성 채널", value=", ".join(f"<#{cid}>" for cid in disabled) or "없음", inline=False)
        embed.add_field(name="서버 금지어", value=str(len(blocked)), inline=True)
        embed.add_field(name="허용어", value=str(len(allowed)), inline=True)
        embed.add_field(name="설정 관리자", value=str(len(config_admins)), inline=True)
        if settings.appeal_channel_id:
            appeal_ch = interaction.guild.get_channel(settings.appeal_channel_id)
            embed.add_field(name="이의제기 채널", value=appeal_ch.mention if appeal_ch else f"<#{settings.appeal_channel_id}>", inline=False)
        if settings.ai_notes:
            embed.add_field(name="AI 메모", value=settings.ai_notes[:500], inline=False)
        await send_interaction(interaction, embed=embed)

    @filter.command(name="ai-note", description="AI에게 서버 특성을 알려주는 메모를 설정합니다")
    @mod_only()
    @app_commands.describe(note="서버 특성 메모 (예: '게임 서버입니다. PvP 표현은 욕설이 아닌 경우가 많습니다.')  비워두면 삭제.")
    async def ai_note(self, interaction: discord.Interaction, note: Optional[str] = None) -> None:
        note_text = (note or "").strip()
        await self.bot.store.update_settings(interaction.guild_id, ai_notes=note_text)
        self._cache.invalidate_guild(interaction.guild_id)
        if note_text:
            await send_interaction(interaction, f"AI 메모 설정됨:\n> {note_text[:500]}")
        else:
            await send_interaction(interaction, "AI 메모 삭제됨")
        await self._log_admin_event(
            interaction.guild,
            "AI 메모 변경",
            f"{interaction.user}님이 AI 메모를 {'설정' if note_text else '삭제'}했습니다."
            + (f"\n> {note_text[:200]}" if note_text else ""),
        )

    @admin.command(name="config-admin-add", description="PrettyWords 설정 가능 관리자 ID를 추가합니다")
    @mod_only()
    @app_commands.describe(user_id="Discord 사용자 ID 또는 멘션")
    async def config_admin_add(self, interaction: discord.Interaction, user_id: str) -> None:
        try:
            parsed_user_id = _parse_discord_id(user_id)
        except ValueError:
            await send_interaction(interaction, "유효한 Discord ID 필요.")
            return

        await self.bot.store.add_config_admin(interaction.guild_id, parsed_user_id, interaction.user.id)
        await send_interaction(interaction, f"설정 관리자 추가됨: `{parsed_user_id}`")
        await self._log_admin_event(
            interaction.guild,
            "설정 관리자 추가",
            f"{interaction.user}님이 `{parsed_user_id}`를 설정 관리자로 추가했습니다.",
        )

    @admin.command(name="config-admin-remove", description="PrettyWords 설정 관리자 ID를 제거합니다")
    @mod_only()
    @app_commands.describe(user_id="Discord 사용자 ID 또는 멘션")
    async def config_admin_remove(self, interaction: discord.Interaction, user_id: str) -> None:
        try:
            parsed_user_id = _parse_discord_id(user_id)
        except ValueError:
            await send_interaction(interaction, "유효한 Discord ID 필요.")
            return

        count = await self.bot.store.remove_config_admin(interaction.guild_id, parsed_user_id)
        await send_interaction(interaction, "설정 관리자 제거됨" if count else "등록된 설정 관리자 없음")
        if count:
            await self._log_admin_event(
                interaction.guild,
                "설정 관리자 제거",
                f"{interaction.user}님이 `{parsed_user_id}`를 설정 관리자에서 제거했습니다.",
            )

    @admin.command(name="config-admin-list", description="PrettyWords 설정 관리자 ID 목록을 봅니다")
    @mod_only()
    async def config_admin_list(self, interaction: discord.Interaction) -> None:
        admins = await self.bot.store.list_config_admins(interaction.guild_id)
        global_admins = sorted(self.bot.config.bot_admin_ids)
        lines = []
        if admins:
            lines.append("서버 설정: " + ", ".join(f"`{admin_id}`" for admin_id in admins))
        if global_admins:
            lines.append(".env: " + ", ".join(f"`{admin_id}`" for admin_id in global_admins))
        await send_interaction(interaction, "\n".join(lines) if lines else "설정 관리자 ID 없음")

    @filter.command(name="set-channel", description="로그/상태/이의제기 채널을 설정합니다")
    @mod_only()
    @app_commands.describe(
        channel_type="채널 종류: log=제재로그, health=상태로그, appeal=이의제기",
        channel="대상 채널",
    )
    @app_commands.choices(channel_type=[
        app_commands.Choice(name="log — 제재/신고 로그", value="log"),
        app_commands.Choice(name="health — 상태 로그", value="health"),
        app_commands.Choice(name="appeal — 이의제기", value="appeal"),
    ])
    async def set_channel(self, interaction: discord.Interaction, channel_type: str, channel: discord.TextChannel) -> None:
        if channel_type == "log":
            settings = await self.bot.store.update_settings(interaction.guild_id, log_channel_id=channel.id)
            self._cache.invalidate_guild(interaction.guild_id)
            await send_interaction(interaction, f"제재/신고 로그 채널 설정됨: {channel.mention}")
            await self._log_admin_event(interaction.guild, "로그 채널 변경", f"{interaction.user}님이 로그 채널을 {channel.mention}로 설정했습니다.")
        elif channel_type == "health":
            settings = await self.bot.store.update_settings(interaction.guild_id, health_log_channel_id=channel.id)
            self._cache.invalidate_guild(interaction.guild_id)
            await send_interaction(interaction, f"상태 로그 채널 설정됨: {channel.mention}")
            await self._send_health_log(interaction.guild, settings, title="PrettyWords 상태 로그 채널 연결됨")
        elif channel_type == "appeal":
            await self.bot.store.update_settings(interaction.guild_id, appeal_channel_id=channel.id)
            self._cache.invalidate_guild(interaction.guild_id)
            await send_interaction(interaction, f"이의제기 채널 설정됨: {channel.mention}")
            await self._log_admin_event(interaction.guild, "이의제기 채널 변경", f"{interaction.user}님이 이의제기 채널을 {channel.mention}로 설정했습니다.")
        else:
            await send_interaction(interaction, "알 수 없는 채널 종류입니다.")

    @filter.command(name="health", description="현재 PrettyWords 상태를 로그 채널로 보냅니다")
    @mod_only()
    async def health(self, interaction: discord.Interaction) -> None:
        settings = await self.bot.store.get_settings(interaction.guild_id)
        if not (settings.health_log_channel_id or settings.log_channel_id):
            await send_interaction(interaction, "상태 로그 채널 먼저 설정 필요: /filter health-log-channel")
            return
        await self._send_health_log(interaction.guild, settings)
        await send_interaction(interaction, "상태 로그 전송됨")

    @filter.command(name="health-log", description="주기적인 상태 로그를 켜거나 끕니다")
    @mod_only()
    async def health_log(self, interaction: discord.Interaction, enabled: bool) -> None:
        settings = await self.bot.store.update_settings(interaction.guild_id, health_log_enabled=enabled)
        self._cache.invalidate_guild(interaction.guild_id)
        await send_interaction(interaction, f"상태 로그: {_enabled_label(enabled)}")
        if enabled and (settings.health_log_channel_id or settings.log_channel_id):
            await self._send_health_log(interaction.guild, settings, title="PrettyWords 상태 로그 켜짐")

    @filter.command(name="timeout", description="비속어 사용 시 타임아웃 시간을 설정합니다")
    @mod_only()
    @app_commands.describe(minutes="0이면 타임아웃 없이 삭제/로그만 합니다")
    async def timeout(self, interaction: discord.Interaction, minutes: app_commands.Range[int, 0, MAX_TIMEOUT_MINUTES]) -> None:
        await self.bot.store.update_settings(interaction.guild_id, timeout_minutes=int(minutes))
        self._cache.invalidate_guild(interaction.guild_id)
        await send_interaction(interaction, f"타임아웃 기본값: {minutes}분")
        await self._log_admin_event(interaction.guild, "타임아웃 변경", f"{interaction.user}님이 기본 타임아웃을 {minutes}분으로 설정했습니다.")

    @filter.command(name="threshold", description="AI/필터 확신도 기준을 설정합니다")
    @mod_only()
    async def threshold(self, interaction: discord.Interaction, confidence: app_commands.Range[float, 0.1, 0.99]) -> None:
        await self.bot.store.update_settings(interaction.guild_id, confidence_threshold=float(confidence))
        self._cache.invalidate_guild(interaction.guild_id)
        await send_interaction(interaction, f"확신도 기준: {confidence:.2f}")

    @filter.command(name="mode", description="삭제/DM/모의 실행/AI/반복 위반 가중 설정을 바꿉니다")
    @mod_only()
    async def mode(
        self,
        interaction: discord.Interaction,
        delete_messages: Optional[bool] = None,
        dm_users: Optional[bool] = None,
        dry_run: Optional[bool] = None,
        ai_enabled: Optional[bool] = None,
        escalate: Optional[bool] = None,
    ) -> None:
        updates = {
            key: value
            for key, value in {
                "delete_messages": delete_messages,
                "dm_users": dm_users,
                "dry_run": dry_run,
                "ai_enabled": ai_enabled,
                "escalate": escalate,
            }.items()
            if value is not None
        }
        settings = await self.bot.store.update_settings(interaction.guild_id, **updates)
        self._cache.invalidate_guild(interaction.guild_id)
        await send_interaction(
            interaction,
            (
                f"모드 업데이트: 삭제={_enabled_label(settings.delete_messages)}, DM={_enabled_label(settings.dm_users)}, "
                f"모의 실행={_enabled_label(settings.dry_run)}, AI={_enabled_label(settings.ai_enabled)}, "
                f"반복 위반 가중={_enabled_label(settings.escalate)}"
            ),
        )

    @filter.command(name="ai", description="AI 제공자/모델 설정을 바꿉니다")
    @mod_only()
    @app_commands.choices(
        provider=[
            app_commands.Choice(name="기본값(.env)", value="default"),
            app_commands.Choice(name="ollama", value="ollama"),
            app_commands.Choice(name="openai", value="openai"),
            app_commands.Choice(name="groq", value="groq"),
            app_commands.Choice(name="사용 안 함", value="none"),
            app_commands.Choice(name="자동", value="auto"),
        ]
    )
    @app_commands.describe(
        model="예: qwen3:4b, qwen3:1.7b, default",
        scan_all="false면 의심 메시지만 AI로 보냅니다",
    )
    async def ai(
        self,
        interaction: discord.Interaction,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        scan_all: Optional[bool] = None,
    ) -> None:
        updates = {}
        if provider is not None:
            updates["ai_provider"] = "" if provider == "default" else provider
        if model is not None:
            clean_model = model.strip()
            updates["ai_model"] = "" if clean_model.lower() in {"default", "env", "reset"} else clean_model
        if scan_all is not None:
            updates["ai_scan_all"] = scan_all

        settings = await self.bot.store.update_settings(interaction.guild_id, **updates)
        self._cache.invalidate_guild(interaction.guild_id)
        _provider, _model, effective_scan_all = self.bot._effective_ai_settings(settings)
        await send_interaction(
            interaction,
            f"AI: {self.bot._ai_label(settings)}, 전체 AI 검사={_enabled_label(effective_scan_all)}",
        )

    @filter.command(name="ai-reset", description="AI 설정을 .env 기본값으로 되돌립니다")
    @mod_only()
    async def ai_reset(self, interaction: discord.Interaction) -> None:
        settings = await self.bot.store.update_settings(
            interaction.guild_id,
            ai_provider="",
            ai_model="",
            ai_scan_all=None,
        )
        self._cache.invalidate_guild(interaction.guild_id)
        _provider, _model, effective_scan_all = self.bot._effective_ai_settings(settings)
        await send_interaction(
            interaction,
            f"AI 설정 초기화됨: {self.bot._ai_label(settings)}, 전체 AI 검사={_enabled_label(effective_scan_all)}",
        )

    @filter.command(name="pause", description="필터를 일시정지합니다")
    @mod_only()
    async def pause(self, interaction: discord.Interaction, reason: Optional[str] = None) -> None:
        await self.bot.store.update_settings(interaction.guild_id, paused=True)
        self._cache.invalidate_guild(interaction.guild_id)
        await send_interaction(interaction, "필터 일시정지됨")
        await self._log_admin_event(interaction.guild, "필터 일시정지", f"{interaction.user}: {reason or '사유 없음'}")

    @filter.command(name="resume", description="필터를 다시 시작합니다")
    @mod_only()
    async def resume(self, interaction: discord.Interaction) -> None:
        await self.bot.store.update_settings(interaction.guild_id, paused=False)
        self._cache.invalidate_guild(interaction.guild_id)
        await send_interaction(interaction, "필터 다시 시작됨")
        await self._log_admin_event(interaction.guild, "필터 다시 시작", f"{interaction.user}님이 필터를 다시 시작했습니다.")

    @filter.command(name="channel", description="특정 채널에서 필터를 켜거나 끕니다")
    @mod_only()
    @app_commands.describe(channel="대상 채널", enabled="True=필터 켜기, False=필터 끄기")
    async def set_channel(
        self,
        interaction: discord.Interaction,
        channel: discord.TextChannel,
        enabled: bool,
    ) -> None:
        await self.bot.store.set_channel_disabled(interaction.guild_id, channel.id, interaction.user.id, not enabled)
        self._cache.invalidate_guild(interaction.guild_id)
        label = "활성화됨" if enabled else "비활성화됨"
        await send_interaction(interaction, f"{label}: {channel.mention}")

    @filter.command(name="add-word", description="서버 전용 비속어/금지어를 등록합니다")
    @mod_only()
    @app_commands.choices(category=CATEGORY_CHOICES)
    async def add_word(
        self,
        interaction: discord.Interaction,
        term: str,
        severity: app_commands.Range[int, 1, 3] = 2,
        category: str = "profanity",
        notes: Optional[str] = None,
    ) -> None:
        normalized_category = normalize_category(category)
        await self.bot.store.add_blocked_term(
            interaction.guild_id,
            term,
            int(severity),
            interaction.user.id,
            notes or "",
            category=normalized_category,
        )
        await self.bot.store.add_learning_event(
            guild_id=interaction.guild_id,
            label="confirmed_bad",
            source_type="manual_term",
            content=term,
            term=term,
            category=normalized_category,
            created_by=interaction.user.id,
        )
        self._cache.invalidate_guild(interaction.guild_id)
        await send_interaction(
            interaction,
            f"등록됨: `{term}` 카테고리={category_label(normalized_category)} 심각도={severity}",
        )

    @filter.command(name="remove-word", description="서버 전용 금지어를 제거합니다")
    @mod_only()
    async def remove_word(self, interaction: discord.Interaction, term: str) -> None:
        count = await self.bot.store.remove_blocked_term(interaction.guild_id, term)
        self._cache.invalidate_guild(interaction.guild_id)
        await send_interaction(interaction, "제거됨" if count else "등록된 항목 없음")

    @filter.command(name="learn-message", description="메시지 ID와 욕설 구간을 카테고리 학습 데이터로 등록합니다")
    @mod_only()
    @app_commands.choices(category=CATEGORY_CHOICES)
    @app_commands.describe(
        message_id="학습할 Discord 메시지 ID",
        term="메시지 안에서 비속어인 부분",
        category="비속어 카테고리",
        channel="메시지가 있는 채널. 비우면 현재 채널",
    )
    async def learn_message(
        self,
        interaction: discord.Interaction,
        message_id: str,
        term: str,
        category: str,
        severity: app_commands.Range[int, 1, 3] = 2,
        channel: Optional[discord.TextChannel] = None,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            parsed_message_id = _parse_discord_id(message_id)
        except ValueError:
            await interaction.followup.send("유효한 메시지 ID 필요.", ephemeral=True)
            return

        target_channel = channel or interaction.channel
        if not hasattr(target_channel, "fetch_message"):
            await interaction.followup.send("메시지를 가져올 수 있는 텍스트 채널 필요.", ephemeral=True)
            return

        try:
            target_message = await target_channel.fetch_message(parsed_message_id)
        except discord.NotFound:
            await interaction.followup.send("메시지를 찾을 수 없음.", ephemeral=True)
            return
        except discord.Forbidden:
            await interaction.followup.send("해당 채널 메시지 읽기 권한 없음.", ephemeral=True)
            return
        except discord.HTTPException:
            LOGGER.exception("Failed to fetch message for learning")
            await interaction.followup.send("메시지 조회 실패. 로그 확인 필요.", ephemeral=True)
            return

        normalized_category = normalize_category(category)
        term_found = compact_text(term) in compact_text(target_message.content)
        await self.bot.store.add_blocked_term(
            interaction.guild_id,
            term,
            int(severity),
            interaction.user.id,
            f"메시지 {target_message.id}에서 학습",
            category=normalized_category,
        )
        await self.bot.store.add_learning_event(
            guild_id=interaction.guild_id,
            label="confirmed_bad",
            source_type="message",
            source_id=target_message.id,
            content=target_message.content,
            term=term,
            category=normalized_category,
            created_by=interaction.user.id,
        )

        self._cache.invalidate_guild(interaction.guild_id)
        note = "" if term_found else "\n주의: 등록 표현이 메시지에 정확히 포함되지는 않음. 우회표현이면 정상일 수 있음."
        await interaction.followup.send(
            (
                f"학습됨: `{term}` → {category_label(normalized_category)} 심각도={severity}\n"
                f"메시지: {target_message.jump_url}{note}"
            ),
            ephemeral=True,
        )
        await self._log_admin_event(
            interaction.guild,
            "메시지 학습",
            (
                f"{interaction.user}님이 [메시지]({target_message.jump_url})의 `{term}`을 "
                f"{category_label(normalized_category)} 카테고리로 학습시켰습니다."
            ),
        )

    @filter.command(name="allow-word", description="오탐 방지용 허용어/문구를 등록합니다")
    @mod_only()
    async def allow_word(self, interaction: discord.Interaction, term: str) -> None:
        await self.bot.store.add_allowed_term(interaction.guild_id, term, interaction.user.id)
        self._cache.invalidate_guild(interaction.guild_id)
        await send_interaction(interaction, f"허용됨: `{term}`")

    @filter.command(name="remove-allow", description="허용어/문구를 제거합니다")
    @mod_only()
    async def remove_allow(self, interaction: discord.Interaction, term: str) -> None:
        count = await self.bot.store.remove_allowed_term(interaction.guild_id, term)
        self._cache.invalidate_guild(interaction.guild_id)
        await send_interaction(interaction, "제거됨" if count else "등록된 항목 없음")

    @filter.command(name="report", description="부적절한 제재/오탐을 신고합니다")
    @app_commands.describe(case_id="로그나 DM에 표시된 PrettyWords 제재 기록 ID")
    async def report(
        self,
        interaction: discord.Interaction,
        reason: str,
        case_id: Optional[int] = None,
        message_id: Optional[str] = None,
    ) -> None:
        if interaction.guild_id is None:
            await send_interaction(interaction, "서버에서만 사용 가능")
            return
        parsed_message_id = int(message_id) if message_id and message_id.isdigit() else None
        report_id = await self.bot.store.create_report(
            interaction.guild_id,
            interaction.user.id,
            reason,
            infraction_id=case_id,
            message_id=parsed_message_id,
        )
        await send_interaction(interaction, f"이의제기/신고 접수됨: 신고 #{report_id}. 관리자 승인 후 학습됩니다.")
        await self._log_report(interaction, report_id, case_id, reason)

    async def _log_report(
        self,
        interaction: discord.Interaction,
        report_id: int,
        case_id: int | None,
        reason: str,
    ) -> None:
        settings = await self.bot.store.get_settings(interaction.guild_id)
        target_ch_id = settings.appeal_channel_id or settings.log_channel_id
        if not target_ch_id:
            return
        channel = interaction.guild.get_channel(target_ch_id)
        if not isinstance(channel, discord.abc.Messageable):
            return
        appeal_embed = discord.Embed(
            title=f"✋ 이의제기 #{report_id}",
            description=reason[:900],
            color=discord.Color.orange(),
        )
        appeal_embed.add_field(name="신청자", value=f"{interaction.user.mention} {interaction.user} (`{interaction.user.id}`)", inline=False)
        embeds = [appeal_embed]
        if case_id:
            appeal_embed.add_field(name="제재 기록 ID", value=f"#{case_id}", inline=True)
            infraction = await self.bot.store.get_infraction(interaction.guild_id, case_id)
            if infraction:
                import json as _json
                try:
                    dec = _json.loads(infraction.decision_json)
                except Exception:
                    dec = {}
                inf_embed = discord.Embed(
                    title=f"📋 원본 제재 기록 #{case_id}",
                    color=discord.Color.red(),
                )
                inf_embed.add_field(name="대상 메시지", value=f"```{infraction.content[:900]}```", inline=False)
                inf_embed.add_field(
                    name="AI 판정",
                    value=(
                        f"위반: {dec.get('violation', '?')} | "
                        f"확신도: {dec.get('confidence', 0):.2f} | "
                        f"심각도: {dec.get('severity', '?')}"
                    ),
                    inline=False,
                )
                if dec.get("categories"):
                    from .filtering import category_label as _cat_label
                    cats = ", ".join(_cat_label(c) for c in dec["categories"])
                    inf_embed.add_field(name="카테고리", value=cats, inline=True)
                if dec.get("reason"):
                    inf_embed.add_field(name="AI 이유", value=dec["reason"][:300], inline=False)
                ch = interaction.guild.get_channel(infraction.channel_id)
                if ch:
                    inf_embed.add_field(name="채널", value=ch.mention, inline=True)
                embeds.append(inf_embed)
        appeal_embed.set_footer(text="아래 버튼 또는 /filter resolve-report 로 처리하세요")
        view = _ResolveReportView(interaction.guild_id, report_id)
        await channel.send(embeds=embeds, view=view)

    @filter.command(name="resolve-report", description="신고를 처리하고 AI 학습 데이터에 반영합니다")
    @mod_only()
    @app_commands.choices(
        outcome=[
            app_commands.Choice(name="오탐: 허용/제재 완화", value="false_positive"),
            app_commands.Choice(name="정상 제재: 학습 강화", value="confirmed"),
            app_commands.Choice(name="중복", value="duplicate"),
            app_commands.Choice(name="기각", value="rejected"),
        ]
    )
    @app_commands.choices(category=CATEGORY_CHOICES)
    async def resolve_report(
        self,
        interaction: discord.Interaction,
        report_id: int,
        outcome: str,
        term: Optional[str] = None,
        category: str = "profanity",
    ) -> None:
        report = await self.bot.store.resolve_report(interaction.guild_id, report_id, outcome)
        if not report:
            await send_interaction(interaction, "신고를 찾을 수 없음")
            return

        infraction = None
        if report["infraction_id"]:
            infraction = await self.bot.store.get_infraction(interaction.guild_id, int(report["infraction_id"]))

        if outcome == "false_positive" and infraction:
            await self.bot.store.add_allowed_hash(
                interaction.guild_id,
                infraction.normalized_hash,
                interaction.user.id,
                reason=f"신고 #{report_id}",
            )
            await self.bot.store.add_learning_event(
                guild_id=interaction.guild_id,
                label="false_positive",
                source_type="report",
                source_id=report_id,
                content=infraction.content,
                category="false_positive",
                created_by=interaction.user.id,
            )
        elif outcome == "confirmed" and infraction:
            normalized_category = normalize_category(category)
            await self.bot.store.add_learning_event(
                guild_id=interaction.guild_id,
                label="confirmed_bad",
                source_type="report",
                source_id=report_id,
                content=infraction.content,
                term=term,
                category=normalized_category,
                created_by=interaction.user.id,
            )
            if term:
                await self.bot.store.add_blocked_term(
                    interaction.guild_id,
                    term,
                    2,
                    interaction.user.id,
                    "신고 처리에서 추가됨",
                    category=normalized_category,
                )

        await send_interaction(interaction, f"신고 #{report_id} 처리됨: {_outcome_label(outcome)}")

    @filter.command(name="exempt-role-add", description="필터 예외 역할을 추가합니다")
    @mod_only()
    async def exempt_role_add(self, interaction: discord.Interaction, role: discord.Role) -> None:
        await self.bot.store.set_role_exempt(interaction.guild_id, role.id, interaction.user.id, True)
        await send_interaction(interaction, f"예외 역할 추가됨: {role.mention}")

    @filter.command(name="exempt-role-remove", description="필터 예외 역할을 제거합니다")
    @mod_only()
    async def exempt_role_remove(self, interaction: discord.Interaction, role: discord.Role) -> None:
        await self.bot.store.set_role_exempt(interaction.guild_id, role.id, interaction.user.id, False)
        await send_interaction(interaction, f"예외 역할 제거됨: {role.mention}")

    @filter.command(name="exempt-user-add", description="필터 예외 유저를 추가합니다")
    @mod_only()
    async def exempt_user_add(self, interaction: discord.Interaction, member: discord.Member) -> None:
        await self.bot.store.set_user_exempt(interaction.guild_id, member.id, interaction.user.id, True)
        await send_interaction(interaction, f"예외 유저 추가됨: {member.mention}")

    @filter.command(name="exempt-user-remove", description="필터 예외 유저를 제거합니다")
    @mod_only()
    async def exempt_user_remove(self, interaction: discord.Interaction, member: discord.Member) -> None:
        await self.bot.store.set_user_exempt(interaction.guild_id, member.id, interaction.user.id, False)
        await send_interaction(interaction, f"예외 유저 제거됨: {member.mention}")

    async def cog_app_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
        if isinstance(error, (app_commands.CheckFailure, app_commands.NoPrivateMessage, app_commands.MissingPermissions)):
            await send_interaction(interaction, "권한 없음. 서버 관리자나 모더레이터 권한 필요.")
            return
        LOGGER.exception("Application command failed", exc_info=error)
        await send_interaction(interaction, "명령 처리 중 오류 발생. 로그 확인 필요.")


def run() -> None:
    config = load_config()
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bot = PrettyWordsBot(config)
    bot.run(config.discord_token, log_handler=None)
