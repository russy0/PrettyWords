from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands, tasks

from .ai import AIContext, GroqClassifier, OllamaClassifier, OpenAIClassifier, RateLimitedError
from .config import BotConfig, load_config
from .filtering import (
    CATEGORY_LABELS,
    LocalClassifier,
    ModerationDecision,
    category_label,
    compact_text,
    combine_decisions,
    message_fingerprint,
    normalize_category,
)
from .storage import GuildSettings, ModerationStore


LOGGER = logging.getLogger(__name__)
MAX_TIMEOUT_MINUTES = 28 * 24 * 60
CATEGORY_CHOICES = [
    app_commands.Choice(name=f"{label} ({value})", value=value)
    for value, label in CATEGORY_LABELS.items()
]


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

    if getattr(interaction.guild, "owner_id", None) == user_id:
        return True

    if hasattr(bot, "config") and user_id in bot.config.bot_admin_ids:
        return True

    if hasattr(bot, "store"):
        if await bot.store.is_config_admin(interaction.guild_id, user_id):
            return True
        if await bot.store.has_config_admins(interaction.guild_id):
            return False

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
            if settings.ai_model or self.config.ollama_model:
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
            return "none"
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


class ModerationCog(commands.Cog):
    filter = app_commands.Group(name="filter", description="AI profanity filter settings")
    admin = app_commands.Group(name="pw", description="PrettyWords bot administration")

    def __init__(self, bot: PrettyWordsBot) -> None:
        self.bot = bot
        self._stats: dict[int, dict[str, int | str]] = {}
        self._groq_queues: dict[int, list[_PendingClassification]] = {}
        self._groq_flush_locks: dict[int, asyncio.Lock] = {}

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
                "last_scan": "never",
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
            await self._send_health_log(guild, settings, title="PrettyWords Health")

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
        title: str = "PrettyWords Health",
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
            description="Message scanning is running." if not settings.paused else "Filter is paused.",
        )
        embed.add_field(name="AI", value=self.bot._ai_label(settings) if settings.ai_enabled else "disabled", inline=True)
        embed.add_field(name="AI Scan All", value=str(scan_all), inline=True)
        embed.add_field(name="Threshold", value=f"{settings.confidence_threshold:.2f}", inline=True)
        embed.add_field(name="Seen", value=str(stats["seen"]), inline=True)
        embed.add_field(name="Checked", value=str(stats["checked"]), inline=True)
        embed.add_field(name="Skipped", value=str(stats["skipped"]), inline=True)
        embed.add_field(name="AI Calls", value=str(stats["ai_calls"]), inline=True)
        embed.add_field(name="AI Failures", value=str(stats["ai_failures"]), inline=True)
        embed.add_field(name="Violations", value=str(stats["violations"]), inline=True)
        embed.add_field(name="Deleted", value=str(stats["deleted"]), inline=True)
        embed.add_field(name="Timeouts", value=str(stats["timeouts"]), inline=True)
        embed.add_field(name="Last Scan", value=str(stats["last_scan"]), inline=False)

        queue_len = len(self._groq_queues.get(guild.id, []))
        if queue_len:
            embed.add_field(name="Groq Queue", value=str(queue_len), inline=True)
        cooldown = self.bot._groq_cooldown_remaining()
        if cooldown > 0:
            embed.add_field(name="Groq Cooldown", value=f"{cooldown:.0f}s (using local fallback)", inline=True)

        if stats.get("last_error"):
            embed.add_field(name="Last Error", value=str(stats["last_error"])[:1000], inline=False)

        try:
            await channel.send(embed=embed)
        except discord.HTTPException:
            LOGGER.exception("Failed to send health log")

    async def _build_ai_context(self, guild_id: int, blocked_terms, allowed_terms) -> AIContext:
        return AIContext(
            blocked_terms=blocked_terms,
            allowed_terms=allowed_terms,
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
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None or message.author.bot or not message.content:
            return

        self._bump(message.guild.id, "seen")
        settings = await self.bot.store.get_settings(message.guild.id)
        if settings.paused:
            self._bump(message.guild.id, "skipped")
            return
        if await self.bot.store.is_channel_disabled(message.guild.id, message.channel.id):
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

        blocked_terms = await self.bot.store.list_blocked_terms(message.guild.id)
        allowed_terms = await self.bot.store.list_allowed_terms(message.guild.id)
        local = self.bot.local_classifier.classify(message.content, blocked_terms, allowed_terms)

        LOGGER.debug(
            "[%s] local: violation=%s conf=%.2f matched=%s",
            message.guild.name,
            local.violation,
            local.confidence,
            list(local.matched_terms) or "[]",
        )

        _provider, _model, scan_all = self.bot._effective_ai_settings(settings)
        classifier, batchable = self._resolve_ai_classifier(settings)
        should_call_ai = bool(settings.ai_enabled and classifier is not None and (scan_all or local.violation))

        LOGGER.debug(
            "[%s] AI: provider=%s batchable=%s should_call=%s ai_enabled=%s",
            message.guild.name,
            getattr(classifier, "provider_name", "none") if classifier else "none",
            batchable,
            should_call_ai,
            settings.ai_enabled,
        )

        if should_call_ai and batchable:
            # Groq is active: queue this message and let it ride out in a batch
            # instead of spending a request per message. _finish_moderation runs
            # later, once the batch comes back.
            await self._queue_for_batch(message, settings, local)
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
            context = await self._build_ai_context(message.guild.id, blocked_terms, allowed_terms)
            ai_decision = await classifier.classify(message.content, context)
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

        decision = combine_decisions(local, ai_decision, settings.confidence_threshold)
        LOGGER.debug(
            "[%s] combined: source=%s violation=%s conf=%.2f threshold=%.2f",
            message.guild.name,
            decision.source,
            decision.violation,
            decision.confidence,
            settings.confidence_threshold,
        )
        await self._finish_moderation(message, decision, settings)

    async def _queue_for_batch(
        self, message: discord.Message, settings: GuildSettings, local: ModerationDecision
    ) -> None:
        queue = self._groq_queues.setdefault(message.guild.id, [])
        queue.append(
            _PendingClassification(
                message=message,
                settings=settings,
                local=local,
                queued_at=datetime.now(timezone.utc),
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

            blocked_terms = await self.bot.store.list_blocked_terms(guild_id)
            allowed_terms = await self.bot.store.list_allowed_terms(guild_id)
            context = await self._build_ai_context(guild_id, blocked_terms, allowed_terms)

            # Re-resolve against the latest settings: the provider may have
            # changed, or a Groq cooldown may have just kicked in/expired.
            current_settings = await self.bot.store.get_settings(guild_id)
            classifier, batchable = self._resolve_ai_classifier(current_settings)
            provider_name = getattr(classifier, "provider_name", "none") if classifier else "none"

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
                    decisions = await classifier.classify_batch(
                        [item.message.content for item in items], context
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
                ai_decision = await classifier.classify(item.message.content, context) if classifier else None
                await self._complete_classification(item, ai_decision, classifier)

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
                    ai_decision = await fallback.classify(item.message.content, context)
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

        decision = combine_decisions(item.local, ai_decision, item.settings.confidence_threshold)
        LOGGER.debug(
            "[guild:%d] combined: source=%s violation=%s conf=%.2f",
            guild_id,
            decision.source,
            decision.violation,
            decision.confidence,
        )
        await self._finish_moderation(item.message, decision, item.settings)

    async def _finish_moderation(
        self,
        message: discord.Message,
        decision: ModerationDecision,
        settings: GuildSettings,
    ) -> None:
        if not decision.violation or decision.confidence < settings.confidence_threshold:
            LOGGER.debug(
                "[%s] pass: source=%s violation=%s conf=%.2f threshold=%.2f | %s | %.60r",
                message.guild.name,
                decision.source,
                decision.violation,
                decision.confidence,
                settings.confidence_threshold,
                message.author,
                message.content,
            )
            return

        fingerprint = message_fingerprint(message.content)
        self._bump(message.guild.id, "violations")
        recent = await self.bot.store.count_recent_infractions(message.guild.id, message.author.id)
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

        await self._log_infraction(message, infraction_id, decision, action, timeout_minutes, settings)
        if settings.dm_users and not settings.dry_run:
            await self._dm_warning(message.author, infraction_id, timeout_minutes, decision.reason)

    def _effective_timeout(self, settings: GuildSettings, recent_count: int) -> int:
        base = max(0, min(MAX_TIMEOUT_MINUTES, settings.timeout_minutes))
        if base == 0:
            return 0
        if not settings.escalate:
            return base
        multiplier = min(16, 2**min(recent_count, 4))
        return min(MAX_TIMEOUT_MINUTES, base * multiplier)

    async def _timeout_member(self, member: discord.Member, minutes: int, reason: str) -> str:
        reason_text = f"PrettyWords profanity filter: {reason[:120]}"
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
        decision,
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
            title=f"PrettyWords Case #{infraction_id}",
            color=discord.Color.orange(),
            description=decision.reason[:350] or "Policy violation detected.",
        )
        embed.add_field(
            name="User",
            value=f"{message.author.mention} {message.author} (`{message.author.id}`)",
            inline=False,
        )
        embed.add_field(name="Channel", value=message.channel.mention, inline=True)
        embed.add_field(name="Action", value=action, inline=True)
        embed.add_field(name="Timeout", value=f"{timeout_minutes}m", inline=True)
        embed.add_field(name="Confidence", value=f"{decision.confidence:.2f}", inline=True)
        embed.add_field(name="Severity", value=str(decision.severity), inline=True)
        embed.add_field(name="Source", value=decision.source, inline=True)
        embed.add_field(name="Message Link", value=f"[Jump]({message.jump_url})", inline=False)
        if decision.categories:
            embed.add_field(
                name="Categories",
                value=", ".join(category_label(category) for category in decision.categories)[:250],
                inline=False,
            )
        if decision.matched_terms:
            embed.add_field(name="Matched", value=", ".join(decision.matched_terms)[:250], inline=False)
        embed.add_field(name="Message", value=message.content[:900] or "(empty)", inline=False)
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
    ) -> None:
        try:
            await user.send(
                f"PrettyWords case #{infraction_id}: 서버 규칙 위반 가능성이 감지되었습니다. "
                f"타임아웃: {timeout_minutes}분. 사유: {reason[:250]} "
                f"오탐이면 서버에서 `/filter report case_id:{infraction_id}` 명령으로 이의제기하세요. "
                "봇 관리자가 승인해야 학습에 반영됩니다."
            )
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
        embed = discord.Embed(title="PrettyWords Status", color=discord.Color.green())
        embed.add_field(name="Paused", value=str(settings.paused), inline=True)
        _provider, _model, scan_all = self.bot._effective_ai_settings(settings)
        embed.add_field(name="AI", value=self.bot._ai_label(settings) if settings.ai_enabled else "disabled", inline=True)
        embed.add_field(name="AI Scan All", value=str(scan_all), inline=True)
        embed.add_field(name="Health Logs", value=str(settings.health_log_enabled), inline=True)
        embed.add_field(name="Dry Run", value=str(settings.dry_run), inline=True)
        embed.add_field(name="Timeout", value=f"{settings.timeout_minutes}m", inline=True)
        embed.add_field(name="Threshold", value=f"{settings.confidence_threshold:.2f}", inline=True)
        embed.add_field(name="Escalate", value=str(settings.escalate), inline=True)
        embed.add_field(name="Log Channel", value=f"<#{settings.log_channel_id}>" if settings.log_channel_id else "not set", inline=False)
        health_channel_value = (
            f"<#{settings.health_log_channel_id}>"
            if settings.health_log_channel_id
            else (f"log channel (<#{settings.log_channel_id}>)" if settings.log_channel_id else "not set")
        )
        embed.add_field(
            name="Health Channel",
            value=health_channel_value,
            inline=False,
        )
        embed.add_field(name="Disabled Channels", value=", ".join(f"<#{cid}>" for cid in disabled) or "none", inline=False)
        embed.add_field(name="Custom Blocked Terms", value=str(len(blocked)), inline=True)
        embed.add_field(name="Allowed Terms", value=str(len(allowed)), inline=True)
        embed.add_field(name="Config Admins", value=str(len(config_admins)), inline=True)
        await send_interaction(interaction, embed=embed)

    @admin.command(name="config-admin-add", description="Allow a Discord user ID to configure PrettyWords")
    @mod_only()
    @app_commands.describe(user_id="Discord user ID or mention")
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
            "Config Admin Added",
            f"{interaction.user} added `{parsed_user_id}`",
        )

    @admin.command(name="config-admin-remove", description="Remove a PrettyWords config admin Discord ID")
    @mod_only()
    @app_commands.describe(user_id="Discord user ID or mention")
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
                "Config Admin Removed",
                f"{interaction.user} removed `{parsed_user_id}`",
            )

    @admin.command(name="config-admin-list", description="List PrettyWords config admin Discord IDs")
    @mod_only()
    async def config_admin_list(self, interaction: discord.Interaction) -> None:
        admins = await self.bot.store.list_config_admins(interaction.guild_id)
        global_admins = sorted(self.bot.config.bot_admin_ids)
        lines = []
        if admins:
            lines.append("Server: " + ", ".join(f"`{admin_id}`" for admin_id in admins))
        if global_admins:
            lines.append(".env: " + ", ".join(f"`{admin_id}`" for admin_id in global_admins))
        await send_interaction(interaction, "\n".join(lines) if lines else "설정 관리자 ID 없음")

    @filter.command(name="log-channel", description="제재/신고 로그 채널을 설정합니다")
    @mod_only()
    async def log_channel(self, interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        settings = await self.bot.store.update_settings(interaction.guild_id, log_channel_id=channel.id)
        await send_interaction(interaction, f"로그 채널 설정됨: {channel.mention}")
        await self._log_admin_event(interaction.guild, "Log Channel Updated", f"{interaction.user} set logs to {channel.mention}")

    @filter.command(name="health-log-channel", description="상태 로그 전용 채널을 설정합니다")
    @mod_only()
    async def health_log_channel(self, interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        settings = await self.bot.store.update_settings(interaction.guild_id, health_log_channel_id=channel.id)
        await send_interaction(interaction, f"상태 로그 채널 설정됨: {channel.mention}")
        await self._send_health_log(interaction.guild, settings, title="PrettyWords Health Channel Connected")

    @filter.command(name="health", description="Send current PrettyWords health to the log channel")
    @mod_only()
    async def health(self, interaction: discord.Interaction) -> None:
        settings = await self.bot.store.get_settings(interaction.guild_id)
        if not (settings.health_log_channel_id or settings.log_channel_id):
            await send_interaction(interaction, "상태 로그 채널 먼저 설정 필요: /filter health-log-channel")
            return
        await self._send_health_log(interaction.guild, settings)
        await send_interaction(interaction, "상태 로그 전송됨")

    @filter.command(name="health-log", description="Enable or disable periodic health logs")
    @mod_only()
    async def health_log(self, interaction: discord.Interaction, enabled: bool) -> None:
        settings = await self.bot.store.update_settings(interaction.guild_id, health_log_enabled=enabled)
        await send_interaction(interaction, f"상태 로그: {enabled}")
        if enabled and (settings.health_log_channel_id or settings.log_channel_id):
            await self._send_health_log(interaction.guild, settings, title="PrettyWords Health Logs Enabled")

    @filter.command(name="timeout", description="비속어 사용 시 타임아웃 시간을 설정합니다")
    @mod_only()
    @app_commands.describe(minutes="0이면 타임아웃 없이 삭제/로그만 합니다")
    async def timeout(self, interaction: discord.Interaction, minutes: app_commands.Range[int, 0, MAX_TIMEOUT_MINUTES]) -> None:
        await self.bot.store.update_settings(interaction.guild_id, timeout_minutes=int(minutes))
        await send_interaction(interaction, f"타임아웃 기본값: {minutes}분")
        await self._log_admin_event(interaction.guild, "Timeout Updated", f"{interaction.user} set timeout to {minutes}m")

    @filter.command(name="threshold", description="AI/필터 확신도 기준을 설정합니다")
    @mod_only()
    async def threshold(self, interaction: discord.Interaction, confidence: app_commands.Range[float, 0.1, 0.99]) -> None:
        await self.bot.store.update_settings(interaction.guild_id, confidence_threshold=float(confidence))
        await send_interaction(interaction, f"확신도 기준: {confidence:.2f}")

    @filter.command(name="mode", description="삭제/DM/dry-run/AI/escalation 설정을 바꿉니다")
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
        await send_interaction(
            interaction,
            (
                f"모드 업데이트: delete={settings.delete_messages}, dm={settings.dm_users}, "
                f"dry_run={settings.dry_run}, ai={settings.ai_enabled}, escalate={settings.escalate}"
            ),
        )

    @filter.command(name="ai", description="AI provider/model settings")
    @mod_only()
    @app_commands.choices(
        provider=[
            app_commands.Choice(name="default (.env)", value="default"),
            app_commands.Choice(name="ollama", value="ollama"),
            app_commands.Choice(name="openai", value="openai"),
            app_commands.Choice(name="groq", value="groq"),
            app_commands.Choice(name="none", value="none"),
            app_commands.Choice(name="auto", value="auto"),
        ]
    )
    @app_commands.describe(
        model="Example: qwen3:4b, qwen3:1.7b, or default",
        scan_all="false = only suspicious messages go to AI",
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
        _provider, _model, effective_scan_all = self.bot._effective_ai_settings(settings)
        await send_interaction(
            interaction,
            f"AI: {self.bot._ai_label(settings)}, scan_all={effective_scan_all}",
        )

    @filter.command(name="ai-reset", description="AI settings back to .env defaults")
    @mod_only()
    async def ai_reset(self, interaction: discord.Interaction) -> None:
        settings = await self.bot.store.update_settings(
            interaction.guild_id,
            ai_provider="",
            ai_model="",
            ai_scan_all=None,
        )
        _provider, _model, effective_scan_all = self.bot._effective_ai_settings(settings)
        await send_interaction(
            interaction,
            f"AI reset: {self.bot._ai_label(settings)}, scan_all={effective_scan_all}",
        )

    @filter.command(name="pause", description="필터를 일시정지합니다")
    @mod_only()
    async def pause(self, interaction: discord.Interaction, reason: Optional[str] = None) -> None:
        await self.bot.store.update_settings(interaction.guild_id, paused=True)
        await send_interaction(interaction, "필터 일시정지됨")
        await self._log_admin_event(interaction.guild, "Filter Paused", f"{interaction.user}: {reason or 'no reason'}")

    @filter.command(name="resume", description="필터를 다시 시작합니다")
    @mod_only()
    async def resume(self, interaction: discord.Interaction) -> None:
        await self.bot.store.update_settings(interaction.guild_id, paused=False)
        await send_interaction(interaction, "필터 다시 시작됨")
        await self._log_admin_event(interaction.guild, "Filter Resumed", f"{interaction.user} resumed filtering")

    @filter.command(name="disable-channel", description="특정 채널에서 필터를 끕니다")
    @mod_only()
    async def disable_channel(self, interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        await self.bot.store.set_channel_disabled(interaction.guild_id, channel.id, interaction.user.id, True)
        await send_interaction(interaction, f"비활성화됨: {channel.mention}")

    @filter.command(name="enable-channel", description="특정 채널에서 필터를 켭니다")
    @mod_only()
    async def enable_channel(self, interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        await self.bot.store.set_channel_disabled(interaction.guild_id, channel.id, interaction.user.id, False)
        await send_interaction(interaction, f"활성화됨: {channel.mention}")

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
        await send_interaction(
            interaction,
            f"등록됨: `{term}` category={category_label(normalized_category)} severity={severity}",
        )

    @filter.command(name="remove-word", description="서버 전용 금지어를 제거합니다")
    @mod_only()
    async def remove_word(self, interaction: discord.Interaction, term: str) -> None:
        count = await self.bot.store.remove_blocked_term(interaction.guild_id, term)
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
            f"learned from message {target_message.id}",
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

        note = "" if term_found else "\n주의: term이 메시지에 정확히 포함되지는 않음. 우회표현이면 정상일 수 있음."
        await interaction.followup.send(
            (
                f"학습됨: `{term}` → {category_label(normalized_category)} severity={severity}\n"
                f"메시지: {target_message.jump_url}{note}"
            ),
            ephemeral=True,
        )
        await self._log_admin_event(
            interaction.guild,
            "Message Learned",
            (
                f"{interaction.user} learned `{term}` as {normalized_category} "
                f"from [message]({target_message.jump_url})"
            ),
        )

    @filter.command(name="allow-word", description="오탐 방지용 허용어/문구를 등록합니다")
    @mod_only()
    async def allow_word(self, interaction: discord.Interaction, term: str) -> None:
        await self.bot.store.add_allowed_term(interaction.guild_id, term, interaction.user.id)
        await send_interaction(interaction, f"허용됨: `{term}`")

    @filter.command(name="remove-allow", description="허용어/문구를 제거합니다")
    @mod_only()
    async def remove_allow(self, interaction: discord.Interaction, term: str) -> None:
        count = await self.bot.store.remove_allowed_term(interaction.guild_id, term)
        await send_interaction(interaction, "제거됨" if count else "등록된 항목 없음")

    @filter.command(name="report", description="부적절한 제재/오탐을 신고합니다")
    @app_commands.describe(case_id="로그나 DM에 표시된 PrettyWords case ID")
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
        await send_interaction(interaction, f"이의제기/신고 접수됨: report #{report_id}. 관리자 승인 후 학습됩니다.")
        await self._log_report(interaction, report_id, case_id, reason)

    async def _log_report(
        self,
        interaction: discord.Interaction,
        report_id: int,
        case_id: int | None,
        reason: str,
    ) -> None:
        settings = await self.bot.store.get_settings(interaction.guild_id)
        if not settings.log_channel_id:
            return
        channel = interaction.guild.get_channel(settings.log_channel_id)
        if not isinstance(channel, discord.abc.Messageable):
            return
        embed = discord.Embed(
            title=f"PrettyWords Report #{report_id}",
            description=reason[:900],
            color=discord.Color.red(),
        )
        embed.add_field(name="Reporter", value=f"{interaction.user.mention} {interaction.user} (`{interaction.user.id}`)", inline=False)
        embed.add_field(name="Case", value=f"#{case_id}" if case_id else "not provided", inline=True)
        embed.set_footer(text="/filter resolve-report outcome:false_positive 승인 시 비속어 아님으로 학습")
        await channel.send(embed=embed)

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
                reason=f"report #{report_id}",
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
                    "from report",
                    category=normalized_category,
                )

        await send_interaction(interaction, f"report #{report_id} 처리됨: {outcome}")

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
