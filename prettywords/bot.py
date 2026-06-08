from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from .ai import AIContext, OllamaClassifier, OpenAIClassifier
from .config import BotConfig, load_config
from .filtering import LocalClassifier, combine_decisions, message_fingerprint
from .storage import GuildSettings, ModerationStore


LOGGER = logging.getLogger(__name__)
MAX_TIMEOUT_MINUTES = 28 * 24 * 60


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

    def _effective_ai_settings(self, settings: GuildSettings) -> tuple[str, str, bool]:
        provider = (settings.ai_provider or self.config.ai_provider).strip().lower()
        if provider == "auto":
            if settings.ai_model or self.config.ollama_model:
                provider = "ollama"
            elif self.config.openai_api_key:
                provider = "openai"
            else:
                provider = "none"

        if provider == "ollama":
            model = settings.ai_model or self.config.ollama_model
        elif provider == "openai":
            model = settings.ai_model or self.config.openai_model
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
            key = ("ollama", self.config.ollama_base_url, model, self.config.ollama_timeout_seconds)
            if key not in self._ai_classifier_cache:
                self._ai_classifier_cache[key] = OllamaClassifier(
                    self.config.ollama_base_url,
                    model,
                    timeout_seconds=self.config.ollama_timeout_seconds,
                )
            return self._ai_classifier_cache[key]
        if provider == "openai":
            if not self.config.openai_api_key:
                LOGGER.warning("AI_PROVIDER=openai but OPENAI_API_KEY is empty; AI disabled")
                return None
            key = ("openai", model, self.config.openai_api_key)
            if key not in self._ai_classifier_cache:
                self._ai_classifier_cache[key] = OpenAIClassifier(self.config.openai_api_key, model)
            return self._ai_classifier_cache[key]
        LOGGER.warning("Unknown AI_PROVIDER=%s; AI disabled", provider)
        return None

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

    def __init__(self, bot: PrettyWordsBot) -> None:
        self.bot = bot

    async def _classify(self, guild_id: int, content: str, settings: GuildSettings):
        blocked_terms = await self.bot.store.list_blocked_terms(guild_id)
        allowed_terms = await self.bot.store.list_allowed_terms(guild_id)
        local = self.bot.local_classifier.classify(content, blocked_terms, allowed_terms)

        ai_decision = None
        provider, _model, scan_all = self.bot._effective_ai_settings(settings)
        classifier = self.bot._get_ai_classifier(settings)
        should_call_ai = scan_all or local.violation
        if settings.ai_enabled and classifier and should_call_ai:
            context = AIContext(
                blocked_terms=blocked_terms,
                allowed_terms=allowed_terms,
                confirmed_examples=await self.bot.store.learning_examples(guild_id, "confirmed_bad"),
                false_positive_examples=await self.bot.store.learning_examples(guild_id, "false_positive"),
                auto_examples=await self.bot.store.learning_examples(guild_id, "auto_flagged"),
            )
            ai_decision = await classifier.classify(content, context)

        return combine_decisions(local, ai_decision, settings.confidence_threshold)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None or message.author.bot or not message.content:
            return

        settings = await self.bot.store.get_settings(message.guild.id)
        if settings.paused:
            return
        if await self.bot.store.is_channel_disabled(message.guild.id, message.channel.id):
            return

        role_ids = [role.id for role in getattr(message.author, "roles", [])]
        if await self.bot.store.is_user_exempt(message.guild.id, message.author.id, role_ids):
            return

        fingerprint = message_fingerprint(message.content)
        if await self.bot.store.is_allowed_hash(message.guild.id, fingerprint):
            return

        decision = await self._classify(message.guild.id, message.content, settings)
        if not decision.violation or decision.confidence < settings.confidence_threshold:
            return

        recent = await self.bot.store.count_recent_infractions(message.guild.id, message.author.id)
        timeout_minutes = self._effective_timeout(settings, recent)
        action_parts: list[str] = []

        if settings.dry_run:
            action_parts.append("dry-run")
        else:
            if settings.delete_messages:
                try:
                    await message.delete(reason="PrettyWords profanity filter")
                    action_parts.append("deleted")
                except discord.NotFound:
                    action_parts.append("already deleted")
                except discord.Forbidden:
                    action_parts.append("delete failed: missing permission")
                except discord.HTTPException:
                    LOGGER.exception("Failed to delete message %s", message.id)
                    action_parts.append("delete failed")

            if timeout_minutes > 0 and isinstance(message.author, discord.Member):
                action_parts.append(await self._timeout_member(message.author, timeout_minutes, decision.reason))

        action = ", ".join(action_parts) or "logged"
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

        if decision.source in {"ai", "local+ai"} and decision.confidence >= 0.9:
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
        embed.add_field(name="User", value=f"{message.author} (`{message.author.id}`)", inline=False)
        embed.add_field(name="Channel", value=message.channel.mention, inline=True)
        embed.add_field(name="Action", value=action, inline=True)
        embed.add_field(name="Timeout", value=f"{timeout_minutes}m", inline=True)
        embed.add_field(name="Confidence", value=f"{decision.confidence:.2f}", inline=True)
        embed.add_field(name="Severity", value=str(decision.severity), inline=True)
        embed.add_field(name="Source", value=decision.source, inline=True)
        if decision.categories:
            embed.add_field(name="Categories", value=", ".join(decision.categories)[:250], inline=False)
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
                f"오탐이면 서버에서 `/filter report case_id:{infraction_id}` 명령으로 신고하세요."
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
        embed.add_field(name="Dry Run", value=str(settings.dry_run), inline=True)
        embed.add_field(name="Timeout", value=f"{settings.timeout_minutes}m", inline=True)
        embed.add_field(name="Threshold", value=f"{settings.confidence_threshold:.2f}", inline=True)
        embed.add_field(name="Escalate", value=str(settings.escalate), inline=True)
        embed.add_field(name="Log Channel", value=f"<#{settings.log_channel_id}>" if settings.log_channel_id else "not set", inline=False)
        embed.add_field(name="Disabled Channels", value=", ".join(f"<#{cid}>" for cid in disabled) or "none", inline=False)
        embed.add_field(name="Custom Blocked Terms", value=str(len(blocked)), inline=True)
        embed.add_field(name="Allowed Terms", value=str(len(allowed)), inline=True)
        embed.add_field(name="Config Admins", value=str(len(config_admins)), inline=True)
        await send_interaction(interaction, embed=embed)

    @filter.command(name="config-admin-add", description="Allow a Discord user ID to configure PrettyWords")
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

    @filter.command(name="config-admin-remove", description="Remove a PrettyWords config admin Discord ID")
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

    @filter.command(name="config-admin-list", description="List PrettyWords config admin Discord IDs")
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
        await self.bot.store.update_settings(interaction.guild_id, log_channel_id=channel.id)
        await send_interaction(interaction, f"로그 채널 설정됨: {channel.mention}")
        await self._log_admin_event(interaction.guild, "Log Channel Updated", f"{interaction.user} set logs to {channel.mention}")

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
    async def add_word(
        self,
        interaction: discord.Interaction,
        term: str,
        severity: app_commands.Range[int, 1, 3] = 2,
        notes: Optional[str] = None,
    ) -> None:
        await self.bot.store.add_blocked_term(interaction.guild_id, term, int(severity), interaction.user.id, notes or "")
        await self.bot.store.add_learning_event(
            guild_id=interaction.guild_id,
            label="confirmed_bad",
            source_type="manual_term",
            content=term,
            term=term,
            created_by=interaction.user.id,
        )
        await send_interaction(interaction, f"등록됨: `{term}` severity={severity}")

    @filter.command(name="remove-word", description="서버 전용 금지어를 제거합니다")
    @mod_only()
    async def remove_word(self, interaction: discord.Interaction, term: str) -> None:
        count = await self.bot.store.remove_blocked_term(interaction.guild_id, term)
        await send_interaction(interaction, "제거됨" if count else "등록된 항목 없음")

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
        await send_interaction(interaction, f"신고 접수됨: report #{report_id}")
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
        embed.add_field(name="Reporter", value=f"{interaction.user} (`{interaction.user.id}`)", inline=False)
        embed.add_field(name="Case", value=f"#{case_id}" if case_id else "not provided", inline=True)
        embed.set_footer(text="/filter resolve-report 로 처리")
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
    async def resolve_report(
        self,
        interaction: discord.Interaction,
        report_id: int,
        outcome: str,
        term: Optional[str] = None,
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
                created_by=interaction.user.id,
            )
        elif outcome == "confirmed" and infraction:
            await self.bot.store.add_learning_event(
                guild_id=interaction.guild_id,
                label="confirmed_bad",
                source_type="report",
                source_id=report_id,
                content=infraction.content,
                term=term,
                created_by=interaction.user.id,
            )
            if term:
                await self.bot.store.add_blocked_term(interaction.guild_id, term, 2, interaction.user.id, "from report")

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
