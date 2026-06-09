from __future__ import annotations

import asyncio
import json
import logging
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .filtering import ModerationDecision, ModerationTerm


LOGGER = logging.getLogger(__name__)

CATEGORY_VALUES = ["profanity", "sexual", "family_insult", "harassment", "hate", "threat", "other"]

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "violation": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "severity": {"type": "integer", "minimum": 0, "maximum": 3},
        "categories": {"type": "array", "items": {"type": "string", "enum": CATEGORY_VALUES}, "maxItems": 6},
        "matched_terms": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "reason": {"type": "string", "maxLength": 240},
        "suggested_action": {
            "type": "string",
            "enum": ["none", "delete", "timeout", "review"],
        },
    },
    "required": [
        "violation",
        "confidence",
        "severity",
        "categories",
        "matched_terms",
        "reason",
        "suggested_action",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a Discord moderation classifier for Korean and English servers. "
    "Detect profanity, masked profanity, insults, harassment, and demeaning language. "
    "Use categories only from this taxonomy: profanity=general swearing, sexual=sexual remarks, "
    "family_insult=insults targeting family/parents, harassment=personal insults or bullying, "
    "hate=identity-based hate/discrimination, threat=threats of harm, other=policy violation not covered. "
    "Do not flag benign education, quoting for moderation, reclaimed terms, or clean jokes. "
    "CRITICAL Korean grammar — NEVER flag as vulgar: '자지마'/'자지 마'/'자지않다' = 'don't sleep' (자다 verb + -지 마 ending, NOT a body part); '보지마'/'보지 마'/'보지않다' = 'don't look' (보다 verb + -지 마 ending, NOT a body part). If '자지' or '보지' is immediately followed by 마/않/못/말/도/도록/않아/않고/마라/마요, it is a verb form, not vulgar. "
    "IMPORTANT: A false positive (flagging an innocent message) is worse than a false negative. "
    "When in doubt, do NOT flag — only flag when you are genuinely confident the message is a violation. "
    "Set confidence below the threshold if the message could plausibly be benign in context. "
    "Use custom blocked terms as strong policy, allowed terms as exceptions, and examples as learning context. "
    "If 'prior_messages' is provided, use it as conversational context to judge the target message. "
    "If 'guild_notes' is provided, treat it as moderator-supplied context about this community. "
    "Write the reason field in Korean. Return only JSON matching the schema. No markdown."
)

BATCH_SYSTEM_PROMPT = (
    "You are a Discord moderation classifier for Korean and English servers. "
    "You will receive a JSON object with a 'messages' array; each entry has a unique integer 'index', "
    "a 'message' string, and an optional 'prior_messages' array of recent channel messages for context. "
    "Classify EVERY message independently — never let one message's verdict influence another's. "
    "Use prior_messages only as conversational context for that entry's message. "
    "Detect profanity, masked profanity, insults, harassment, and demeaning language. "
    "Use categories only from this taxonomy: profanity, sexual, family_insult, harassment, hate, "
    "threat, other. Do not flag benign education, quoting for moderation, reclaimed terms, or clean jokes. "
    "CRITICAL Korean grammar — NEVER flag as vulgar: '자지마'/'자지 마'/'자지않다' = 'don't sleep' (자다 verb + -지 마 ending, NOT a body part); '보지마'/'보지 마'/'보지않다' = 'don't look' (보다 verb + -지 마 ending, NOT a body part). If '자지' or '보지' is immediately followed by 마/않/못/말/도/도록/않아/않고/마라/마요, it is a verb form, not vulgar. "
    "IMPORTANT: A false positive (flagging an innocent message) is worse than a false negative. "
    "When in doubt, do NOT flag — only flag when genuinely confident the message is a violation. "
    "Set confidence below threshold if the message could plausibly be benign in context. "
    "Use custom blocked terms as strong policy, allowed terms as exceptions, and examples as learning context. "
    "If 'guild_notes' is provided, treat it as moderator-supplied context about this community. "
    "Write every reason field in Korean. "
    "Return only JSON matching the schema: an object with a 'decisions' array containing exactly one "
    "decision object per input message, each carrying the same 'index' as its source message so results "
    "can be matched back. No markdown."
)

BATCH_DECISION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        **DECISION_SCHEMA["properties"],
    },
    "required": ["index", *DECISION_SCHEMA["required"]],
    "additionalProperties": False,
}

BATCH_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": BATCH_DECISION_ITEM_SCHEMA,
            "maxItems": 64,
        },
    },
    "required": ["decisions"],
    "additionalProperties": False,
}


class RateLimitedError(Exception):
    """Raised by a classifier when the provider reports it is being rate limited (HTTP 429)."""

    def __init__(self, retry_after: float | None = None, message: str = "") -> None:
        super().__init__(message or "AI 제공자의 요청 제한에 걸렸습니다")
        self.retry_after = retry_after


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class AIContext:
    blocked_terms: list[ModerationTerm] = field(default_factory=list)
    allowed_terms: list[str] = field(default_factory=list)
    confirmed_examples: list[str] = field(default_factory=list)
    false_positive_examples: list[str] = field(default_factory=list)
    auto_examples: list[str] = field(default_factory=list)
    guild_notes: str = ""


def _context_payload(context: AIContext) -> dict[str, object]:
    d: dict[str, object] = {
        "custom_blocked_terms": [
            {"term": term.term, "severity": term.severity, "category": term.category}
            for term in context.blocked_terms[:80]
        ],
        "allowed_terms": context.allowed_terms[:80],
        "confirmed_bad_examples": context.confirmed_examples[:8],
        "known_false_positive_examples": context.false_positive_examples[:8],
        "recent_auto_flagged_examples": context.auto_examples[:8],
    }
    if context.guild_notes:
        d["guild_notes"] = context.guild_notes[:500]
    return d


def _payload(message: str, context: AIContext, prior_messages: list[str] | None = None) -> dict[str, object]:
    d: dict[str, object] = {"message": message[:1800], **_context_payload(context)}
    if prior_messages:
        d["prior_messages"] = [m[:300] for m in prior_messages[-10:]]
    return d


def _batch_payload(
    messages: list[str],
    context: AIContext,
    prior_messages: list[list[str]] | None = None,
) -> dict[str, object]:
    items: list[dict[str, object]] = []
    for index, text in enumerate(messages):
        item: dict[str, object] = {"index": index, "message": text[:1800]}
        if prior_messages and index < len(prior_messages) and prior_messages[index]:
            item["prior_messages"] = [m[:300] for m in prior_messages[index][-10:]]
        items.append(item)
    return {"messages": items, **_context_payload(context)}


def _extract_json_object(content: str) -> dict[str, object]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def _decision_from_data(data: dict[str, object], source: str) -> ModerationDecision:
    return ModerationDecision(
        violation=bool(data["violation"]),
        confidence=float(data["confidence"]),
        severity=int(data["severity"]),
        categories=tuple(str(item) for item in data.get("categories", [])),
        matched_terms=tuple(str(item) for item in data.get("matched_terms", [])),
        reason=str(data.get("reason", "")),
        source=source,
        suggested_action=str(data.get("suggested_action", "none")),
    )


class OpenAIClassifier:
    def __init__(self, api_key: str, model: str) -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=api_key, timeout=8.0)
        self.model = model
        self.provider_name = "openai"

    async def classify(self, message: str, context: AIContext, prior_messages: list[str] | None = None) -> ModerationDecision | None:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": json.dumps(_payload(message, context, prior_messages), ensure_ascii=False),
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "prettywords_moderation_decision",
                        "strict": True,
                        "schema": DECISION_SCHEMA,
                    },
                },
            )
        except Exception:
            LOGGER.exception("OpenAI moderation request failed")
            return None

        try:
            content = response.choices[0].message.content or "{}"
            return _decision_from_data(_extract_json_object(content), "openai")
        except Exception:
            LOGGER.exception("OpenAI moderation response parsing failed")
            return None


class OllamaClassifier:
    def __init__(self, base_url: str, model: str, timeout_seconds: float = 12.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.provider_name = "ollama"
        self.last_error = ""

    async def classify(self, message: str, context: AIContext, prior_messages: list[str] | None = None) -> ModerationDecision | None:
        LOGGER.debug("Ollama request: model=%s | %.80r", self.model, message)
        try:
            result = await asyncio.to_thread(self._classify_sync, message, context, prior_messages)
        except Exception as exc:
            self.last_error = f"Ollama 요청 실패: {exc.__class__.__name__}: {exc}"
            LOGGER.exception("Ollama moderation request failed")
            return None
        if result is not None:
            LOGGER.debug(
                "Ollama result: violation=%s conf=%.2f severity=%d reason=%.120s",
                result.violation, result.confidence, result.severity, result.reason,
            )
        return result

    def _classify_sync(self, message: str, context: AIContext, prior_messages: list[str] | None = None) -> ModerationDecision | None:
        self.last_error = ""
        body = {
            "model": self.model,
            "stream": False,
            "format": DECISION_SCHEMA,
            "options": {"temperature": 0, "num_ctx": 2048},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(_payload(message, context, prior_messages), ensure_ascii=False)},
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except (TimeoutError, socket.timeout) as exc:
            self.last_error = f"Ollama 응답 시간 초과: {self.timeout_seconds:.0f}초"
            LOGGER.warning("%s", self.last_error)
            return None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self.last_error = f"Ollama HTTP 오류 {exc.code}: {detail[:180]}"
            LOGGER.error("%s", self.last_error)
            return None
        except urllib.error.URLError as exc:
            self.last_error = f"Ollama 연결 실패: {exc.reason}"
            LOGGER.warning("%s", self.last_error)
            return None

        content = json.loads(raw).get("message", {}).get("content", "{}")
        return _decision_from_data(_extract_json_object(str(content)), "ollama")


class GroqClassifier:
    """OpenAI-compatible classifier backed by Groq's hosted free-tier inference.

    Groq exposes an OpenAI-compatible Chat Completions API, so this reuses the
    `openai` SDK pointed at Groq's base URL. Unlike the other classifiers, this
    one is built around batching: `classify_batch` sends many messages in a
    single request (cheap on Groq's per-request rate limits), and `classify`
    is a thin convenience wrapper around a one-item batch for call sites that
    need a single decision (e.g. the rate-limit cooldown path).
    """

    BASE_URL = "https://api.groq.com/openai/v1"

    # Fallback cooldown for a single key when Groq reports 429 without a
    # Retry-After header — long enough to stop hammering it, short enough that
    # the rotation gets back to it before the global cooldown would have.
    DEFAULT_KEY_COOLDOWN_SECONDS = 30.0

    def __init__(
        self,
        api_keys: str | list[str] | tuple[str, ...],
        model: str,
        timeout_seconds: float = 20.0,
    ) -> None:
        from openai import AsyncOpenAI

        if isinstance(api_keys, str):
            api_keys = [api_keys]
        keys: list[str] = []
        seen: set[str] = set()
        for raw_key in api_keys:
            key = (raw_key or "").strip()
            if key and key not in seen:
                seen.add(key)
                keys.append(key)
        if not keys:
            raise ValueError("GroqClassifier requires at least one Groq API key")

        self._clients = [
            AsyncOpenAI(api_key=key, base_url=self.BASE_URL, timeout=timeout_seconds) for key in keys
        ]
        self._key_count = len(self._clients)
        self._cursor = 0
        self._key_cooldowns: list[datetime | None] = [None] * self._key_count
        self.model = model
        self.provider_name = "groq"
        self.last_error = ""
        # Some Groq models don't support json_schema structured output.
        # We start optimistic and downgrade to json_object on the first 400.
        self._use_json_schema = True

    async def classify(self, message: str, context: AIContext, prior_messages: list[str] | None = None) -> ModerationDecision | None:
        results = await self.classify_batch([message], context, [prior_messages or []])
        return results[0] if results else None

    def _attempt_order(self) -> list[int]:
        """Indices to try this round: keys not in cooldown first (round-robin
        from the cursor), then any cooling-down keys ordered by soonest-ready
        (so a request still goes out even if every key is currently limited)."""
        now = datetime.now(timezone.utc)
        rotation = [(self._cursor + offset) % self._key_count for offset in range(self._key_count)]
        ready = [i for i in rotation if self._key_cooldowns[i] is None or self._key_cooldowns[i] <= now]
        cooling = sorted(
            (i for i in rotation if i not in ready),
            key=lambda i: self._key_cooldowns[i],
        )
        return ready + cooling

    def _mark_rate_limited(self, index: int, retry_after: float | None) -> None:
        seconds = retry_after if retry_after and retry_after > 0 else self.DEFAULT_KEY_COOLDOWN_SECONDS
        self._key_cooldowns[index] = datetime.now(timezone.utc) + timedelta(seconds=seconds)

    def _mark_success(self, index: int) -> None:
        self._key_cooldowns[index] = None
        self._cursor = (index + 1) % self._key_count

    def _build_request_kwargs(self, payload: str) -> dict:
        base = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
        }
        if self._use_json_schema:
            base["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "prettywords_moderation_batch",
                    "strict": True,
                    "schema": BATCH_DECISION_SCHEMA,
                },
            }
        else:
            base["response_format"] = {"type": "json_object"}
        return base

    async def _send_sub_batch(
        self,
        key_index: int,
        messages: list[str],
        context: AIContext,
        prior_messages: list[list[str]] | None,
        index_offset: int,
    ) -> tuple[list[ModerationDecision | None], float | None]:
        """Send one chunk to a specific key. Returns (results, retry_after_or_None).
        results is [None]*len(messages) on failure."""
        from openai import BadRequestError, RateLimitError

        payload = json.dumps(
            _batch_payload(messages, context, prior_messages), ensure_ascii=False
        )
        try:
            response = await self._clients[key_index].chat.completions.create(
                **self._build_request_kwargs(payload)
            )
        except RateLimitError as exc:
            retry_after = _retry_after_seconds(exc)
            self._mark_rate_limited(key_index, retry_after)
            LOGGER.warning("Groq key #%d/%d rate limited (parallel)", key_index + 1, self._key_count)
            return [None] * len(messages), retry_after
        except BadRequestError as exc:
            if self._use_json_schema and "json_schema" in str(exc).lower():
                LOGGER.warning(
                    "Model %s does not support json_schema; switching to json_object", self.model
                )
                self._use_json_schema = False
                try:
                    response = await self._clients[key_index].chat.completions.create(
                        **self._build_request_kwargs(payload)
                    )
                except Exception:
                    LOGGER.exception("Groq sub-batch failed after json_object fallback")
                    return [None] * len(messages), None
            else:
                LOGGER.error("Groq bad request on key #%d: %s", key_index + 1, exc)
                return [None] * len(messages), None
        except Exception:
            LOGGER.exception("Groq sub-batch request failed (key #%d)", key_index + 1)
            return [None] * len(messages), None

        self._mark_success(key_index)
        try:
            content = response.choices[0].message.content or "{}"
            raw = _extract_json_object(content).get("decisions", [])
        except Exception:
            LOGGER.exception("Groq sub-batch response parsing failed")
            return [None] * len(messages), None

        results: list[ModerationDecision | None] = [None] * len(messages)
        for entry in raw:
            try:
                local_idx = int(entry["index"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= local_idx < len(messages) and results[local_idx] is None:
                try:
                    results[local_idx] = _decision_from_data(entry, "groq")
                except Exception:
                    LOGGER.exception("Groq sub-batch item parse error (local_idx=%s)", local_idx)
        return results, None

    async def classify_batch(
        self, messages: list[str], context: AIContext, prior_messages: list[list[str]] | None = None
    ) -> list[ModerationDecision | None]:
        """Classify messages using all available (non-rate-limited) API keys in parallel.

        Messages are split evenly across ready keys and sent concurrently with
        asyncio.gather. Failed sub-batches return None entries which the caller
        handles via the normal fallback path. Raises RateLimitedError only when
        every configured key is currently rate-limited.
        """
        if not messages:
            return []

        self.last_error = ""
        now = datetime.now(timezone.utc)
        ready_keys = [
            i for i in range(self._key_count)
            if self._key_cooldowns[i] is None or self._key_cooldowns[i] <= now
        ]

        if not ready_keys:
            # 모든 키가 rate-limit 중
            best_retry_after = min(
                (self._key_cooldowns[i] - now).total_seconds()
                for i in range(self._key_count)
                if self._key_cooldowns[i] is not None
            )
            if self._key_count > 1:
                self.last_error = f"Groq 키 {self._key_count}개가 모두 요청 제한에 걸렸습니다"
            else:
                self.last_error = "Groq 요청 제한에 걸렸습니다"
            LOGGER.warning("%s", self.last_error)
            raise RateLimitedError(best_retry_after, self.last_error)

        n_keys = len(ready_keys)
        # 메시지를 준비된 키 수만큼 균등 분할
        chunk_size = max(1, -(-len(messages) // n_keys))  # ceiling division
        chunks: list[list[str]] = []
        prior_chunks: list[list[list[str]] | None] = []
        for start in range(0, len(messages), chunk_size):
            end = start + chunk_size
            chunks.append(messages[start:end])
            prior_chunks.append(prior_messages[start:end] if prior_messages else None)

        # 실제 사용할 키: 청크 수만큼 (키가 많아도 청크 수 이상은 안 씀)
        used_keys = ready_keys[:len(chunks)]

        LOGGER.debug(
            "Groq parallel batch: %d messages → %d chunk(s) across %d/%d keys | model=%s",
            len(messages), len(chunks), len(used_keys), self._key_count, self.model,
        )

        coros = [
            self._send_sub_batch(key_idx, chunk, context, prior_chunk, i * chunk_size)
            for i, (key_idx, chunk, prior_chunk) in enumerate(zip(used_keys, chunks, prior_chunks))
        ]
        sub_results = await asyncio.gather(*coros)

        # 결과를 원래 순서대로 합칩니다
        merged: list[ModerationDecision | None] = []
        all_rate_limited = True
        best_retry: float | None = None
        for chunk_decisions, retry_after in sub_results:
            merged.extend(chunk_decisions)
            if retry_after is None:
                all_rate_limited = False
            elif best_retry is None or retry_after < best_retry:
                best_retry = retry_after

        # 준비된 키가 있었는데 전부 rate-limit 됐다면 예외 발생
        if all_rate_limited and all(d is None for d in merged):
            if self._key_count > 1:
                self.last_error = f"Groq 키 {self._key_count}개가 모두 요청 제한에 걸렸습니다"
            else:
                self.last_error = "Groq 요청 제한에 걸렸습니다"
            raise RateLimitedError(best_retry, self.last_error)

        missing = sum(1 for d in merged if d is None)
        if missing:
            self.last_error = f"Groq 배치 응답에서 {missing}/{len(messages)}개 판정 누락"
            LOGGER.warning("%s", self.last_error)
        else:
            LOGGER.debug("Groq parallel batch complete: %d decisions", len(merged))
        return merged
