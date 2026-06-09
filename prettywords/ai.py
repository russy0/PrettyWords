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
    "Use custom blocked terms as strong policy, allowed terms as exceptions, and examples as learning context. "
    "Write the reason field in Korean. Return only JSON matching the schema. No markdown."
)

BATCH_SYSTEM_PROMPT = (
    "You are a Discord moderation classifier for Korean and English servers. "
    "You will receive a JSON object with a 'messages' array; each entry has a unique integer 'index' "
    "and a 'message' string. Classify EVERY message independently — never let one message's content "
    "influence another's verdict. Detect profanity, masked profanity, insults, harassment, and demeaning "
    "language. Use categories only from this taxonomy: profanity, sexual, family_insult, harassment, hate, "
    "threat, other. Do not flag benign education, quoting for moderation, reclaimed terms, or clean jokes. "
    "Use custom blocked terms as strong policy, allowed terms as exceptions, and examples as learning context. "
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


def _context_payload(context: AIContext) -> dict[str, object]:
    return {
        "custom_blocked_terms": [
            {"term": term.term, "severity": term.severity, "category": term.category}
            for term in context.blocked_terms[:80]
        ],
        "allowed_terms": context.allowed_terms[:80],
        "confirmed_bad_examples": context.confirmed_examples[:8],
        "known_false_positive_examples": context.false_positive_examples[:8],
        "recent_auto_flagged_examples": context.auto_examples[:8],
    }


def _payload(message: str, context: AIContext) -> dict[str, object]:
    return {"message": message[:1800], **_context_payload(context)}


def _batch_payload(messages: list[str], context: AIContext) -> dict[str, object]:
    return {
        "messages": [{"index": index, "message": text[:1800]} for index, text in enumerate(messages)],
        **_context_payload(context),
    }


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

    async def classify(self, message: str, context: AIContext) -> ModerationDecision | None:
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
                        "content": json.dumps(_payload(message, context), ensure_ascii=False),
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

    async def classify(self, message: str, context: AIContext) -> ModerationDecision | None:
        LOGGER.debug("Ollama request: model=%s | %.80r", self.model, message)
        try:
            result = await asyncio.to_thread(self._classify_sync, message, context)
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

    def _classify_sync(self, message: str, context: AIContext) -> ModerationDecision | None:
        self.last_error = ""
        body = {
            "model": self.model,
            "stream": False,
            "format": DECISION_SCHEMA,
            "options": {"temperature": 0, "num_ctx": 2048},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(_payload(message, context), ensure_ascii=False)},
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

    async def classify(self, message: str, context: AIContext) -> ModerationDecision | None:
        results = await self.classify_batch([message], context)
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

    async def classify_batch(
        self, messages: list[str], context: AIContext
    ) -> list[ModerationDecision | None]:
        """Classify many messages in one request.

        Returns a list of decisions (or None for items the model didn't return
        a usable verdict for) aligned with the input order. If multiple Groq
        API keys are configured, a key that gets HTTP 429'd is parked in its
        own short cooldown and the request is retried on the next key — so a
        single rate-limited key doesn't take the whole batch down. Only when
        *every* configured key is currently rate limited does this raise
        `RateLimitedError`, so the caller can fall back to a local model.
        """
        if not messages:
            return []

        from openai import BadRequestError, RateLimitError

        self.last_error = ""
        payload = json.dumps(_batch_payload(messages, context), ensure_ascii=False)

        def _build_request_kwargs() -> dict:
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

        attempt_order = self._attempt_order()
        ready_count = sum(
            1 for i in attempt_order
            if self._key_cooldowns[i] is None or self._key_cooldowns[i] <= datetime.now(timezone.utc)
        )
        LOGGER.debug(
            "Groq batch request: %d messages | %d/%d keys available | model=%s | format=%s",
            len(messages), ready_count, self._key_count, self.model,
            "json_schema" if self._use_json_schema else "json_object",
        )
        rate_limited_keys = 0
        best_retry_after: float | None = None
        response = None

        for key_index in attempt_order:
            try:
                response = await self._clients[key_index].chat.completions.create(
                    **_build_request_kwargs()
                )
            except RateLimitError as exc:
                retry_after = _retry_after_seconds(exc)
                self._mark_rate_limited(key_index, retry_after)
                rate_limited_keys += 1
                if retry_after is not None:
                    best_retry_after = retry_after if best_retry_after is None else min(best_retry_after, retry_after)
                LOGGER.warning(
                    "Groq key #%d/%d rate limited; trying next key", key_index + 1, self._key_count
                )
                continue
            except BadRequestError as exc:
                if self._use_json_schema and "json_schema" in str(exc).lower():
                    LOGGER.warning(
                        "Model %s does not support json_schema; switching to json_object for all future requests",
                        self.model,
                    )
                    self._use_json_schema = False
                    try:
                        response = await self._clients[key_index].chat.completions.create(
                            **_build_request_kwargs()
                        )
                    except Exception:
                        LOGGER.exception("Groq moderation batch request failed after json_object fallback")
                        self.last_error = "Groq 요청 실패"
                        return [None] * len(messages)
                    self._mark_success(key_index)
                    break
                LOGGER.error("Groq bad request: %s", exc)
                self.last_error = f"Groq 잘못된 요청: {exc}"
                return [None] * len(messages)
            except Exception:
                LOGGER.exception("Groq moderation batch request failed")
                self.last_error = "Groq 요청 실패"
                return [None] * len(messages)
            else:
                self._mark_success(key_index)
                break

        if response is None:
            suffix = f" ({best_retry_after:.0f}초 후 재시도)" if best_retry_after else ""
            if self._key_count > 1:
                self.last_error = f"Groq 키 {self._key_count}개가 모두 요청 제한에 걸렸습니다{suffix}"
            else:
                self.last_error = f"Groq 요청 제한에 걸렸습니다{suffix}"
            LOGGER.warning("%s", self.last_error)
            raise RateLimitedError(best_retry_after, self.last_error)

        if rate_limited_keys:
            LOGGER.info(
                "Groq batch succeeded after rotating past %d rate-limited key(s)", rate_limited_keys
            )

        try:
            content = response.choices[0].message.content or "{}"
            raw_decisions = _extract_json_object(content).get("decisions", [])
        except Exception:
            LOGGER.exception("Groq moderation batch response parsing failed")
            self.last_error = "Groq 응답 파싱 실패"
            return [None] * len(messages)

        results: list[ModerationDecision | None] = [None] * len(messages)
        for entry in raw_decisions:
            try:
                index = int(entry["index"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= index < len(messages) and results[index] is None:
                try:
                    results[index] = _decision_from_data(entry, "groq")
                except Exception:
                    LOGGER.exception("Groq moderation batch item parsing failed (index=%s)", index)

        missing = sum(1 for item in results if item is None)
        if missing:
            self.last_error = f"Groq 배치 응답에서 {missing}/{len(messages)}개 판정 누락"
            LOGGER.warning("%s", self.last_error)
        return results
