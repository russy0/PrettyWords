from __future__ import annotations

import asyncio
import json
import logging
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .filtering import ModerationDecision, ModerationTerm


LOGGER = logging.getLogger(__name__)

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "violation": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "severity": {"type": "integer", "minimum": 0, "maximum": 3},
        "categories": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
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
    "Do not flag benign education, quoting for moderation, reclaimed terms, or clean jokes. "
    "Use custom blocked terms as strong policy, allowed terms as exceptions, and examples as learning context. "
    "Return only JSON matching the schema. No markdown."
)


@dataclass(slots=True)
class AIContext:
    blocked_terms: list[ModerationTerm] = field(default_factory=list)
    allowed_terms: list[str] = field(default_factory=list)
    confirmed_examples: list[str] = field(default_factory=list)
    false_positive_examples: list[str] = field(default_factory=list)
    auto_examples: list[str] = field(default_factory=list)


def _payload(message: str, context: AIContext) -> dict[str, object]:
    return {
        "message": message[:1800],
        "custom_blocked_terms": [
            {"term": term.term, "severity": term.severity} for term in context.blocked_terms[:80]
        ],
        "allowed_terms": context.allowed_terms[:80],
        "confirmed_bad_examples": context.confirmed_examples[:8],
        "known_false_positive_examples": context.false_positive_examples[:8],
        "recent_auto_flagged_examples": context.auto_examples[:8],
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
        try:
            return await asyncio.to_thread(self._classify_sync, message, context)
        except Exception as exc:
            self.last_error = f"{exc.__class__.__name__}: {exc}"
            LOGGER.exception("Ollama moderation request failed")
            return None

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
            self.last_error = f"Ollama timeout after {self.timeout_seconds:.0f}s"
            LOGGER.warning("%s", self.last_error)
            return None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self.last_error = f"Ollama HTTP {exc.code}: {detail[:180]}"
            LOGGER.error("%s", self.last_error)
            return None
        except urllib.error.URLError as exc:
            self.last_error = f"Ollama connection failed: {exc.reason}"
            LOGGER.warning("%s", self.last_error)
            return None

        content = json.loads(raw).get("message", {}).get("content", "{}")
        return _decision_from_data(_extract_json_object(str(content)), "ollama")
