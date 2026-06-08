from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field


ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)
LEET_TABLE = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    }
)

DEFAULT_BLOCKED_TERMS = (
    ("시발", 3),
    ("씨발", 3),
    ("ㅅㅂ", 2),
    ("병신", 3),
    ("ㅂㅅ", 2),
    ("좆", 3),
    ("닥쳐", 2),
    ("꺼져", 2),
    ("fuck", 3),
    ("shit", 2),
    ("bitch", 3),
)


@dataclass(frozen=True, slots=True)
class ModerationTerm:
    term: str
    severity: int = 2
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ModerationDecision:
    violation: bool
    confidence: float
    severity: int
    categories: tuple[str, ...] = field(default_factory=tuple)
    matched_terms: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""
    source: str = "local"
    suggested_action: str = "none"

    def to_dict(self) -> dict[str, object]:
        return {
            "violation": self.violation,
            "confidence": round(max(0.0, min(1.0, self.confidence)), 4),
            "severity": max(0, min(3, self.severity)),
            "categories": list(self.categories),
            "matched_terms": list(self.matched_terms),
            "reason": self.reason,
            "source": self.source,
            "suggested_action": self.suggested_action,
        }


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(ZERO_WIDTH).lower()
    return normalized.translate(LEET_TABLE)


def compact_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", normalize_text(text), flags=re.UNICODE)


def collapse_repeats(text: str) -> str:
    return re.sub(r"(.)\1+", r"\1", text)


def message_fingerprint(text: str) -> str:
    stable = collapse_repeats(compact_text(text))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _contains(haystack: str, compact_haystack: str, needle: str) -> bool:
    needle_norm = collapse_repeats(normalize_text(needle))
    needle_compact = collapse_repeats(compact_text(needle))
    if not needle_norm or not needle_compact:
        return False
    return needle_norm in haystack or needle_compact in compact_haystack


class LocalClassifier:
    """Small adaptive filter used before/after AI and when AI is unavailable."""

    def classify(
        self,
        message: str,
        blocked_terms: list[ModerationTerm],
        allowed_terms: list[str],
    ) -> ModerationDecision:
        normalized = collapse_repeats(normalize_text(message))
        compacted = collapse_repeats(compact_text(message))

        for term in allowed_terms:
            if _contains(normalized, compacted, term):
                return ModerationDecision(
                    violation=False,
                    confidence=0.95,
                    severity=0,
                    categories=("allowlist",),
                    matched_terms=(term,),
                    reason="Allowed term or phrase matched.",
                    source="local",
                )

        all_terms = list(blocked_terms)
        all_terms.extend(ModerationTerm(term, severity) for term, severity in DEFAULT_BLOCKED_TERMS)

        matched: list[ModerationTerm] = []
        for term in all_terms:
            if _contains(normalized, compacted, term.term):
                matched.append(term)

        if not matched:
            return ModerationDecision(
                violation=False,
                confidence=0.2,
                severity=0,
                reason="No local term match.",
                source="local",
            )

        severity = max(max(term.severity, 1) for term in matched)
        confidence = min(0.99, 0.68 + (0.08 * len(matched)) + (0.07 * severity))
        return ModerationDecision(
            violation=True,
            confidence=confidence,
            severity=severity,
            categories=("profanity", "manual_or_seed_term"),
            matched_terms=tuple(term.term for term in matched[:8]),
            reason="Blocked term matched after normalization.",
            source="local",
            suggested_action="timeout" if severity >= 2 else "delete",
        )


def combine_decisions(
    local: ModerationDecision,
    ai: ModerationDecision | None,
    threshold: float,
) -> ModerationDecision:
    if ai is None:
        return local

    if local.violation and not ai.violation and ai.confidence >= 0.86:
        return ModerationDecision(
            violation=False,
            confidence=ai.confidence,
            severity=0,
            categories=tuple(dict.fromkeys((*local.categories, *ai.categories, "ai_context_override"))),
            matched_terms=local.matched_terms,
            reason=f"AI judged local match as contextual false positive: {ai.reason}",
            source="local+ai",
            suggested_action="review",
        )

    if local.violation and (ai.violation or local.confidence >= threshold):
        return ModerationDecision(
            violation=True,
            confidence=max(local.confidence, ai.confidence),
            severity=max(local.severity, ai.severity),
            categories=tuple(dict.fromkeys((*local.categories, *ai.categories))),
            matched_terms=tuple(dict.fromkeys((*local.matched_terms, *ai.matched_terms))),
            reason=ai.reason if ai.violation else local.reason,
            source="local+ai",
            suggested_action="timeout",
        )

    if ai.violation and ai.confidence >= threshold:
        return ai

    return ModerationDecision(
        violation=False,
        confidence=max(local.confidence if not local.violation else 0.0, ai.confidence if not ai.violation else 0.0),
        severity=0,
        categories=tuple(dict.fromkeys((*local.categories, *ai.categories))),
        matched_terms=tuple(dict.fromkeys((*local.matched_terms, *ai.matched_terms))),
        reason=ai.reason or local.reason,
        source="local+ai",
        suggested_action="none",
    )
