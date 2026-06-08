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

CATEGORY_LABELS = {
    "profanity": "욕설",
    "sexual": "성적발언",
    "family_insult": "패드립",
    "harassment": "괴롭힘/모욕",
    "hate": "혐오/차별",
    "threat": "위협",
    "other": "기타",
}

CATEGORY_ALIASES = {
    "욕설": "profanity",
    "비속어": "profanity",
    "profanity": "profanity",
    "sexual": "sexual",
    "성적": "sexual",
    "성적발언": "sexual",
    "섹드립": "sexual",
    "패드립": "family_insult",
    "family": "family_insult",
    "family_insult": "family_insult",
    "괴롭힘": "harassment",
    "모욕": "harassment",
    "harassment": "harassment",
    "혐오": "hate",
    "차별": "hate",
    "hate": "hate",
    "위협": "threat",
    "협박": "threat",
    "threat": "threat",
    "기타": "other",
    "other": "other",
}

DEFAULT_BLOCKED_TERMS = (
    # \u2500\u2500 \uc2dc\ubc1c \uacc4\uc5f4 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ("\uc2dc\ubc1c", 3),
    ("\uc528\ubc1c", 3),
    ("\uc529\ubc1c", 3),
    ("\uc528\ud314", 3),   # \ud314 \uc6b0\ud68c
    ("\uc2dc\ud314", 3),
    ("\uc528\ube68", 3),   # \ube68 \uc6b0\ud68c
    ("\uc2dc\ube68", 3),
    ("\u3145\u3142", 2),
    ("\u3146\u3142", 2),
    # \u2500\u2500 \ubcd1\uc2e0 \uacc4\uc5f4 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ("\ubcd1\uc2e0", 3),
    ("\ube05\uc2e0", 3),   # \ubaa8\uc74c \uc6b0\ud68c (\u315b\u2192\u3160)
    ("\ubd79\uc2e0", 3),
    ("\ubcd1\uc270", 3),   # \uc270 \uc6b0\ud68c
    ("\u3142\u3145", 2),
    # \u2500\u2500 \uac1c\uc0c8\ub07c \uacc4\uc5f4 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ("\uac1c\uc0c8\ub07c", 3),
    ("\uac1c\uc0c8", 3),
    ("\uac1c\uc0c9\uae30", 3),  # \uc0c8\ub07c \uc6b0\ud68c
    ("\uac1c\uc138\ub07c", 3),
    ("\uac1c\uc250\ub07c", 3),
    # \u2500\u2500 \uc0c8\ub07c/\uc0c9\uae30 \uacc4\uc5f4 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ("\uc0c8\ub07c", 2),
    ("\uc0c9\uae30", 2),   # \uac00\uc7a5 \ud754\ud55c \uc0c8\ub07c \uc6b0\ud68c
    ("\uc138\ub07c", 2),
    ("\uc0c9\ud788", 2),
    # \u2500\u2500 \uc874\ub098/\uc870\ub0b8 \uacc4\uc5f4 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ("\uc874\ub098", 2),
    ("\uc870\ub0b8", 2),   # \uc874\ub098 \uc6b0\ud68c
    ("\ucac0\ub098", 2),
    ("\u3148\u3134", 2),
    # \u2500\u2500 \uc9c0\ub784 \uacc4\uc5f4 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ("\uc9c0\ub784", 2),
    ("\u3148\u3139", 2),
    # \u2500\u2500 \uc878\ub77c \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ("\uc878\ub77c", 1),
    # \u2500\u2500 \uc8f5/\uc8f6 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ("\uc8f5", 3),
    ("\uc8f6", 2),
    # \u2500\u2500 \ub2e5\uccd0/\uaebc\uc838 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ("\ub2e5\uccd0", 2),
    ("\uaebc\uc838", 2),
    # \u2500\u2500 \ubbf8\uce5c \uacc4\uc5f4 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ("\ubbf8\uce5c", 2),
    ("\ubbf8\uce70", 2),   # \ubbf8\uce5c \ud0a4\ubcf4\ub4dc \uc6b0\ud68c
    ("\u3141\u314a", 2),
    ("\ubbf8\uce5c\ub188", 3),
    ("\ubbf8\uce5c\ub144", 3),
    ("\ubbf8\uce5c\uc0c8\ub07c", 3),
    # \u2500\u2500 \uc5ff\uba39\uc5b4/\uc560\ubbf8/\ub290\uae08 \uacc4\uc5f4 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ("\uc5ff\uba39\uc5b4", 2),
    ("\uc5ff", 1),
    ("\uc560\ubbf8", 3),
    ("\ub290\uae08", 3),
    ("\ub290\uadf8\ubbf8", 3),
    ("\ub2c8\uc560\ubbf8", 3),  # \ub290\uae08\ub9c8 \uacc4\uc5f4 \uc6b0\ud68c
    ("\ub2c8\uc5d0\ubbf8", 3),
    # \u2500\u2500 \uc601\uc5b4 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    ("fuck", 3),
    ("fck", 2),
    ("shit", 2),
    ("bitch", 3),
    ("asshole", 3),
    ("dick", 2),
    ("idiot", 1),
    ("moron", 1),
)

SEXUAL_SEED_TERMS = {
    "\uc8f5",
    "\uc8f6",
    "dick",
}

FAMILY_INSULT_SEED_TERMS = {
    "\uc560\ubbf8",
    "\ub290\uae08",
    "\ub290\uadf8\ubbf8",
    "\ub2c8\uc560\ubbf8",
    "\ub2c8\uc5d0\ubbf8",
}

HARASSMENT_SEED_TERMS = {
    "\ubcd1\uc2e0",
    "\ube05\uc2e0",
    "\ubd79\uc2e0",
    "\ubcd1\uc270",
    "\u3142\u3145",
    "\uac1c\uc0c8\ub07c",
    "\uac1c\uc0c8",
    "\uac1c\uc0c9\uae30",
    "\uac1c\uc138\ub07c",
    "\uac1c\uc250\ub07c",
    "\uc0c8\ub07c",
    "\uc0c9\uae30",
    "\uc138\ub07c",
    "\uc0c9\ud788",
    "\ubbf8\uce5c\ub188",
    "\ubbf8\uce5c\ub144",
    "\ubbf8\uce5c\uc0c8\ub07c",
    "bitch",
    "asshole",
    "idiot",
    "moron",
}


def normalize_category(category: str | None) -> str:
    if not category:
        return "profanity"
    return CATEGORY_ALIASES.get(category.strip().lower(), category.strip().lower() or "profanity")


def category_label(category: str) -> str:
    normalized = normalize_category(category)
    return CATEGORY_LABELS.get(normalized, normalized)


def seed_category(term: str) -> str:
    if term in SEXUAL_SEED_TERMS:
        return "sexual"
    if term in FAMILY_INSULT_SEED_TERMS:
        return "family_insult"
    if term in HARASSMENT_SEED_TERMS:
        return "harassment"
    return "profanity"


@dataclass(frozen=True, slots=True)
class ModerationTerm:
    term: str
    severity: int = 2
    notes: str = ""
    category: str = "profanity"


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
                    reason="허용어 또는 허용 문구와 일치했습니다.",
                    source="local",
                )

        all_terms = list(blocked_terms)
        all_terms.extend(
            ModerationTerm(term, severity, category=seed_category(term)) for term, severity in DEFAULT_BLOCKED_TERMS
        )

        matched: list[ModerationTerm] = []
        for term in all_terms:
            if _contains(normalized, compacted, term.term):
                matched.append(term)

        if not matched:
            return ModerationDecision(
                violation=False,
                confidence=0.2,
                severity=0,
                reason="로컬 금지어와 일치하지 않았습니다.",
                source="local",
            )

        severity = max(max(term.severity, 1) for term in matched)
        categories = tuple(dict.fromkeys(normalize_category(term.category) for term in matched))
        confidence = min(0.99, 0.68 + (0.08 * len(matched)) + (0.07 * severity))
        return ModerationDecision(
            violation=True,
            confidence=confidence,
            severity=severity,
            categories=categories or ("profanity",),
            matched_terms=tuple(term.term for term in matched[:8]),
            reason="정규화 후 금지어와 일치했습니다.",
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
            reason=f"AI가 문맥상 오탐으로 판단했습니다: {ai.reason}",
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
