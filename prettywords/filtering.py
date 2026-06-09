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
    # ── 시발 계열 ────────────────────────────────────────────────────
    ("시발", 3),
    ("씨발", 3),
    ("씩발", 3),
    ("씨팔", 3),
    ("시팔", 3),
    ("씨빨", 3),
    ("시빨", 3),
    ("ㅅㅂ", 2),
    ("ㅆㅂ", 2),
    # ── 병신 계열 ────────────────────────────────────────────────────
    ("병신", 3),
    ("븅신", 3),
    ("뵹신", 3),
    ("병쉰", 3),
    ("ㅂㅅ", 2),
    # ── 개새끼 계열 ──────────────────────────────────────────────────
    ("개새끼", 3),
    ("개새", 3),
    ("개색기", 3),
    ("개세끼", 3),
    ("개쉐끼", 3),
    # ── 새끼/색기 계열 ───────────────────────────────────────────────
    ("새끼", 2),
    ("색기", 2),
    ("세끼", 2),
    ("색히", 2),
    # ── 존나/조낸 계열 ───────────────────────────────────────────────
    ("존나", 2),
    ("조낸", 2),
    ("쫀나", 2),
    ("ㅈㄴ", 2),
    # ── 지랄 계열 ────────────────────────────────────────────────────
    ("지랄", 2),
    ("ㅈㄹ", 2),
    # ── 졸라 ─────────────────────────────────────────────────────────
    ("졸라", 1),
    # ── 죠/죶 ────────────────────────────────────────────────────────
    ("죵", 3),
    ("죶", 2),
    # ── 닥쳐/꺼져 ────────────────────────────────────────────────────
    ("닥쳐", 2),
    ("꺼져", 2),
    # ── 미친 계열 ────────────────────────────────────────────────────
    ("미친", 2),
    ("미칰", 2),
    ("ㅁㅊ", 2),
    ("미친놈", 3),
    ("미친년", 3),
    ("미친새끼", 3),
    # ── 찐따 계열 ────────────────────────────────────────────────────
    ("찐따", 3),
    ("찐다", 2),
    ("등신", 3),
    ("멍청이", 1),
    # ── 엿먹어 계열 ──────────────────────────────────────────────────
    ("엿먹어", 2),
    ("엿", 1),
    # ── 영어 욕설 ────────────────────────────────────────────────────
    ("fuck", 3),
    ("fck", 2),
    ("fucker", 3),
    ("fucking", 2),
    ("shit", 2),
    ("bullshit", 2),
    ("bitch", 3),
    ("asshole", 3),
    ("ass", 1),
    ("bastard", 2),
    ("dumbass", 2),
    ("idiot", 1),
    ("moron", 1),
    ("retard", 3),
    ("retarded", 3),
    ("loser", 1),
    # ── 패드립: 엄마 계열 ─────────────────────────────────────────────
    ("애미", 3),
    ("니애미", 3),
    ("니에미", 3),
    ("느금마", 3),
    ("느그마", 3),
    ("느금", 3),
    ("느그미", 3),
    ("니기미", 3),
    ("이기미", 3),
    ("ㄴㄱㅁ", 2),          # 느금마 자모 줄임
    # ── 패드립: 아빠 계열 ─────────────────────────────────────────────
    ("애비", 3),
    ("니애비", 3),
    ("니에비", 3),
    ("느애비", 3),
    # ── 성적 발언: 성기/행위 계열 ─────────────────────────────────────
    ("좆", 3),
    ("좇", 2),
    ("죳", 2),              # 좆 우회
    ("씹", 3),
    ("씹년", 3),
    ("씹새끼", 3),
    ("씹새", 3),
    ("씹할", 3),
    ("씹치", 3),
    ("씹탱이", 3),
    ("ㅆㅍ", 2),            # 씹 자모 줄임
    ("찌찌", 2),
    ("자위", 2),
    ("딸딸이", 2),
    ("딸치", 2),
    ("ㄷㄷㅇ", 2),
    ("정액", 2),
    ("성교", 3),
    # ── 성적 발언: 성직업/비하 계열 ───────────────────────────────────
    ("창녀", 3),
    ("창년", 3),
    ("갈보", 3),
    ("보빨", 2),
    ("보빠", 2),
    # ── 성적 발언: 미디어 계열 ────────────────────────────────────────
    ("야동", 2),
    ("야설", 2),
    ("포르노", 2),
    ("섹스", 3),
    ("섹드립", 3),
    ("쎅스", 3),
    ("섹", 2),
    ("음란", 2),
    # ── 영어 성적 발언 ────────────────────────────────────────────────
    ("pussy", 3),
    ("cock", 3),
    ("cunt", 3),
    ("slut", 3),
    ("whore", 3),
    ("cum", 2),
    ("porn", 2),
    ("hentai", 2),
    ("anal", 2),
    ("blowjob", 3),
    ("handjob", 3),
    ("masturbat", 2),       # masturbate/masturbation 공통 prefix
    # ── 혐오/차별 계열 ────────────────────────────────────────────────
    ("n-word", 3),
    ("nigger", 3),
    ("nigga", 2),
    ("faggot", 3),
    ("fag", 2),
    ("dyke", 2),
    ("tranny", 3),
    ("chink", 3),
    ("jap", 2),
    ("gook", 3),
)

# seed_category가 올바른 카테고리를 반환하도록 각 집합을 확장합니다.
SEXUAL_SEED_TERMS = {
    "죵", "죶", "좆", "좇", "죳",
    "씹", "씹년", "씹새끼", "씹새", "씹할", "씹치", "씹탱이", "ㅆㅍ",
    "찌찌", "자위", "딸딸이", "딸치", "ㄷㄷㅇ", "정액", "성교",
    "창녀", "창년", "갈보", "걸레", "보빨", "보빠",
    "야동", "야설", "포르노", "섹스", "섹드립", "쎅스", "섹", "음란",
    # 영어
    "dick", "pussy", "cock", "cunt", "slut", "whore", "cum", "porn",
    "hentai", "anal", "blowjob", "handjob", "masturbat",
}

FAMILY_INSULT_SEED_TERMS = {
    "애미", "니애미", "니에미", "어미", "니어미",
    "느금마", "느그마", "느금", "느그미", "니기미", "이기미", "ㄴㄱㅁ",
    "애비", "니애비", "니에비", "느애비",
}

HARASSMENT_SEED_TERMS = {
    "병신", "븅신", "뵹신", "병쉰", "ㅂㅅ",
    "찐따", "찐다", "등신", "멍청이",
    "retard", "retarded", "dumbass",
}

HATE_SEED_TERMS = {
    "n-word", "nigger", "nigga",
    "faggot", "fag", "dyke", "tranny",
    "chink", "jap", "gook",
}

def normalize_category(category: str) -> str:
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
    if term in HATE_SEED_TERMS:
        return "hate"
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
    # needle_norm.strip() 은 공백만 있는 표현이 모든 문장에 매칭되는 것을 방지합니다.
    if not needle_norm.strip() or not needle_compact:
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
