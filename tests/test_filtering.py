from prettywords.filtering import LocalClassifier, ModerationTerm, combine_decisions, message_fingerprint


def test_local_classifier_detects_compacted_registered_term():
    classifier = LocalClassifier()
    decision = classifier.classify("나쁜 말: ㅅ ㅂ", [ModerationTerm("ㅅㅂ", 2)], [])

    assert decision.violation is True
    assert decision.severity >= 2
    assert "ㅅㅂ" in decision.matched_terms


def test_allowed_term_overrides_local_match():
    classifier = LocalClassifier()
    decision = classifier.classify("문서 예시: ㅅㅂ", [ModerationTerm("ㅅㅂ", 2)], ["문서 예시"])

    assert decision.violation is False
    assert "allowlist" in decision.categories


def test_fingerprint_ignores_spacing_and_repeats():
    assert message_fingerprint("ㅅ ㅂ!!!") == message_fingerprint("ㅅㅂ")
    assert message_fingerprint("bad!!!") == message_fingerprint("baaad")


def test_ai_false_positive_can_override_local_match():
    local = LocalClassifier().classify("quote: bad", [ModerationTerm("bad", 2)], [])
    from prettywords.filtering import ModerationDecision

    ai_clean = ModerationDecision(
        violation=False,
        confidence=0.9,
        severity=0,
        categories=("context",),
        reason="quoted educational context",
        source="ai",
    )

    combined = combine_decisions(local, ai_clean, 0.78)
    assert combined.violation is False
    assert combined.suggested_action == "review"
