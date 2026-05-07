"""Unit tests for story_utils — the deterministic, code-only safety layer.

These tests do NOT call any LLM. They protect the invariants that the LLM cannot:
- the deterministic blocklist actually fires on real story phrasings
- `deterministic_safety_flags` reflects code, never LLM hallucination
- the brief / critique normalization handles malformed inputs without crashing
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bedtime.story_utils import (  # noqa: E402
    apply_deterministic_branded_check,
    apply_deterministic_safety_checks,
    clamp_age,
    critique_passes,
    extract_age_hint,
    extract_json,
    fallback_critique,
    normalize_brief,
    normalize_critique,
    normalize_required_details,
)


def _safe_critique() -> dict:
    return {
        "passes": True,
        "hard_checks": {
            "safe_for_children": True,
            "age_appropriate": True,
            "follows_request": True,
        },
        "soft_scores": {
            "bedtime_tone": 1,
            "vocabulary_fit": 1,
            "story_arc": 1,
            "read_aloud_quality": 1,
        },
    }


@pytest.mark.parametrize(
    "phrase",
    [
        "drank the potion",
        "lifted off the ground",
        "climbed onto the windowsill",
        "tiptoed out of bed",
        "into the dense trees",
        "just taking a little nap",
    ],
)
def test_blocklist_catches_known_unsafe_phrasings(phrase: str) -> None:
    story = f"Once upon a time, {phrase} and then the night settled in."
    result = apply_deterministic_safety_checks(_safe_critique(), story)
    assert result["passes"] is False
    assert result["hard_checks"]["safe_for_children"] is False
    assert phrase in result["deterministic_safety_flags"]


def test_blocklist_passes_clean_story() -> None:
    story = (
        "Maya curled under her blanket as her grown-up read about a tiny owl who hummed "
        "lullabies to the stars. The room glowed softly and the whole world felt safe."
    )
    result = apply_deterministic_safety_checks(_safe_critique(), story)
    assert result["passes"] is True
    assert result["deterministic_safety_flags"] == []


def test_deterministic_safety_flags_overwrites_llm_hallucination() -> None:
    """Even if the critic invents this field, code must replace it."""
    forged = _safe_critique()
    forged["deterministic_safety_flags"] = ["drank the potion"]  # not actually in story

    clean_story = "The bunny waved goodnight and the moon smiled back."
    result = apply_deterministic_safety_checks(forged, clean_story)

    assert result["deterministic_safety_flags"] == []
    assert result["passes"] is True


def test_blocklist_is_case_insensitive() -> None:
    story = "The Rocket LIFTED OFF THE GROUND in a flash."
    result = apply_deterministic_safety_checks(_safe_critique(), story)
    assert "lifted off the ground" in result["deterministic_safety_flags"]


@pytest.mark.parametrize("raw,expected", [(3, 5), (7, 7), (13, 10), ("not a number", 7)])
def test_clamp_age_clamps_to_supported_range(raw, expected) -> None:
    assert clamp_age(raw) == expected


def test_normalize_brief_replaces_branded_terms() -> None:
    raw = {"premise": "Elsa visits Hogwarts", "main_characters": ["Elsa", "Spider-Man"]}
    brief = normalize_brief(raw, user_request="Elsa visits Hogwarts")
    assert "Elsa" not in brief["premise"]
    assert "Hogwarts" not in brief["premise"]
    assert "Spider-Man" not in brief["main_characters"]


def test_normalize_required_details_filters_unsafe_terms() -> None:
    request = "A story with glowing berries and the kids eating them"
    details = ["glowing berries", "children eating the berries", "purple sky"]
    filtered = normalize_required_details(details, request)
    assert "glowing berries" in filtered
    assert "children eating the berries" not in filtered


def test_normalize_critique_fail_closed_on_missing_keys() -> None:
    """Empty critic output must default every hard check to False, not True."""
    critique = normalize_critique({})
    assert critique["passes"] is False
    for key in ("safe_for_children", "age_appropriate", "follows_request"):
        assert critique["hard_checks"][key] is False


def test_critique_passes_requires_all_hard_checks_and_min_soft_scores() -> None:
    good = {
        "hard_checks": {"safe_for_children": True, "age_appropriate": True, "follows_request": True},
        "soft_scores": {"bedtime_tone": 1, "vocabulary_fit": 1, "story_arc": 1, "read_aloud_quality": 1},
    }
    assert critique_passes(good) is True

    good["soft_scores"]["bedtime_tone"] = 0
    assert critique_passes(good) is False

    good["soft_scores"]["bedtime_tone"] = 1
    good["hard_checks"]["safe_for_children"] = False
    assert critique_passes(good) is False


def test_extract_json_strips_markdown_fence() -> None:
    text = "```json\n{\"premise\": \"x\"}\n```"
    assert extract_json(text) == {"premise": "x"}


def test_fallback_critique_marks_failed_with_reason() -> None:
    crit = fallback_critique("parse error: missing brace")
    assert crit["passes"] is False
    assert any("parse error: missing brace" in s for s in crit["revision_suggestions"])


# ---------------------------------------------------------------------------
# Bug 1 — branded names in story body should fail follows_request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "leaked_name",
    ["Elsa", "Peter Parker", "Hogwarts", "Spider-Man"],
)
def test_branded_check_fails_follows_request_on_leak(leaked_name: str) -> None:
    """Even if the brief is clean, if the generator re-attaches a brand name
    in the story body, the critic verdict must flip follows_request=False."""
    story = f"Once upon a time, a kind hero named {leaked_name} went on an adventure."
    critique = _safe_critique()
    result = apply_deterministic_branded_check(critique, story)

    assert result["passes"] is False
    assert result["hard_checks"]["follows_request"] is False
    assert leaked_name.lower() in result["branded_names_in_story"]


def test_branded_check_passes_clean_story() -> None:
    story = "A kind snow queen and a friendly web-slinging hero played in the garden."
    result = apply_deterministic_branded_check(_safe_critique(), story)
    assert result["passes"] is True
    assert result["branded_names_in_story"] == []


def test_branded_check_overwrites_hallucinated_field() -> None:
    """Same forge-prevention contract as the safety check."""
    forged = _safe_critique()
    forged["branded_names_in_story"] = ["elsa"]  # not actually in story
    clean_story = "The bunny waved goodnight and the moon smiled back."
    result = apply_deterministic_branded_check(forged, clean_story)
    assert result["branded_names_in_story"] == []
    assert result["passes"] is True


# ---------------------------------------------------------------------------
# Bug 2 — out-of-range ages must clamp, not silently fall back to default 7
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "request_text,expected_age",
    [
        ("A bedtime story for a 5-year-old about a turtle", 5),
        ("A story for an 8 year old about a turtle", 8),
        ("A bedtime mystery for a 13-year-old about a clock tower", 13),
        ("A story for a 3-year-old about a sleepy cloud", 3),
        ("Tell me about age 9 and the moon", 9),
        ("A turtle story", None),  # no numeric age
    ],
)
def test_extract_age_hint_pulls_numeric_age(request_text: str, expected_age) -> None:
    assert extract_age_hint(request_text) == expected_age


def test_normalize_brief_overrides_intake_age_with_user_stated_age() -> None:
    """The intake LLM sometimes returns 7 (default) for out-of-range ages.
    The user-stated age must win, then get clamped."""
    raw = {"premise": "x", "target_age": 7}  # what the LLM might return
    request = "A bedtime mystery for a 13-year-old about a clock tower"
    brief = normalize_brief(raw, user_request=request)
    assert brief["target_age"] == 10  # 13 extracted, then clamped to 10


def test_normalize_brief_below_range_clamps_to_5() -> None:
    raw = {"premise": "x", "target_age": 7}
    request = "A story for a 3-year-old about a sleepy cloud"
    brief = normalize_brief(raw, user_request=request)
    assert brief["target_age"] == 5


def test_normalize_brief_keeps_intake_age_when_user_silent() -> None:
    raw = {"premise": "x", "target_age": 8}
    request = "A turtle story"  # no age in user text
    brief = normalize_brief(raw, user_request=request)
    assert brief["target_age"] == 8
