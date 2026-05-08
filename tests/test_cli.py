"""CLI behavior test: when the critic refuses after max revisions, the user must
see a refusal message (not the unsafe draft) and the process must exit with code 2.
The unsafe draft is preserved in the run log only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator, List

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main as main_module  # noqa: E402
from bedtime import run_logger, story_pipeline  # noqa: E402


SAFE_BRIEF = json.dumps(
    {
        "premise": "test",
        "main_characters": ["Alice"],
        "target_age": 7,
        "vibe": "cozy",
        "theme_or_lesson": "kindness",
        "required_details": [],
        "avoid": [],
    }
)
DRAFT_STORY = "Once upon a time, Alice slept soundly. The end."
UNSAFE_CRITIQUE = json.dumps(
    {
        "passes": False,
        "hard_checks": {"safe_for_children": False, "age_appropriate": True, "follows_request": True},
        "soft_scores": {"bedtime_tone": 1, "vocabulary_fit": 1, "story_arc": 1, "read_aloud_quality": 1},
        "strengths": [],
        "revision_suggestions": ["Story still depicts an unsafe rocket flight."],
    }
)


def _stub_llm(responses: List[str]):
    iterator: Iterator[str] = iter(responses)

    def fake_call_llm(prompt: str, max_tokens: int = 3000, temperature: float = 0.1) -> str:
        return next(iterator)

    return fake_call_llm


def test_unsafe_run_prints_refusal_and_exits_with_code_2(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    # Pipeline that never passes — forces max-revision exhaustion (default 2 revisions).
    monkeypatch.setattr(
        story_pipeline,
        "call_llm",
        _stub_llm(
            [
                SAFE_BRIEF,
                DRAFT_STORY,
                UNSAFE_CRITIQUE,  # round 0
                DRAFT_STORY,
                UNSAFE_CRITIQUE,  # round 1
                DRAFT_STORY,
                UNSAFE_CRITIQUE,  # round 2 (final)
            ]
        ),
    )
    # config.LOG_DIR was bound at import time; patch run_logger's reference directly.
    monkeypatch.setattr(run_logger, "LOG_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["main.py", "An unsafe rocket request"])

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 2

    captured = capsys.readouterr()
    assert main_module.REFUSAL_MESSAGE in captured.out
    # Story body must NOT reach stdout when the pipeline failed.
    assert DRAFT_STORY not in captured.out
    # Trace id is printed to stderr so the user can find the unsafe draft in logs.
    assert "trace_id=" in captured.err

    # The unsafe draft is preserved in the run log even though the user did not see it.
    log_files = list(tmp_path.glob("*.log"))
    assert len(log_files) == 1
    assert DRAFT_STORY in log_files[0].read_text()


# ---------------------------------------------------------------------------
# Brain-juice idea #2 — interactive follow-up
# ---------------------------------------------------------------------------


REVISED_STORY = "Alice and Bob fell asleep counting tiny stars. The end."
PASS_CRITIQUE = json.dumps(
    {
        "passes": True,
        "hard_checks": {"safe_for_children": True, "age_appropriate": True, "follows_request": True},
        "soft_scores": {"bedtime_tone": 1, "vocabulary_fit": 1, "story_arc": 1, "read_aloud_quality": 1},
        "strengths": ["calm tone"],
        "revision_suggestions": [],
    }
)


def test_interactive_followup_applies_user_feedback_and_reprints(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """--interactive: after the first story prints, one round of user feedback
    triggers a reviser+critic pass; the revised story is printed; then a blank
    line ends the loop."""

    # 7 LLM calls expected:
    # initial run = brief + draft + critique (passes)
    # follow-up   = reviser + critique (passes)
    monkeypatch.setattr(
        story_pipeline,
        "call_llm",
        _stub_llm(
            [
                SAFE_BRIEF,
                DRAFT_STORY,
                PASS_CRITIQUE,
                REVISED_STORY,    # reviser output
                PASS_CRITIQUE,    # critic re-judges revision
            ]
        ),
    )
    monkeypatch.setattr(run_logger, "LOG_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["main.py", "--interactive", "Alice and Bob"])

    # Simulate user typing "make it shorter" then pressing Enter on a blank line.
    inputs = iter(["make it shorter", ""])
    monkeypatch.setattr("builtins.input", lambda *args, **kwargs: next(inputs))

    main_module.main()

    captured = capsys.readouterr()
    assert DRAFT_STORY in captured.out          # initial story printed
    assert "Follow-up mode" in captured.out     # follow-up banner
    assert REVISED_STORY in captured.out        # revised story printed
    assert "--- Revised story ---" in captured.out
