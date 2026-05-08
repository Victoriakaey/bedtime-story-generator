import time
from typing import Any, Dict, List, Optional

from bedtime.config import MAX_REVISION_PASSES
from bedtime.model_client import call_llm, current_model, current_provider
from bedtime.prompts import (
    build_critic_prompt,
    build_generator_prompt,
    build_revision_prompt,
    build_story_brief_prompt,
)
from bedtime.run_logger import RunLogger
from bedtime.story_utils import (
    apply_deterministic_branded_check,
    apply_deterministic_safety_checks,
    critique_passes,
    extract_json,
    fallback_brief,
    fallback_critique,
    normalize_brief,
    normalize_critique,
)


def build_story_brief(user_request: str) -> Dict[str, Any]:
    prompt = build_story_brief_prompt(user_request)
    response = call_llm(prompt, max_tokens=800, temperature=0.1)
    try:
        return normalize_brief(extract_json(response), user_request)
    except ValueError:
        return fallback_brief(user_request)


def generate_story(brief: Dict[str, Any]) -> str:
    prompt = build_generator_prompt(brief)
    return call_llm(prompt, max_tokens=1400, temperature=0.7).strip()


def judge_story(brief: Dict[str, Any], story: str) -> Dict[str, Any]:
    prompt = build_critic_prompt(brief, story)
    response = call_llm(prompt, max_tokens=900, temperature=0.0)
    try:
        critique = normalize_critique(extract_json(response))
    except ValueError as exc:
        critique = fallback_critique(str(exc))
    critique = apply_deterministic_safety_checks(critique, story)
    critique = apply_deterministic_branded_check(critique, story)
    return critique


def revise_story(brief: Dict[str, Any], story: str, critique: Dict[str, Any]) -> str:
    prompt = build_revision_prompt(brief, story, critique)
    return call_llm(prompt, max_tokens=1500, temperature=0.6).strip()


def run_step(
    logger: RunLogger,
    name: str,
    kind: str,
    input_data: Any,
    fn: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    start_time = time.time()
    try:
        output = fn()
    except Exception as exc:
        logger.span(name, kind, input_data, None, start_time, status="error", metadata=metadata, error_message=str(exc))
        logger.section(f"Error: {name}", str(exc))
        raise
    logger.span(name, kind, input_data, output, start_time, metadata=metadata)
    return output


def run_story_pipeline(user_request: str, max_revisions: int = MAX_REVISION_PASSES, log_enabled: bool = True) -> Dict[str, Any]:
    logger = RunLogger(enabled=log_enabled)
    run_start = time.time()
    critiques: List[Dict[str, Any]] = []
    stories: List[str] = []

    logger.section(
        "Run Metadata",
        {
            "trace_id": logger.trace_id,
            "provider": current_provider(),
            "model": current_model(),
            "user_request": user_request,
            "max_revision_passes": max_revisions,
        },
    )

    brief = run_step(
        logger,
        "build_story_brief",
        "llm",
        {"user_request": user_request},
        lambda: build_story_brief(user_request),
    )
    logger.section("Story Brief", brief)

    story = run_step(
        logger,
        "generate_story",
        "llm",
        {"brief": brief},
        lambda: generate_story(brief),
        metadata={"revision_round": 0},
    )
    stories.append(story)
    logger.section("Draft Story: Round 0", story)

    revision_count = 0
    for round_index in range(max_revisions + 1):
        critique = run_step(
            logger,
            "judge_story",
            "evaluator",
            {"brief": brief, "story": story},
            lambda: judge_story(brief, story),
            metadata={"revision_round": round_index},
        )
        critiques.append(critique)
        logger.section(f"Critic Output: Round {round_index}", critique)

        if critique_passes(critique):
            break

        if round_index >= max_revisions:
            break

        revision_count += 1
        logger.section(
            f"Revision Feedback: Round {revision_count}",
            {
                "strengths": critique.get("strengths", []),
                "revision_suggestions": critique.get("revision_suggestions", []),
            },
        )
        story = run_step(
            logger,
            "revise_story",
            "llm",
            {"brief": brief, "story": story, "critique": critique},
            lambda: revise_story(brief, story, critique),
            metadata={"revision_round": revision_count},
        )
        stories.append(story)
        logger.section(f"Revised Story: Round {revision_count}", story)

    passed = critique_passes(critiques[-1]) if critiques else False
    duration_ms = round((time.time() - run_start) * 1000, 2)
    logger.section("Final Story", story)
    logger.section(
        "Run Summary",
        {
            "passed": passed,
            "revision_count": revision_count,
            "total_duration_ms": duration_ms,
            "log_files": logger.paths(),
        },
    )

    return {
        "trace_id": logger.trace_id,
        "provider": current_provider(),
        "model": current_model(),
        "brief": brief,
        "stories": stories,
        "final_story": story,
        "critiques": critiques,
        "revision_count": revision_count,
        "passed": passed,
        "duration_ms": duration_ms,
        "logs": logger.paths(),
    }


def apply_user_feedback(brief: Dict[str, Any], current_story: str, feedback: str) -> Dict[str, Any]:
    """Apply one round of user-driven feedback to an existing story.

    The user's feedback (e.g., "make it shorter", "make it funnier") is wrapped
    in a synthetic critique whose only revision_suggestion is the user's words.
    The reviser produces a new draft; the critic re-judges it (so safety,
    branded-name, and deterministic checks still run on the user-driven
    revision); if the new draft fails any hard check we discard it and keep
    the previous story.

    Returns a dict with `accepted` (bool), `story` (str — new if accepted, old
    if not), and `critique` (the verdict on whichever story is returned).
    """
    synthetic_critique: Dict[str, Any] = {
        "passes": False,
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
        "strengths": [],
        "revision_suggestions": [f"User feedback after reading the story: {feedback}"],
        "deterministic_safety_flags": [],
    }
    candidate = revise_story(brief, current_story, synthetic_critique)
    verdict = judge_story(brief, candidate)
    if critique_passes(verdict):
        return {"accepted": True, "story": candidate, "critique": verdict}
    return {"accepted": False, "story": current_story, "critique": verdict}
