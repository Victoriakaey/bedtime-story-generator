import argparse
import json
import sys

from bedtime.config import MAX_REVISION_PASSES
from bedtime.story_pipeline import run_story_pipeline


REFUSAL_MESSAGE = (
    "I'm sorry — I couldn't write a bedtime story for that request that meets our safety "
    "guidelines. Please try a different request, for example one that does not require a "
    "child to leave home alone, build something that actually flies, or take an unfamiliar "
    "magical food or medicine. The unsafe draft is preserved in the run log for review."
)


"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

I would replace the same-model self-critic with either a cross-model judge or a pairwise comparison so the soft-score column produces real tuning signal rather than a rubber stamp (~100% of critiques score all four soft axes at 1.0 even after rewriting the rubric — see docs/ANALYSIS.md). I would also add an interactive follow-up mode so a parent could ask for a calmer / shorter / funnier revision after reading.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an age-appropriate bedtime story with a critic loop.")
    parser.add_argument("request", nargs="*", help="Story request. If omitted, the script prompts interactively.")
    parser.add_argument("--debug", action="store_true", help="Print the story brief, critic outputs, and log paths.")
    parser.add_argument("--max-revisions", type=int, default=MAX_REVISION_PASSES, help="Maximum critic-driven revision passes.")
    parser.add_argument("--no-log", action="store_true", help="Disable local JSONL and human-readable run logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    user_input = " ".join(args.request).strip()
    if not user_input:
        user_input = input("What kind of story do you want to hear? ").strip()

    if not user_input:
        raise SystemExit("Please provide a bedtime story request.")

    try:
        result = run_story_pipeline(user_input, max_revisions=max(0, args.max_revisions), log_enabled=not args.no_log)
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    if args.debug:
        debug_payload = {
            "trace_id": result["trace_id"],
            "provider": result["provider"],
            "model": result["model"],
            "logs": result["logs"],
            "brief": result["brief"],
            "critiques": result["critiques"],
            "revision_count": result["revision_count"],
            "passed": result["passed"],
            "duration_ms": result["duration_ms"],
        }
        print("=== Debug ===")
        print(json.dumps(debug_payload, indent=2, ensure_ascii=False))
        print()

    if not result["passed"]:
        print(REFUSAL_MESSAGE)
        print(f"\n[trace_id={result['trace_id']}]", file=sys.stderr)
        raise SystemExit(2)

    print(result["final_story"])


if __name__ == "__main__":
    main()
