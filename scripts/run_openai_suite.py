"""Run every prompt-driven case in docs/test_cases.json through the OpenAI path.

Writes per-case logs to logs/openai/ and an index.json that maps test-case id -> trace files.
Also captures the final summary so the comparison step does not need to reparse logs.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force OpenAI path and isolated log dir before importing pipeline modules.
os.environ["USE_LOCAL_MODEL"] = "false"
os.environ.setdefault("LOG_DIR", str(ROOT / "logs" / "openai"))

from bedtime.story_pipeline import run_story_pipeline  # noqa: E402

OUT_DIR = Path(os.environ["LOG_DIR"])
OUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = OUT_DIR / "index.json"


def load_cases() -> list[dict]:
    cases_path = ROOT / "docs" / "test_cases.json"
    data = json.loads(cases_path.read_text())
    return [c for c in data["cases"] if "request" in c]


def main() -> None:
    cases = load_cases()
    print(f"Running {len(cases)} OpenAI cases -> {OUT_DIR}")
    index: list[dict] = []
    suite_start = time.time()

    for i, case in enumerate(cases, 1):
        case_id = case["id"]
        request = case["request"]
        category = case["category"]
        print(f"[{i:>2}/{len(cases)}] {case_id} ({category}) ... ", end="", flush=True)
        case_start = time.time()
        entry: dict = {
            "id": case_id,
            "category": category,
            "request": request,
            "expected": case.get("expected", []),
            "risk_area": case.get("risk_area", []),
        }
        try:
            result = run_story_pipeline(request, max_revisions=2, log_enabled=True)
            entry.update(
                {
                    "status": "ok",
                    "trace_id": result["trace_id"],
                    "logs": result["logs"],
                    "passed": result["passed"],
                    "revision_count": result["revision_count"],
                    "duration_ms": result["duration_ms"],
                    "final_critique": result["critiques"][-1] if result["critiques"] else None,
                    "brief": result["brief"],
                }
            )
            print(
                f"passed={result['passed']} revisions={result['revision_count']} "
                f"({result['duration_ms']:.0f} ms)"
            )
        except Exception as exc:  # pragma: no cover - reporting path
            entry.update(
                {
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "duration_ms": round((time.time() - case_start) * 1000, 2),
                }
            )
            print(f"ERROR: {exc}")

        index.append(entry)
        # Persist after each case so a crash mid-suite does not lose work.
        INDEX_PATH.write_text(json.dumps({"cases": index}, indent=2, ensure_ascii=False))

    suite_duration = round(time.time() - suite_start, 2)
    summary = {
        "total": len(cases),
        "ok": sum(1 for e in index if e.get("status") == "ok"),
        "errors": sum(1 for e in index if e.get("status") == "error"),
        "passed": sum(1 for e in index if e.get("passed") is True),
        "failed_critic": sum(
            1 for e in index if e.get("status") == "ok" and e.get("passed") is False
        ),
        "suite_duration_seconds": suite_duration,
    }
    INDEX_PATH.write_text(
        json.dumps({"summary": summary, "cases": index}, indent=2, ensure_ascii=False)
    )
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
