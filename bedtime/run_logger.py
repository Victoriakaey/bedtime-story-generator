import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from bedtime.config import LOG_DIR
from bedtime.model_client import current_model, current_provider


class RunLogger:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.trace_id = str(uuid.uuid4())
        self.started_at = datetime.now(timezone.utc)
        self.jsonl_path: Optional[Path] = None
        self.text_path: Optional[Path] = None

        if self.enabled:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = self.started_at.strftime("%Y%m%d-%H%M%S")
            prefix = f"{timestamp}-{self.trace_id[:8]}"
            self.jsonl_path = LOG_DIR / f"{prefix}.jsonl"
            self.text_path = LOG_DIR / f"{prefix}.log"

    def span(
        self,
        name: str,
        kind: str,
        input_data: Any,
        output_data: Any,
        start_time: float,
        status: str = "ok",
        metadata: Optional[Dict[str, Any]] = None,
        error_message: str = "",
    ) -> None:
        if not self.enabled or self.jsonl_path is None:
            return

        end_time = time.time()
        event = {
            "trace_id": self.trace_id,
            "span_id": str(uuid.uuid4()),
            "parent_span_id": None,
            "name": name,
            "kind": kind,
            "provider": current_provider(),
            "model": current_model(),
            "start_time": datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
            "end_time": datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
            "duration_ms": round((end_time - start_time) * 1000, 2),
            "status": status,
            "input": input_data,
            "output": output_data,
            "metadata": metadata or {},
            "error": error_message,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def section(self, title: str, content: Any) -> None:
        if not self.enabled or self.text_path is None:
            return

        with self.text_path.open("a", encoding="utf-8") as f:
            f.write(f"\n=== {title} ===\n")
            if isinstance(content, (dict, list)):
                f.write(json.dumps(content, indent=2, ensure_ascii=False))
            else:
                f.write(str(content))
            f.write("\n")

    def paths(self) -> Dict[str, Optional[str]]:
        return {
            "jsonl": str(self.jsonl_path) if self.jsonl_path else None,
            "human_log": str(self.text_path) if self.text_path else None,
        }
