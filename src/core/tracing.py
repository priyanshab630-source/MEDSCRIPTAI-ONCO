"""Run-level observability.

Three sinks, all optional:
  * local JSONL  - always written (traces/runs.jsonl), greppable, no account needed
  * LangSmith    - when LANGSMITH_API_KEY is set
  * Langfuse     - when LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are set

Both remote sinks are best-effort: a network or schema problem is logged and the
run continues, because observability must never break the pipeline.
"""

import sys
import json
import time
import uuid
from datetime import datetime, timezone

import requests

import config


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


class Tracer:
    def __init__(self, name: str, inputs: dict = None):
        self.name = name
        self.inputs = inputs or {}
        self.run_id = str(uuid.uuid4())
        self.started = time.time()
        self.events = []
        self.sinks = ["local-jsonl"]
        if config.LANGSMITH_API_KEY:
            self.sinks.append("langsmith")
        if config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY:
            self.sinks.append("langfuse")

    def event(self, name: str, payload: dict = None):
        self.events.append({
            "name": name,
            "offset_ms": int((time.time() - self.started) * 1000),
            "payload": payload or {},
        })
        return self

    def end(self, outputs: dict = None) -> dict:
        finished = time.time()
        record = {
            "run_id": self.run_id,
            "name": self.name,
            "inputs": self.inputs,
            "outputs": outputs or {},
            "events": self.events,
            "started_at": _iso(self.started),
            "ended_at": _iso(finished),
            "duration_ms": int((finished - self.started) * 1000),
            "sinks": self.sinks,
        }
        if config.TRACE_ENABLED:
            self._write_local(record)
        self._send_langsmith(record)
        self._send_langfuse(record)
        return record

    # ------------------------------------------------------------------ sinks
    def _write_local(self, record: dict) -> None:
        try:
            config.TRACE_DIR.mkdir(parents=True, exist_ok=True)
            with open(config.TRACE_DIR / "runs.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError as exc:
            print(f"[tracing] could not write local trace ({exc})", file=sys.stderr)

    def _send_langsmith(self, record: dict) -> None:
        if not config.LANGSMITH_API_KEY:
            return
        payload = {
            "id": record["run_id"],
            "name": record["name"],
            "run_type": "chain",
            "inputs": record["inputs"],
            "outputs": record["outputs"],
            "start_time": record["started_at"],
            "end_time": record["ended_at"],
            "session_name": config.LANGSMITH_PROJECT,
            "extra": {"events": record["events"]},
        }
        try:
            requests.post(
                "https://api.smith.langchain.com/runs",
                json=payload,
                headers={"x-api-key": config.LANGSMITH_API_KEY,
                         "Content-Type": "application/json"},
                timeout=10,
            )
        except Exception as exc:                 # noqa: BLE001
            print(f"[tracing] LangSmith upload failed ({type(exc).__name__})", file=sys.stderr)

    def _send_langfuse(self, record: dict) -> None:
        if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
            return
        batch = [{
            "type": "trace-create",
            "timestamp": record["ended_at"],
            "body": {
                "id": record["run_id"],
                "name": record["name"],
                "input": record["inputs"],
                "output": record["outputs"],
                "metadata": {"events": record["events"],
                             "duration_ms": record["duration_ms"]},
            },
        }]
        try:
            requests.post(
                f"{config.LANGFUSE_HOST}/api/public/ingestion",
                json={"batch": batch},
                auth=(config.LANGFUSE_PUBLIC_KEY, config.LANGFUSE_SECRET_KEY),
                timeout=10,
            )
        except Exception as exc:                 # noqa: BLE001
            print(f"[tracing] Langfuse upload failed ({type(exc).__name__})", file=sys.stderr)


def sinks() -> list:
    found = ["local-jsonl"]
    if config.LANGSMITH_API_KEY:
        found.append("langsmith")
    if config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY:
        found.append("langfuse")
    return found
