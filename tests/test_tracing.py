"""Local JSONL traces, with remote sinks opt-in."""

import json

import config
from src.core import tracing
from src.core.tracing import Tracer


def test_trace_is_written_locally(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)
    monkeypatch.setattr(config, "TRACE_ENABLED", True)

    tracer = Tracer("test.run", {"drug": "SHP099"})
    tracer.event("planner", {"message": "planned"})
    record = tracer.end({"verdict": "SUPPORTED"})

    lines = (tmp_path / "runs.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["run_id"] == record["run_id"]
    assert stored["events"][0]["name"] == "planner"
    assert stored["outputs"]["verdict"] == "SUPPORTED"
    assert "local-jsonl" in stored["sinks"]


def test_remote_sinks_need_keys(monkeypatch):
    monkeypatch.setattr(config, "LANGSMITH_API_KEY", "")
    monkeypatch.setattr(config, "LANGFUSE_PUBLIC_KEY", "")
    assert tracing.sinks() == ["local-jsonl"]


def test_sinks_report_configured_backends(monkeypatch):
    monkeypatch.setattr(config, "LANGSMITH_API_KEY", "fake-key")
    assert "langsmith" in tracing.sinks()


def test_pipeline_writes_one_trace(tmp_path, monkeypatch):
    from src.core.schemas import Hypothesis
    from src.agents.workflow import run_pipeline

    monkeypatch.setattr(config, "TRACE_DIR", tmp_path)
    run_pipeline(Hypothesis("SHP099", "SHP2", smiles="Nc1nc(N2CCNCC2)c2cc(Cl)ccc2n1"))

    lines = (tmp_path / "runs.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["inputs"]["engine"] in ("builtin", "langgraph")
