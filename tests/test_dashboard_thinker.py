import json
import sys
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dashboard.backend.main import _summarize_tool_input, _parse_thinker_events


def test_summarize_grep():
    result = _summarize_tool_input("Grep", {"pattern": "probe_below_val", "path": "/long/path/pipeline.py"})
    assert "probe_below_val" in result
    assert "pipeline.py" in result


def test_summarize_read():
    result = _summarize_tool_input("Read", {"file_path": "/some/path/feature_catalog.md"})
    assert "feature_catalog.md" in result


def test_summarize_bash():
    result = _summarize_tool_input("Bash", {"command": "ls research/signals/"})
    assert "ls research" in result


def test_summarize_unknown():
    result = _summarize_tool_input("UnknownTool", {"foo": "bar"})
    assert len(result) <= 100


def test_parse_thinker_events_empty(tmp_path):
    f = tmp_path / "session.jsonl"
    f.write_text("")
    events = _parse_thinker_events(f)
    assert events == []


def test_parse_thinker_events_text(tmp_path):
    f = tmp_path / "session.jsonl"
    entry = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "Now I understand the pattern."}]
        }
    }
    f.write_text(json.dumps(entry) + "\n")
    events = _parse_thinker_events(f)
    assert len(events) == 1
    assert events[0]["type"] == "text"
    assert "understand" in events[0]["content"]


def test_parse_thinker_events_tool_call(tmp_path):
    f = tmp_path / "session.jsonl"
    entry = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_01",
                "name": "Grep",
                "input": {"pattern": "probe_below", "path": "/some/pipeline.py"}
            }]
        }
    }
    f.write_text(json.dumps(entry) + "\n")
    events = _parse_thinker_events(f)
    assert len(events) == 1
    assert events[0]["type"] == "tool_call"
    assert events[0]["tool"] == "Grep"
    assert "probe_below" in events[0]["summary"]


def test_parse_thinker_events_tool_result(tmp_path):
    f = tmp_path / "session.jsonl"
    call_entry = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_01",
                "name": "Grep",
                "input": {"pattern": "x", "path": "/y.py"}
            }]
        }
    }
    result_entry = {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_01",
                "content": "469-  has_prev = pl.col('prev_day_val')"
            }]
        }
    }
    f.write_text(json.dumps(call_entry) + "\n" + json.dumps(result_entry) + "\n")
    events = _parse_thinker_events(f)
    assert any(e["type"] == "tool_result" for e in events)
    result_event = next(e for e in events if e["type"] == "tool_result")
    assert "has_prev" in result_event["summary"]
    assert result_event["tool"] == "Grep"


def test_parse_thinker_events_max_events(tmp_path):
    f = tmp_path / "session.jsonl"
    lines = []
    for i in range(60):
        entry = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": f"message {i}"}]
            }
        }
        lines.append(json.dumps(entry))
    f.write_text("\n".join(lines) + "\n")
    events = _parse_thinker_events(f, max_events=40)
    assert len(events) == 40


def test_parse_thinker_events_ignores_bad_lines(tmp_path):
    f = tmp_path / "session.jsonl"
    f.write_text("not json\n{\"type\": \"other\"}\n")
    events = _parse_thinker_events(f)
    assert events == []
