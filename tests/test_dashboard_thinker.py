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


def test_summarize_write():
    result = _summarize_tool_input("Write", {"file_path": "/some/path/output.py"})
    assert "output.py" in result


def test_summarize_edit():
    result = _summarize_tool_input("Edit", {"file_path": "/some/path/main.py"})
    assert "main.py" in result


from unittest.mock import patch
from fastapi.testclient import TestClient
from src.dashboard.backend.main import app


def test_thinker_endpoint_no_active_session(tmp_path):
    """Returns empty events when no thinker session found."""
    with patch("src.dashboard.backend.main._find_thinker_session_file", return_value=None), \
         patch("src.dashboard.backend.main._find_fallback_session_file", return_value=None):
        client = TestClient(app)
        resp = client.get("/api/autonomy/thinker")
        assert resp.status_code == 200
        data = resp.json()
        assert data["events"] == []
        assert data["is_active"] is False
        assert data["session_id"] is None


def test_thinker_endpoint_with_session(tmp_path):
    """Returns parsed events from an identified session file."""
    f = tmp_path / "abc123.jsonl"
    import json as _json
    entry = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "Analyzing value area features."}]
        }
    }
    f.write_text(_json.dumps(entry) + "\n")

    with patch("src.dashboard.backend.main._find_thinker_session_file", return_value=f):
        client = TestClient(app)
        resp = client.get("/api/autonomy/thinker")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["events"]) == 1
        assert data["events"][0]["type"] == "text"
        assert data["is_active"] is True
        assert data["session_id"] == "abc123"
