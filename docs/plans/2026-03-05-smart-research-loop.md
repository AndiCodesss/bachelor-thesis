# Smart Research Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the LLM orchestrator so errors are fed back into the next hypothesis and broken generated code gets repaired inline instead of silently discarded.

**Architecture:** Three targeted changes to `scripts/llm_orchestrator.py`: (1) add a new `_collect_orchestrator_feedback_items` function that reads `generation_rejected`/`generation_error` events from the orchestrator's own log, (2) merge all three feedback sources instead of using an either/or fallback, (3) add an inline coder repair loop that retries the coder with the exact validation error injected when generated code fails pre-flight checks.

**Tech Stack:** Python, the existing `LLMRawClient` protocol, `_call_stage_json` and `_normalize_with_semantic_retry` helpers already in the file.

---

### Task 1: Add `_collect_orchestrator_feedback_items` and fix merged feedback

**Files:**
- Modify: `scripts/llm_orchestrator.py:378-416` (after `_collect_feedback_items_from_handoffs`)
- Test: `tests/test_llm_orchestrator.py`

#### Step 1: Write the failing tests

Add to `tests/test_llm_orchestrator.py`:

```python
def test_collect_orchestrator_feedback_items_reads_generation_rejected(tmp_path: Path):
    mod = _load_module()
    log = tmp_path / "llm_orchestrator.jsonl"
    log.write_text(
        json.dumps({
            "event": "generation_rejected",
            "strategy_name": "bad_strat",
            "errors": ["bad_strat: generate_signal failed for tick_610: KeyError: 'ema_ratio_8'"],
            "hypothesis_id": "h001",
        }) + "\n" +
        json.dumps({
            "event": "generation_error",
            "error": "LLMClientError: timeout",
            "iteration": 1,
        }) + "\n" +
        json.dumps({
            "event": "generation_enqueued",
            "strategy_name": "good_strat",
        }) + "\n",
        encoding="utf-8",
    )
    out = mod._collect_orchestrator_feedback_items(log, max_items=10)
    assert len(out) == 2
    events = {x["event"] for x in out}
    assert "generation_rejected" in events
    assert "generation_error" in events
    # generation_enqueued must NOT be included
    assert all(x["event"] != "generation_enqueued" for x in out)


def test_collect_orchestrator_feedback_items_empty_when_no_log(tmp_path: Path):
    mod = _load_module()
    out = mod._collect_orchestrator_feedback_items(tmp_path / "missing.jsonl", max_items=10)
    assert out == []


def test_collect_orchestrator_feedback_items_respects_max_items(tmp_path: Path):
    mod = _load_module()
    log = tmp_path / "llm_orchestrator.jsonl"
    lines = "\n".join(
        json.dumps({"event": "generation_rejected", "strategy_name": f"s{i}", "errors": ["e"]})
        for i in range(10)
    ) + "\n"
    log.write_text(lines, encoding="utf-8")
    out = mod._collect_orchestrator_feedback_items(log, max_items=3)
    assert len(out) == 3


def test_merged_feedback_includes_all_sources(tmp_path: Path):
    """_build_merged_feedback_items merges handoffs + research log + orchestrator log."""
    mod = _load_module()

    handoffs = tmp_path / "handoffs.json"
    handoffs.write_text(json.dumps({
        "schema_version": "1.0", "pending": [],
        "completed": [{
            "handoff_type": "validation_request",
            "payload": {"strategy_name": "s_handoff", "hypothesis_id": "h1"},
            "result": {"overall_verdict": "FAIL", "task_count": 1,
                       "pass_count": 0, "fail_count": 1, "error_count": 0,
                       "avg_sharpe_ratio": -0.5, "avg_trade_count": 10},
        }],
    }), encoding="utf-8")

    research_log = tmp_path / "research.jsonl"
    research_log.write_text(
        json.dumps({"event": "task_error", "strategy_name": "s_error",
                    "error": "ValueError: oops"}) + "\n",
        encoding="utf-8",
    )

    orch_log = tmp_path / "orch.jsonl"
    orch_log.write_text(
        json.dumps({"event": "generation_rejected", "strategy_name": "s_rejected",
                    "errors": ["causality failed"]}) + "\n",
        encoding="utf-8",
    )

    out = mod._build_merged_feedback_items(
        handoffs_path=handoffs,
        research_log_path=research_log,
        orchestrator_log_path=orch_log,
        max_items=40,
    )
    events = [x["event"] for x in out]
    assert "validation_result" in events
    assert "task_error" in events
    assert "generation_rejected" in events


def test_merged_feedback_works_when_some_sources_empty(tmp_path: Path):
    mod = _load_module()
    # Only research log has content
    research_log = tmp_path / "research.jsonl"
    research_log.write_text(
        json.dumps({"event": "task_result", "strategy_name": "s1", "verdict": "FAIL",
                    "bar_config": "tick_610", "metrics": {"sharpe_ratio": -1.0,
                    "trade_count": 5}}) + "\n",
        encoding="utf-8",
    )
    out = mod._build_merged_feedback_items(
        handoffs_path=tmp_path / "missing_handoffs.json",
        research_log_path=research_log,
        orchestrator_log_path=tmp_path / "missing_orch.jsonl",
        max_items=40,
    )
    assert len(out) == 1
    assert out[0]["event"] == "task_result"
```

#### Step 2: Run tests to verify they fail

```bash
uv run pytest tests/test_llm_orchestrator.py::test_collect_orchestrator_feedback_items_reads_generation_rejected tests/test_llm_orchestrator.py::test_merged_feedback_includes_all_sources -v
```
Expected: `AttributeError: module 'llm_orchestrator_module' has no attribute '_collect_orchestrator_feedback_items'`

#### Step 3: Implement `_collect_orchestrator_feedback_items` and `_build_merged_feedback_items`

In `scripts/llm_orchestrator.py`, add after the existing `_collect_feedback_items_from_handoffs` function (after line ~463):

```python
def _collect_orchestrator_feedback_items(
    log_path: Path,
    limit: int = 4000,
    max_items: int = 16,
) -> list[dict[str, Any]]:
    """Collect generation_rejected and generation_error events from orchestrator log."""
    rows = _tail_lines(log_path, limit)
    if not rows:
        return []

    items: list[dict[str, Any]] = []
    for raw in reversed(rows):
        try:
            row = json.loads(raw)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        event = str(row.get("event", ""))
        if event == "generation_rejected":
            errors = row.get("errors", [])
            error_str = str(errors[0]) if isinstance(errors, list) and errors else ""
            items.append({
                "event": "generation_rejected",
                "strategy_name": str(row.get("strategy_name", "")),
                "hypothesis_id": str(row.get("hypothesis_id", "")),
                "error": error_str[:300],
            })
        elif event == "generation_error":
            items.append({
                "event": "generation_error",
                "error": str(row.get("error", ""))[:240],
                "iteration": row.get("iteration"),
            })
        if len(items) >= max_items:
            break

    return items


def _build_merged_feedback_items(
    *,
    handoffs_path: Path,
    research_log_path: Path,
    orchestrator_log_path: Path,
    max_items: int = 40,
) -> list[dict[str, Any]]:
    """Merge feedback from all three sources. Most recent first, capped at max_items."""
    per_source = max(1, max_items // 3)

    handoff_items = _collect_feedback_items_from_handoffs(
        handoffs_path, max_items=per_source
    )
    research_items = _collect_feedback_items(
        research_log_path, max_items=per_source
    )
    orch_items = _collect_orchestrator_feedback_items(
        orchestrator_log_path, max_items=per_source
    )

    # Interleave: orch errors first (most actionable), then validator results, then handoffs
    merged: list[dict[str, Any]] = []
    merged.extend(orch_items)
    merged.extend(research_items)
    merged.extend(handoff_items)
    return merged[:max_items]
```

Then in `main()`, replace lines ~1348-1352 (the either/or feedback collection):

```python
# OLD:
feedback_items = _collect_feedback_items_from_handoffs(handoffs_path)
if not feedback_items:
    feedback_items = _collect_feedback_items(research_log_path)

# NEW:
feedback_items = _build_merged_feedback_items(
    handoffs_path=handoffs_path,
    research_log_path=research_log_path,
    orchestrator_log_path=orchestrator_log_path,
)
```

#### Step 4: Run tests to verify they pass

```bash
uv run pytest tests/test_llm_orchestrator.py::test_collect_orchestrator_feedback_items_reads_generation_rejected tests/test_llm_orchestrator.py::test_collect_orchestrator_feedback_items_empty_when_no_log tests/test_llm_orchestrator.py::test_collect_orchestrator_feedback_items_respects_max_items tests/test_llm_orchestrator.py::test_merged_feedback_includes_all_sources tests/test_llm_orchestrator.py::test_merged_feedback_works_when_some_sources_empty -v
```
Expected: all 5 PASS

#### Step 5: Run full test suite to check regressions

```bash
uv run pytest tests/test_llm_orchestrator.py -v
```
Expected: all existing tests still PASS

#### Step 6: Commit

```bash
git add scripts/llm_orchestrator.py tests/test_llm_orchestrator.py
git commit -m "feat: merge all three feedback sources into orchestrator loop"
```

---

### Task 2: Add inline coder repair loop

**Files:**
- Modify: `scripts/llm_orchestrator.py:1496-1534` (the `_validate_generated_strategy` call site)
- Test: `tests/test_llm_orchestrator.py`

#### Step 1: Write the failing test

Add to `tests/test_llm_orchestrator.py`:

```python
def test_build_coder_repair_user_prompt_includes_errors_and_code():
    mod = _load_module()
    thinker_handoff = {"hypothesis": {"hypothesis_id": "h001", "thesis": "test thesis"}}
    previous_code = "def generate_signal(df, params):\n    return df['bad_col'].to_numpy()\n"
    validation_errors = [
        "my_strat: generate_signal failed for tick_610: KeyError: 'bad_col'",
        "my_strat: contract failed for volume_2000: signal contains NaN",
    ]
    prompt = mod._build_coder_repair_user_prompt(
        thinker_handoff=thinker_handoff,
        previous_code=previous_code,
        validation_errors=validation_errors,
        common_columns=["close", "ema_ratio_8", "cvd_price_divergence_6"],
    )
    assert "bad_col" in prompt
    assert "KeyError" in prompt
    assert "NaN" in prompt
    assert "generate_signal" in prompt
    assert "ema_ratio_8" in prompt


def test_build_coder_repair_user_prompt_truncates_long_code():
    mod = _load_module()
    long_code = "x = 1\n" * 1000  # very long
    prompt = mod._build_coder_repair_user_prompt(
        thinker_handoff={},
        previous_code=long_code,
        validation_errors=["error"],
        common_columns=[],
    )
    # Should be truncated
    assert len(prompt) < len(long_code) + 2000
```

#### Step 2: Run tests to verify they fail

```bash
uv run pytest tests/test_llm_orchestrator.py::test_build_coder_repair_user_prompt_includes_errors_and_code tests/test_llm_orchestrator.py::test_build_coder_repair_user_prompt_truncates_long_code -v
```
Expected: `AttributeError: module ... has no attribute '_build_coder_repair_user_prompt'`

#### Step 3: Implement `_build_coder_repair_user_prompt`

In `scripts/llm_orchestrator.py`, add after `_build_coder_user_prompt` (after line ~1022):

```python
_REPAIR_CODE_MAX_CHARS = 4000


def _build_coder_repair_user_prompt(
    *,
    thinker_handoff: dict[str, Any],
    previous_code: str,
    validation_errors: list[str],
    common_columns: list[str],
) -> str:
    code_snippet = previous_code
    if len(code_snippet) > _REPAIR_CODE_MAX_CHARS:
        code_snippet = code_snippet[:_REPAIR_CODE_MAX_CHARS] + "\n... [truncated]"

    errors_blob = "\n".join(f"  - {e}" for e in validation_errors)
    cols_hint = ", ".join(common_columns[:40]) if common_columns else "(see feature knowledge)"

    return (
        "Your previously generated code FAILED validation. Fix ONLY the listed errors.\n\n"
        "ORIGINAL_THINKER_HANDOFF_JSON_BEGIN\n"
        f"{json.dumps(thinker_handoff, indent=2, sort_keys=True, default=str)}\n"
        "ORIGINAL_THINKER_HANDOFF_JSON_END\n\n"
        "PREVIOUS_CODE_BEGIN\n"
        f"{code_snippet}\n"
        "PREVIOUS_CODE_END\n\n"
        f"VALIDATION_ERRORS:\n{errors_blob}\n\n"
        f"AVAILABLE_COLUMNS_HINT (common across bar configs): {cols_hint}\n\n"
        "Return corrected JSON with same schema: strategy_name, bar_configs, params, code.\n"
        "Fix ONLY the validation errors above. Do not change the overall strategy logic."
    )
```

#### Step 4: Run tests to verify they pass

```bash
uv run pytest tests/test_llm_orchestrator.py::test_build_coder_repair_user_prompt_includes_errors_and_code tests/test_llm_orchestrator.py::test_build_coder_repair_user_prompt_truncates_long_code -v
```
Expected: both PASS

#### Step 5: Integrate repair loop into `main()`

In `scripts/llm_orchestrator.py`, in `main()`, find where `max_code_repair_attempts` is read from config (after `semantic_retry_attempts` line ~1245):

```python
# Add after:  semantic_retry_attempts = int(...)
max_code_repair_attempts = int(runtime_cfg.get("max_code_repair_attempts", 2))
```

Then replace the validation/rejection block in `main()` (lines ~1521-1575). The current structure is:

```python
validation_errors = _validate_generated_strategy(...)

if validation_errors:
    if not args.dry_run and is_new_path and module_path.exists():
        module_path.unlink()
    log_experiment({... "event": "generation_rejected" ...})
    print(f"rejected {module_name}: ...")
else:
    # enqueue tasks ...
```

Replace with:

```python
validation_errors = _validate_generated_strategy(
    strategy_name=module_name,
    signals_dir=signals_dir,
    params=params,
    bar_configs=chosen_bars,
    split=split,
    session_filter=session_filter,
    feature_group=feature_group,
    sample_cache=sample_cache,
)

# Inline repair loop: retry coder with injected errors
if validation_errors and not args.dry_run and max_code_repair_attempts > 0:
    common_cols = list(feature_knowledge.get("common_columns", []))
    for repair_attempt in range(max_code_repair_attempts):
        print(
            f"  repair attempt {repair_attempt + 1}/{max_code_repair_attempts} "
            f"for {module_name}: {validation_errors[0]}"
        )
        try:
            repair_user_prompt = _build_coder_repair_user_prompt(
                thinker_handoff=thinker_handoff,
                previous_code=code,
                validation_errors=validation_errors,
                common_columns=common_cols,
            )
            repair_generation = _call_stage_json(
                stage_name="coder_repair",
                schema_hint="keys: strategy_name, bar_configs, params, code",
                client=coder_client,
                system_prompt=_build_coder_system_prompt(),
                user_prompt=repair_user_prompt,
                temperature=0.1,  # lower temp for targeted repair
                max_output_tokens=int(coder_role["max_output_tokens"]),
                max_attempts=2,
                json_repair_attempts=json_repair_attempts,
                stage_backoff_seconds=stage_backoff_seconds,
                quota_backoff_seconds=quota_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
            )
            repaired_normalized, repair_generation = _normalize_with_semantic_retry(
                stage_name="coder_repair",
                stage_result=repair_generation,
                normalize_fn=lambda payload: _normalize_coder_payload(
                    payload,
                    mission_bar_configs=mission_bar_configs,
                    thinker_brief=thinker_brief,
                ),
                client=coder_client,
                system_prompt=_build_coder_system_prompt(),
                base_user_prompt=repair_user_prompt,
                temperature=0.1,
                max_output_tokens=int(coder_role["max_output_tokens"]),
                max_semantic_retries=semantic_retry_attempts,
                max_attempts=2,
                json_repair_attempts=json_repair_attempts,
                stage_backoff_seconds=stage_backoff_seconds,
                quota_backoff_seconds=quota_backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
                schema_hint="keys: strategy_name, bar_configs, params, code",
            )
        except (LLMClientError, ValueError, RuntimeError) as repair_exc:
            print(f"  repair call failed: {type(repair_exc).__name__}: {repair_exc}")
            break

        # Update working variables with repaired output
        code = repaired_normalized["code"]
        params = repaired_normalized["params"]
        chosen_bars = repaired_normalized["bar_configs"]
        code_hash = _sha256_text(code)[:16]

        # Overwrite module file with repaired code
        _atomic_write_text(module_path, code)

        # Re-validate repaired code
        validation_errors = _validate_generated_strategy(
            strategy_name=module_name,
            signals_dir=signals_dir,
            params=params,
            bar_configs=chosen_bars,
            split=split,
            session_filter=session_filter,
            feature_group=feature_group,
            sample_cache=sample_cache,
        )
        if not validation_errors:
            print(f"  repair succeeded on attempt {repair_attempt + 1}")
            break

if validation_errors:
    if not args.dry_run and is_new_path and module_path.exists():
        module_path.unlink()
    log_experiment(
        {
            "run_id": run_id,
            "agent": "llm_orchestrator",
            "event": "generation_rejected",
            # ... rest of existing log fields unchanged ...
        },
        experiments_path=orchestrator_log_path,
        lock_path=orchestrator_log_lock,
    )
    print(f"rejected {module_name}: {validation_errors[0]}")
else:
    # existing enqueue block unchanged
    enqueued_task_ids: list[str] = []
    # ... rest unchanged ...
```

> **Important:** The `log_experiment` call for `generation_rejected` already exists in the code. In this step, you are only *inserting* the repair loop block **between** the `_validate_generated_strategy` call and the existing `if validation_errors:` block. Do not rewrite the log_experiment call — keep it exactly as-is.

#### Step 6: Run all orchestrator tests

```bash
uv run pytest tests/test_llm_orchestrator.py -v
```
Expected: all PASS

#### Step 7: Commit

```bash
git add scripts/llm_orchestrator.py tests/test_llm_orchestrator.py
git commit -m "feat: add inline coder repair loop with injected validation errors"
```

---

### Task 3: Add `max_code_repair_attempts` to agent config

**Files:**
- Modify: `configs/agents/llm_orchestrator.yaml`

#### Step 1: Add the key

In `configs/agents/llm_orchestrator.yaml`, under `runtime:`, add:

```yaml
runtime:
  max_iterations: 20
  max_pending_tasks: 25
  poll_seconds: 10
  max_runtime_hours: 6
  stage_max_attempts: 3
  json_repair_attempts: 1
  semantic_retry_attempts: 1
  stage_backoff_seconds: 3
  quota_backoff_seconds: 20
  max_backoff_seconds: 90
  max_code_repair_attempts: 2   # NEW: retry coder with injected errors on validation failure
```

#### Step 2: Verify config loads correctly

```bash
uv run python -c "
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path('configs/agents/llm_orchestrator.yaml').read_text())
v = cfg['runtime']['max_code_repair_attempts']
print(f'max_code_repair_attempts={v}')
assert v == 2
print('OK')
"
```
Expected: `max_code_repair_attempts=2\nOK`

#### Step 3: Commit

```bash
git add configs/agents/llm_orchestrator.yaml
git commit -m "config: add max_code_repair_attempts=2 to llm_orchestrator agent config"
```

---

### Task 4: Final regression check

#### Step 1: Run full test suite

```bash
uv run pytest -q
```
Expected: all tests pass, no regressions

#### Step 2: Smoke-check the orchestrator parses correctly

```bash
uv run python -c "
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('orch', Path('scripts/llm_orchestrator.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# Check all new functions exist
assert callable(mod._collect_orchestrator_feedback_items)
assert callable(mod._build_merged_feedback_items)
assert callable(mod._build_coder_repair_user_prompt)
print('All new functions present: OK')
"
```
Expected: `All new functions present: OK`

#### Step 3: Commit

No new code changes — if previous commits are clean, this is just a final verification step.

---

## What Changes and Why

| Change | Where | Why |
|---|---|---|
| `_collect_orchestrator_feedback_items` | `llm_orchestrator.py` | Surfaces `generation_rejected` errors (exact failure reasons) into the feedback analyst's context |
| `_build_merged_feedback_items` | `llm_orchestrator.py` | Replaces the either/or fallback with always-merged feedback from all 3 sources |
| Replace `feedback_items = ...` in `main()` | `llm_orchestrator.py:~1348` | Uses `_build_merged_feedback_items` instead of the broken either/or |
| `_build_coder_repair_user_prompt` | `llm_orchestrator.py` | Constructs targeted repair prompt with exact errors + available column hints |
| Repair loop in `main()` | `llm_orchestrator.py:~1521` | Retries coder inline with injected errors before giving up on the whole iteration |
| `max_code_repair_attempts: 2` | `llm_orchestrator.yaml` | Makes repair attempt count configurable; set to 0 to disable |
