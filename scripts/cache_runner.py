#!/usr/bin/env python3
"""Cache rebuild helper for feature matrices.

Examples:
  uv run python scripts/cache_runner.py --split all --session-filter eth --bar-filter tick_610 --clean
  uv run python scripts/cache_runner.py --split train validate --session-filter both --bar-filter tick_610
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.framework.api import ExecutionMode
from src.framework.data.bar_configs import BAR_CONFIGS, bar_config_label
from src.framework.data.loader import get_execution_mode, set_execution_mode
from src.framework.features_canonical.builder import (
    CACHE_DIR,
    _feature_cache_dir,
    build_full_cache,
)

_VALID_SPLITS = ("train", "validate", "test")
_VALID_SESSION_FILTERS = ("eth", "rth", "both")
_VALID_EXEC_MODES = ("auto", "research", "promotion")


def _bar_label(cfg: dict) -> str:
    return bar_config_label(cfg)


def _parse_splits(values: list[str]) -> list[str]:
    raw: list[str] = []
    for value in values:
        raw.extend(part.strip().lower() for part in value.split(",") if part.strip())
    if not raw:
        return list(_VALID_SPLITS)
    if "all" in raw:
        return list(_VALID_SPLITS)
    bad = [s for s in raw if s not in _VALID_SPLITS]
    if bad:
        raise ValueError(f"Invalid split(s): {bad}. Valid: {list(_VALID_SPLITS)} or all")
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for split in raw:
        if split not in seen:
            seen.add(split)
            out.append(split)
    return out


def _expand_sessions(session_filter: str) -> list[str]:
    sf = session_filter.lower()
    if sf not in _VALID_SESSION_FILTERS:
        raise ValueError(f"Invalid session_filter={session_filter}")
    if sf == "both":
        return ["eth", "rth"]
    return [sf]


def _select_mode(mode_arg: str, splits: list[str]) -> ExecutionMode:
    mode_norm = mode_arg.lower()
    if mode_norm not in _VALID_EXEC_MODES:
        raise ValueError(f"Invalid execution_mode={mode_arg}")
    if mode_norm == "research":
        return ExecutionMode.RESEARCH
    if mode_norm == "promotion":
        return ExecutionMode.PROMOTION
    # auto: test split requires promotion mode
    return ExecutionMode.PROMOTION if "test" in splits else ExecutionMode.RESEARCH


def _cache_dirs_to_clean(session_filters: list[str], bar_filter: str | None) -> list[Path]:
    selected_cfgs = list(BAR_CONFIGS)
    if bar_filter:
        selected_cfgs = [cfg for cfg in BAR_CONFIGS if _bar_label(cfg) == bar_filter]
        if not selected_cfgs:
            available = ", ".join(_bar_label(cfg) for cfg in BAR_CONFIGS)
            raise ValueError(f"Unknown bar_filter '{bar_filter}'. Available: {available}")

    dirs: list[Path] = []
    for sf in session_filters:
        for cfg in selected_cfgs:
            dirs.append(_feature_cache_dir(cfg["bar_size"], cfg["bar_type"], cfg["bar_threshold"], sf))
    return dirs


def _clean_cache_dirs(dirs: list[Path]) -> None:
    if not dirs:
        return
    print("Cleaning cache directories:")
    for d in dirs:
        print(f"  - {d}")
        shutil.rmtree(d, ignore_errors=True)
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _parse_args() -> argparse.Namespace:
    available_bars = ", ".join(_bar_label(cfg) for cfg in BAR_CONFIGS)
    parser = argparse.ArgumentParser(description="Build feature cache for selected splits/session/bar configs.")
    parser.add_argument(
        "--split",
        nargs="+",
        default=["all"],
        help="Split(s): train validate test all (supports comma-separated values)",
    )
    parser.add_argument(
        "--session-filter",
        default="eth",
        choices=list(_VALID_SESSION_FILTERS),
        help="Session to build: eth, rth, or both",
    )
    parser.add_argument(
        "--bar-filter",
        default=None,
        help=f"Optional single bar config: {available_bars}",
    )
    parser.add_argument(
        "--file-limit",
        type=int,
        default=None,
        help="Process only first N files per split",
    )
    parser.add_argument(
        "--execution-mode",
        default="auto",
        choices=list(_VALID_EXEC_MODES),
        help="Execution mode: auto (test=>promotion), research, promotion",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete selected cache directories before rebuilding",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    try:
        splits = _parse_splits(args.split)
        session_filters = _expand_sessions(args.session_filter)
        mode = _select_mode(args.execution_mode, splits)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    current = get_execution_mode()
    if current is None:
        set_execution_mode(mode)
    elif current != mode:
        print(f"ERROR: execution mode already set to {current}, requested {mode}")
        return 2

    if args.clean:
        try:
            _clean_cache_dirs(_cache_dirs_to_clean(session_filters, args.bar_filter))
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 2

    total_tasks = 0
    total_completed = 0
    total_errors = 0

    for session_filter in session_filters:
        for split in splits:
            print("\n" + "=" * 80)
            print(
                f"Building split={split} session={session_filter} "
                f"bar_filter={args.bar_filter or 'all'} file_limit={args.file_limit}"
            )
            print("=" * 80)
            stats = build_full_cache(
                split=split,
                session_filter=session_filter,
                file_limit=args.file_limit,
                bar_filter=args.bar_filter,
            )
            total_tasks += int(stats.get("total", 0))
            total_completed += int(stats.get("completed", 0))
            total_errors += int(stats.get("errors", 0))

    print("\n" + "#" * 80)
    print(
        f"All done: completed={total_completed - total_errors}/{total_completed} "
        f"tasks={total_tasks} errors={total_errors}"
    )
    print("#" * 80)
    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
