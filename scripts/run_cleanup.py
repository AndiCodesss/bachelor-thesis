#!/usr/bin/env python3
"""Clean up generated signal files and all run artifacts for a fresh start.

Usage:
    uv run python scripts/run_cleanup.py          # dry-run (shows what would be deleted)
    uv run python scripts/run_cleanup.py --force   # actually deletes
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Signal files to KEEP (never touch these)
SIGNAL_KEEP = {"__init__.py", "example_ema_turn.py"}

# State files to reset (written fresh on next run)
STATE_RESET = [
    "research/.state/experiment_queue.json",
    "research/.state/experiment_queue.lock",
    "research/.state/handoffs.json",
    "research/.state/handoffs.lock",
    "research/.state/llm_orchestrator.json",
    "research/.state/mission_budget.json",
]

# Log files to delete
LOG_FILES = [
    "results/logs/llm_orchestrator.jsonl",
    "results/logs/llm_orchestrator.lock",
    "results/logs/llm_orchestrator.out",
    "results/logs/research_experiments.jsonl",
    "results/logs/research_experiments.lock",
    "results/logs/research_worker.out",
]

# Directories to delete entirely
RUN_DIRS = [ROOT / "results" / "runs"]


def collect_signal_files() -> list[Path]:
    signals_dir = ROOT / "research" / "signals"
    return [
        p for p in sorted(signals_dir.glob("*.py"))
        if p.name not in SIGNAL_KEEP
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean up run artifacts for a fresh start.")
    parser.add_argument("--force", action="store_true", help="Actually delete (default: dry-run).")
    args = parser.parse_args()
    dry = not args.force

    if dry:
        print("DRY RUN — pass --force to actually delete\n")

    total = 0

    # Generated signal modules
    signal_files = collect_signal_files()
    if signal_files:
        print(f"Signal files ({len(signal_files)}):")
        for p in signal_files:
            print(f"  {p.relative_to(ROOT)}")
            total += 1
            if not dry:
                p.unlink()
    else:
        print("Signal files: none to remove")

    # State files
    print(f"\nState files:")
    for rel in STATE_RESET:
        p = ROOT / rel
        if p.exists():
            print(f"  {rel}")
            total += 1
            if not dry:
                p.unlink()

    # Log files
    print(f"\nLog files:")
    for rel in LOG_FILES:
        p = ROOT / rel
        if p.exists():
            print(f"  {rel}")
            total += 1
            if not dry:
                p.unlink()

    # Run directories
    print(f"\nRun directories:")
    for run_root in RUN_DIRS:
        if run_root.exists():
            subdirs = sorted(run_root.iterdir())
            if subdirs:
                for d in subdirs:
                    print(f"  {d.relative_to(ROOT)}/")
                    total += 1
                    if not dry:
                        shutil.rmtree(d)
            else:
                print(f"  {run_root.relative_to(ROOT)}/ (empty)")
        else:
            print(f"  {run_root.relative_to(ROOT)}/ (missing)")

    print(f"\n{'Would remove' if dry else 'Removed'} {total} item(s).")
    if dry and total > 0:
        print("Run with --force to execute.")


if __name__ == "__main__":
    main()
