from __future__ import annotations

import pytest

from src.framework.data.loader import ExecutionMode, get_parquet_files, set_execution_mode


def test_research_mode_blocks_test_split():
    set_execution_mode(ExecutionMode.RESEARCH)
    with pytest.raises(PermissionError):
        get_parquet_files("test")


def test_execution_mode_is_immutable_per_run():
    set_execution_mode(ExecutionMode.RESEARCH)
    with pytest.raises(RuntimeError):
        set_execution_mode(ExecutionMode.PROMOTION)


def test_test_split_requires_mode_to_be_set():
    with pytest.raises(RuntimeError):
        get_parquet_files("test")
