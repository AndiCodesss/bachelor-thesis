from __future__ import annotations

import pytest

from src.framework.data.loader import _reset_execution_mode


@pytest.fixture(autouse=True)
def _reset_loader_mode():
    _reset_execution_mode()
    yield
    _reset_execution_mode()
