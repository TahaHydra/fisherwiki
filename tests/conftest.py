"""Shared pytest fixtures and path setup."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "tools", REPO / "ml"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402


@pytest.fixture()
def tmp_data_root(tmp_path, monkeypatch):
    """Point the data layer at an isolated temp root for the duration of a test."""
    from fwdata import config

    paths = config.Paths(tmp_path).ensure()
    monkeypatch.setattr(config, "PATHS", paths)
    return paths
