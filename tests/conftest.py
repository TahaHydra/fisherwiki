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


@pytest.fixture(autouse=True)
def isolated_work_dir(tmp_path_factory, monkeypatch):
    """Keep test progress out of the live D: status directory.

    Without this, a test that runs a stage writes work/status/prepare.json into
    the real data root, and `tools/status.py` then reports a twelve-image test
    fixture as the state of the corpus.
    """
    monkeypatch.setenv("FISHERWIKI_WORK",
                       str(tmp_path_factory.mktemp("work")))
