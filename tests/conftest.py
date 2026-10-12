"""Shared test setup: keep the suite's tool calls out of the real repositories' memory."""

import pytest

from golestan_code_mcp import memory


@pytest.fixture
def real_recording():
    """Request this to record every run, including those about real repositories (flask, gin)."""


@pytest.fixture(autouse=True)
def _keep_real_repositories_out_of_memory(request, monkeypatch):
    """Many tests call tools against flask and gin. Their runs would be consolidated into those repositories' facts
    (a test's synthetic caller count, 40 timing calls), so they are dropped. Throwaway `t-` repositories still record."""
    if "real_recording" in request.fixturenames:
        return
    record = memory.record
    monkeypatch.setattr(memory, "record",
                        lambda run: None if any(not r.startswith("t-") for r in run["repos"]) else record(run))
