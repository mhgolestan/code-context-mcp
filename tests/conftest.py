"""Shared test setup: keep the suite's tool calls out of the real repositories' memory."""

import pytest

from golestan_code_mcp import memory


@pytest.fixture
def real_recording():
    """Request this to record every run, including those about real repositories (flask, gin)."""


@pytest.fixture(autouse=True)
def _keep_real_repositories_out_of_memory(request, monkeypatch):
    """Many tests call tools against flask and gin. Their runs would be consolidated into those repositories' facts
    (a test's synthetic caller count, 40 timing calls), so they are dropped, and so are their uses, which would boost
    flask results in later searches. Throwaway `t-` repositories still record."""
    if "real_recording" in request.fixturenames:
        return
    record = memory.record

    def real(item):
        return [item["use"]["repo"]] if "use" in item else item["repos"]

    monkeypatch.setattr(memory, "record",
                        lambda item: None if any(not r.startswith("t-") for r in real(item)) else record(item))
