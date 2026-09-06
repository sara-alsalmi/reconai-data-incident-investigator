from __future__ import annotations

import sys
from types import SimpleNamespace

from src.agents.runtime import CrewAIRuntime
from src.models import InvestigationAttempt


def test_structured_agent_output_retries_once(monkeypatch):
    calls: list[dict] = []
    expected = InvestigationAttempt(
        findings=["Supported finding"],
        hypothesis="A cautious data-level explanation.",
        evidence_ids=["E001"],
    )

    class FakeCrew:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def kickoff(self):
            if len(calls) == 1:
                InvestigationAttempt.model_validate({})
            return SimpleNamespace(pydantic=expected)

    fake_crewai = SimpleNamespace(
        Crew=FakeCrew,
        Process=SimpleNamespace(sequential="sequential"),
        Task=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "crewai", fake_crewai)
    agent = SimpleNamespace(role="Data Investigator Agent")

    result = CrewAIRuntime._run_structured(agent, "Investigate safely.", InvestigationAttempt)

    assert result == expected
    assert len(calls) == 2
    assert all(call["tracing"] is False for call in calls)


def test_structured_agent_output_does_not_retry_connection_errors(monkeypatch):
    calls: list[dict] = []

    class FailingCrew:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def kickoff(self):
            raise ConnectionError("OpenRouter connection failed")

    fake_crewai = SimpleNamespace(
        Crew=FailingCrew,
        Process=SimpleNamespace(sequential="sequential"),
        Task=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "crewai", fake_crewai)
    agent = SimpleNamespace(role="Data Investigator Agent")

    try:
        CrewAIRuntime._run_structured(
            agent, "Investigate safely.", InvestigationAttempt
        )
    except ConnectionError:
        pass
    else:
        raise AssertionError("ConnectionError should be returned to the UI")

    assert len(calls) == 1
