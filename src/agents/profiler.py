"""Data Profiler Agent factory."""

from __future__ import annotations

from typing import Any


def create_profiler_agent(llm: Any) -> Any:
    from crewai import Agent

    return Agent(
        role="Data Profiler Agent",
        goal=(
            "Understand uploaded datasets from deterministic compact profiles and identify "
            "only relationship candidates supported by column metadata."
        ),
        backstory=(
            "You are a cautious enterprise data architect. You never invent columns, "
            "relationships, row counts, or statistics."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

