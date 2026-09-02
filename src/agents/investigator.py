"""Data Investigator Agent factory."""

from __future__ import annotations

from typing import Any


def create_investigator_agent(llm: Any, tools: list[Any]) -> Any:
    from crewai import Agent

    return Agent(
        role="Data Investigator Agent",
        goal=(
            "Adaptively investigate the user's data inconsistency by choosing audited "
            "Pandas tools, collecting numerical evidence, and proposing a cautious hypothesis."
        ),
        backstory=(
            "You are the lead enterprise reconciliation analyst. You begin with the user's "
            "question and dataset profiles, rank informative dimensions by business meaning "
            "and cardinality, and build a minimal but complete evidence chain. Every important "
            "number comes from a provided tool. You distinguish the data-level explanation "
            "from the deeper technical mechanism, distinguish association from causation, "
            "and explore a genuinely new angle after verifier rejection."
        ),
        llm=llm,
        tools=tools,
        allow_delegation=False,
        max_iter=16,
        max_retry_limit=2,
        respect_context_window=True,
        verbose=False,
    )
