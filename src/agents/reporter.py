"""Business Report Agent factory."""

from __future__ import annotations

from typing import Any


def create_reporter_agent(llm: Any) -> Any:
    from crewai import Agent

    return Agent(
        role="Business Report Agent",
        goal=(
            "Produce a plain-English, scan-friendly incident report whose claims are traceable "
            "to evidence IDs, whose impact figures come only from deterministic calculations, "
            "and which separates the data-level cause from the underlying technical cause."
        ),
        backstory=(
            "You communicate uncertainty honestly to business stakeholders. You never "
            "fabricate monetary impact and clearly label inconclusive investigations."
        ),
        llm=llm,
        allow_delegation=False,
        max_retry_limit=2,
        respect_context_window=True,
        verbose=False,
    )
