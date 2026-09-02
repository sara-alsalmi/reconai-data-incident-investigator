"""Evidence Verifier Agent factory."""

from __future__ import annotations

from typing import Any


def create_verifier_agent(llm: Any) -> Any:
    from crewai import Agent

    return Agent(
        role="Evidence Verifier Agent",
        goal=(
            "Critically determine whether tool-generated evidence supports the proposed "
            "hypothesis, rejecting causal overreach and requesting specific missing evidence."
        ),
        backstory=(
            "You are an independent audit reviewer. You test the full evidence chain, reject "
            "constant or irrelevant segment fields, and distinguish a proven data-level cause "
            "from an unproven technical mechanism. Correlation is not causation, high confidence "
            "requires multiple consistent observations, and inconclusive is valid."
        ),
        llm=llm,
        allow_delegation=False,
        max_retry_limit=2,
        respect_context_window=True,
        verbose=False,
    )
