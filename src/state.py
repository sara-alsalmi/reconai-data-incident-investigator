"""Small helpers for explicit investigation state updates."""

from __future__ import annotations

from datetime import datetime

from src.models import InvestigationState, TraceEvent


def add_trace(state: InvestigationState, category: str, message: str) -> None:
    event = TraceEvent(
        sequence=len(state.trace) + 1, category=category, message=message
    )
    state.trace.append(event)
    timestamp = datetime.now().strftime("%H:%M:%S")
    run_id = state.investigation_id.split("-")[0]
    print(
        f"[{timestamp}] [ReconAI:{run_id}] [{category.upper()}] "
        f"{event.sequence}. {message}",
        flush=True,
    )


def render_trace(state: InvestigationState) -> str:
    return "\n".join(f"{event.sequence}. {event.message}" for event in state.trace)
