"""Small helpers for explicit investigation state updates."""

from __future__ import annotations

import sys
from datetime import datetime

from src.models import InvestigationState, TraceEvent


def _console_safe(value: str, encoding: str | None = None) -> str:
    """Make model-supplied trace text printable on legacy Windows terminals."""
    target_encoding = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    return value.encode(target_encoding, errors="replace").decode(target_encoding)


def add_trace(state: InvestigationState, category: str, message: str) -> None:
    event = TraceEvent(
        sequence=len(state.trace) + 1, category=category, message=message
    )
    state.trace.append(event)
    timestamp = datetime.now().strftime("%H:%M:%S")
    run_id = state.investigation_id.split("-")[0]
    line = (
        f"[{timestamp}] [ReconAI:{run_id}] [{category.upper()}] "
        f"{event.sequence}. {message}"
    )
    print(_console_safe(line), flush=True)


def render_trace(state: InvestigationState) -> str:
    return "\n".join(f"{event.sequence}. {event.message}" for event in state.trace)
