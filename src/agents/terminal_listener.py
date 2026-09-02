"""Safe, live CrewAI event tracing for the PowerShell terminal.

The listener deliberately omits prompts, model responses, and chain-of-thought. It
shows operational events only: tasks, LLM request lifecycle, and bounded tool calls.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from typing import Any


_PRINT_LOCK = threading.Lock()
_LISTENER: Any | None = None


def _redact(value: Any, limit: int = 500) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = re.sub(r"sk-or-v1-[A-Za-z0-9_-]+", "[REDACTED_API_KEY]", text)
    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,}\]]+",
        r"\1[REDACTED]",
        text,
    )
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else f"{compact[:limit]}..."


def _emit(kind: str, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    with _PRINT_LOCK:
        print(f"[{timestamp}] [CrewAI] [{kind}] {message}", flush=True)


def ensure_terminal_listener() -> None:
    """Register one process-wide CrewAI event listener."""
    global _LISTENER
    if _LISTENER is not None:
        return

    from crewai.events import (
        BaseEventListener,
        LLMCallCompletedEvent,
        LLMCallFailedEvent,
        LLMCallStartedEvent,
        TaskCompletedEvent,
        TaskFailedEvent,
        TaskStartedEvent,
        ToolUsageErrorEvent,
        ToolUsageFinishedEvent,
        ToolUsageStartedEvent,
    )

    class ReconAITerminalListener(BaseEventListener):
        def setup_listeners(self, event_bus: Any) -> None:
            @event_bus.on(TaskStartedEvent)
            def on_task_started(_: Any, event: Any) -> None:
                task = getattr(event, "task", None)
                agent = getattr(task, "agent", None)
                role = getattr(agent, "role", None) or "CrewAI agent"
                _emit("TASK", f"{role} started")

            @event_bus.on(TaskCompletedEvent)
            def on_task_completed(_: Any, event: Any) -> None:
                task = getattr(event, "task", None)
                agent = getattr(task, "agent", None)
                role = getattr(agent, "role", None) or "CrewAI agent"
                _emit("TASK", f"{role} completed")

            @event_bus.on(TaskFailedEvent)
            def on_task_failed(_: Any, event: Any) -> None:
                _emit("ERROR", f"Task failed: {_redact(event.error)}")

            @event_bus.on(LLMCallStartedEvent)
            def on_llm_started(_: Any, event: Any) -> None:
                role = getattr(event, "agent_role", None) or "agent"
                model = getattr(event, "model", None) or "configured model"
                _emit("LLM", f"Request started for {role} using {model}")

            @event_bus.on(LLMCallCompletedEvent)
            def on_llm_completed(_: Any, event: Any) -> None:
                role = getattr(event, "agent_role", None) or "agent"
                usage = getattr(event, "usage", None) or {}
                tokens = usage.get("total_tokens")
                suffix = f" ({tokens} tokens)" if tokens is not None else ""
                _emit("LLM", f"Response received for {role}{suffix}")

            @event_bus.on(LLMCallFailedEvent)
            def on_llm_failed(_: Any, event: Any) -> None:
                _emit("ERROR", f"LLM request failed: {_redact(event.error)}")

            @event_bus.on(ToolUsageStartedEvent)
            def on_tool_started(_: Any, event: Any) -> None:
                _emit(
                    "TOOL",
                    f"{event.tool_name} called with {_redact(event.tool_args)}",
                )

            @event_bus.on(ToolUsageFinishedEvent)
            def on_tool_finished(_: Any, event: Any) -> None:
                _emit("TOOL", f"{event.tool_name} completed")

            @event_bus.on(ToolUsageErrorEvent)
            def on_tool_error(_: Any, event: Any) -> None:
                _emit(
                    "ERROR",
                    f"{event.tool_name} failed: {_redact(event.error)}",
                )

    _LISTENER = ReconAITerminalListener()
    _emit("READY", "Live operational trace enabled (prompts and hidden reasoning omitted)")

