"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

MAX_INVESTIGATION_ATTEMPTS = 3
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_SAMPLE_ITEMS = 10


class ConfigurationError(RuntimeError):
    """Raised when runtime configuration is incomplete."""


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str

    @classmethod
    def from_env(cls, require_api_key: bool = True) -> "Settings":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if require_api_key and not api_key:
            raise ConfigurationError(
                "OPENROUTER_API_KEY is not configured. Copy .env.example to .env, "
                "add your OpenRouter API key, and restart ReconAI."
            )
        return cls(
            openrouter_api_key=api_key,
            openrouter_model=os.getenv(
                "OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
            ).strip(),
            openrouter_base_url=os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            ).strip(),
        )


def friendly_llm_error(error: Exception) -> str:
    """Convert provider failures into a safe, actionable UI message."""
    message = str(error).lower()
    if any(token in message for token in ("401", "403", "auth", "api key")):
        detail = "OpenRouter authentication failed. Check OPENROUTER_API_KEY in .env."
    elif any(token in message for token in ("429", "rate limit", "quota")):
        detail = "OpenRouter rate limit reached. Wait briefly or select another model."
    elif any(token in message for token in ("provider", "unavailable", "503", "404")):
        detail = (
            "The selected OpenRouter model/provider is unavailable. "
            "Change OPENROUTER_MODEL in .env and retry."
        )
    elif any(token in message for token in ("timeout", "connection", "network")):
        detail = "Could not reach OpenRouter. Check the network and retry."
    else:
        detail = "The AI investigation service returned an unexpected error."
    return f"Investigation could not complete: {detail}"
