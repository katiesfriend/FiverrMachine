"""
engines/ai_model.py

Thin wrapper for OpenAI-style chat calls.

IMPORTANT:
- This module must import cleanly even when the `openai` package is not
  installed (e.g., in CI).
- We only error at *call time* if OpenAI is unavailable or misconfigured.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    # `openai` is optional so tests/CI can run without it installed.
    from openai import OpenAI  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - CI without openai
    OpenAI = None  # type: ignore[assignment]


OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MODEL = os.getenv("FM_OPENAI_MODEL", "gpt-4.1-mini")


class OpenAIUnavailable(RuntimeError):
    """Raised when code tries to use OpenAI but it's not installed or configured."""


def get_openai_client() -> "OpenAI":
    """
    Lazily construct an OpenAI client.

    This is only called when we actually need to talk to the API, so that
    importing this module is safe in environments where `openai` is missing.
    """
    if OpenAI is None:
        raise OpenAIUnavailable(
            "The `openai` package is not installed. "
            "Install it into your venv (e.g. `pip install openai`) "
            "to use engines.ai_model."
        )

    api_key = os.getenv(OPENAI_API_KEY_ENV)
    if not api_key:
        raise OpenAIUnavailable(
            f"{OPENAI_API_KEY_ENV} is not set; cannot create OpenAI client."
        )

    return OpenAI(api_key=api_key)


def call_openai_chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 800,
) -> str:
    """
    Generic chat completion wrapper.

    `messages` is the standard OpenAI messages list:
    [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        ...
    ]
    """
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    content = response.choices[0].message.content
    return content or ""
