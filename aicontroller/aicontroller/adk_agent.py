"""Google ADK-backed utilities for AIController.

This module wraps the (optional) google-genai client, provides a
robust chat() helper with retry and fallback model resolution, and
offers helpers to extract and normalize JSON action payloads returned
by an LLM. The functions here are intentionally small and testable.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

try:
    from google import genai as google_genai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    google_genai = None  # type: ignore

LOGGER = logging.getLogger(__name__)

# Patterns and defaults
_OID_RE = re.compile(r"^[0-9a-fA-F]{24}$")
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)
DEFAULT_TEMPERATURE = 0.2
DEFAULT_RETRIES = 3
RETRY_BACKOFF_BASE = 0.5


class GoogleGenAIUnavailable(RuntimeError):
    """Raised when google-genai is not available or misconfigured."""


def google_client() -> Any:
    """Return an authenticated google-genai Client.

    Raises:
        GoogleGenAIUnavailable: when the library isn't installed or the
            API key is missing.
    """
    if google_genai is None:
        raise GoogleGenAIUnavailable("google-genai is not installed. Install with: pip install google-genai")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise GoogleGenAIUnavailable("GOOGLE_API_KEY is not set. Export it before running the agent.")
    return google_genai.Client(api_key=api_key)


def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Format a message list into a single prompt string for the model.

    Keeps the simple Role: Content lines used across the project.
    """
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "")
        parts.append(f"{role}: {content}\n")
    parts.append("Assistant:")
    return "".join(parts)


def _extract_text_from_response(resp: Any) -> Optional[str]:
    """Try common fields on the google-genai response to extract text.

    The ADK response shapes have changed across versions; we probe a
    few fields conservatively.
    """
    text = getattr(resp, "output_text", None) or getattr(resp, "text", None)
    if text:
        return text
    candidates = getattr(resp, "candidates", None)
    if candidates:
        first = candidates[0]
        content = getattr(first, "content", None)
        if content is not None:
            return getattr(content, "text", None) or getattr(content, "output_text", None)
    return None


def _resolve_preferred_model(client: Any) -> Optional[str]:
    """Return a preferred Gemini model name by listing available models.

    Tries to prefer a flash-family model, falling back to any Gemini.
    """
    try:
        names = [m.name for m in client.models.list()]
    except Exception as exc:  # pragma: no cover - depends on remote service
        LOGGER.debug("Failed to list models: %s", exc)
        return None
    for prefer in ("gemini-2.0-flash", "gemini-2.5-flash"):
        found = next((n for n in names if prefer in n), None)
        if found:
            return found
    # fallback to any gemini mention
    return next((n for n in names if "gemini" in n), None)


def chat(messages: List[Dict[str, str]], model: str, temperature: float = DEFAULT_TEMPERATURE, *, retries: int = DEFAULT_RETRIES) -> str:
    """Send messages to Gemini (via google-genai) and return assistant text.

    Behavior:
    - Validates the client and API key.
    - Attempts `generate_content` with the requested model.
    - On failure, will attempt to resolve a preferred Gemini model and retry.
    - Retries transient errors with exponential backoff.

    Raises:
        GoogleGenAIUnavailable: if the client cannot be constructed.
        RuntimeError: if the model call fails after retries.
    """
    client = google_client()
    prompt = messages_to_prompt(messages)

    # Normalize a common user-provided model name to the ADK format
    target_model = model if model.startswith("models/") else f"models/{model}"

    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.models.generate_content(model=target_model, contents=prompt, config={"temperature": float(temperature)})
            text = _extract_text_from_response(resp)
            return (text or "").strip()
        except Exception as exc:
            LOGGER.debug("generate_content attempt %d failed for model=%s: %s", attempt, target_model, exc)
            last_exc = exc
            # Try resolving a better model name on first failure
            if attempt == 1:
                prefer = _resolve_preferred_model(client)
                if prefer:
                    LOGGER.info("Falling back to preferred model: %s", prefer)
                    target_model = prefer
                    # continue to retry with resolved model
                    time.sleep(RETRY_BACKOFF_BASE)
                    continue
            # backoff then retry
            if attempt < retries:
                time.sleep(RETRY_BACKOFF_BASE * (2 ** (attempt - 1)))
                continue
            # final failure -> surface a clear error
            raise RuntimeError(f"Gemini model call failed after {retries} attempts: {last_exc}") from last_exc


def extract_json(s: str) -> Optional[Dict[str, Any]]:
    """Extract the first (largest) JSON object from a string, if any.

    Returns a parsed dict or None when parsing fails. This is intentionally
    tolerant: it will try a basic strip of surrounding backticks if needed.
    """
    if not s:
        return None
    # find all brace matches and prefer the longest match (heuristic)
    matches = list(_BRACE_RE.finditer(s))
    if not matches:
        return None
    # choose the longest candidate (often the top-level JSON)
    candidate = max((m.group(0) for m in matches), key=len)
    for attempt in (candidate, candidate.strip().strip("`")):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    return None


def normalize_action(js: Dict[str, Any] | None, last_user: str) -> Dict[str, Any] | None:
    """Normalize a JSON action payload emitted by the LLM into our tool shape.

    Accepts several synonyms (show/display -> show_rounds, next/suggest -> suggest_next,
    and summary shortcuts). Also unwraps nested JSON when the assistant emits a
    `final` wrapper with JSON in `text`.
    """
    if not js:
        return None
    act = js.get("action")

    if act == "final":
        txt = js.get("text", "")
        inner = extract_json(txt)
        if inner and inner.get("action"):
            js = inner
            act = js.get("action")

    name = js.get("name")
    args = js.get("args") or {}
    run_id = js.get("run_id") or args.get("run_id") or args.get("id") or args.get("round_id")

    # map short forms to our tool names
    if act in {"show", "display"} and (name in {"rounds", "show_rounds"} or js.get("target") == "rounds"):
        if isinstance(run_id, str) and _OID_RE.match(run_id):
            return {"action": "tool", "name": "show_rounds", "args": {"run_id": run_id}}

    if act in {"next", "suggest"}:
        return {"action": "tool", "name": "suggest_next", "args": ({"run_id": run_id} if run_id else {})}

    if act in {"get_run_summary", "summarize"}:
        if isinstance(run_id, str) and _OID_RE.match(run_id):
            return {"action": "tool", "name": "summarize_run", "args": {"run_id": run_id}}
        return {"action": "tool", "name": "summarize_run", "args": {"latest": True}}

    if act in {"get_run_details"}:
        if isinstance(run_id, str) and _OID_RE.match(run_id):
            return {"action": "tool", "name": "show_rounds", "args": {"run_id": run_id}}

    # direct mapping for common tool names
    if act in {"list_runs", "show_rounds", "summarize_run", "suggest_next", "compare_strategies", "run_flower"}:
        return {"action": "tool", "name": act, "args": args}

    # already tool-shaped
    if act == "tool":
        return js

    return js


__all__ = ["google_client", "messages_to_prompt", "chat", "extract_json", "normalize_action"]
