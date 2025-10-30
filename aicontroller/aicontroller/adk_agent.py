"""
Google ADK-backed agent utilities for AIController.
Encapsulates google-genai client and JSON action extraction.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from google import genai as google_genai  # type: ignore
except Exception:  # pragma: no cover
    google_genai = None  # type: ignore


def google_client():
    if google_genai is None:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Export it before running the agent.")
    return google_genai.Client(api_key=api_key)


def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        out.append(f"{role.capitalize()}: {content}\n")
    out.append("Assistant:")
    return "".join(out)


def chat(messages: List[Dict[str, str]], model: str, temperature: float = 0.2) -> str:
    client = google_client()
    prompt = messages_to_prompt(messages)
    if not model.startswith("models/"):
        model = f"models/{model}"
    try:
        resp = client.models.generate_content(model=model, contents=prompt, config={"temperature": temperature})
    except Exception:
        # Fallback: resolve a valid Gemini Flash model via listing
        try:
            names = [m.name for m in client.models.list()]
            prefer = next((n for n in names if "gemini-2.0-flash" in n), None) or \
                     next((n for n in names if "gemini-2.5-flash" in n), None) or \
                     next((n for n in names if "gemini" in n), None)
            if not prefer:
                raise RuntimeError("No Gemini models available")
            resp = client.models.generate_content(model=prefer, contents=prompt, config={"temperature": temperature})
        except Exception as e2:
            raise RuntimeError(f"Gemini call failed: {e2}")

    # Try several common fields for text
    text = getattr(resp, "output_text", None) or getattr(resp, "text", None)
    if not text and hasattr(resp, "candidates") and resp.candidates:
        cand = resp.candidates[0]
        content = getattr(cand, "content", None)
        text = getattr(content, "text", None)
    return (text or "").strip()


# JSON tool-call extraction compatible with existing flow
_OID_RE = re.compile(r"^[0-9a-fA-F]{24}$")


def extract_json(s: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    chunk = m.group(0)
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        chunk = chunk.strip().strip("`")
        try:
            return json.loads(chunk)
        except Exception:
            return None


def normalize_action(js: Dict[str, Any] | None, last_user: str) -> Dict[str, Any] | None:
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

    if act in {"show", "display"} and (name in {"rounds", "show_rounds"} or js.get("target") == "rounds"):
        if run_id and isinstance(run_id, str) and _OID_RE.match(run_id):
            return {"action": "tool", "name": "show_rounds", "args": {"run_id": run_id}}

    if act in {"next", "suggest"}:
        return {"action": "tool", "name": "suggest_next", "args": {"run_id": run_id} if run_id else {}}

    if act in {"get_run_summary", "summarize"}:
        if run_id and isinstance(run_id, str) and _OID_RE.match(run_id):
            return {"action": "tool", "name": "summarize_run", "args": {"run_id": run_id}}
        return {"action": "tool", "name": "summarize_run", "args": {"latest": True}}

    if act in {"get_run_details"}:
        if run_id and isinstance(run_id, str) and _OID_RE.match(run_id):
            return {"action": "tool", "name": "show_rounds", "args": {"run_id": run_id}}

    if act in {"list_runs", "show_rounds", "summarize_run", "suggest_next", "compare_strategies", "run_flower"}:
        return {"action": "tool", "name": act, "args": args}

    if act == "tool":
        return js

    return js
