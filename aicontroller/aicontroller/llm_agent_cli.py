#!/usr/bin/env python3
"""
LLM-powered Terminal Agent for Flower + Mongo (human-style UI)
--------------------------------------------------------------

- Natural narration (no raw JSON shown).
- Local commands: `status`, `tool list`, `list runs`, `unlock`.
- NEW: Local *run* shortcut — typing:
      run 1 rounds with FedAvg (lr 0.01, local-epochs 2, fraction-train 0.6)
  immediately launches Flower without relying on the model to emit tool JSON.

Keeps:
- Ollama local model (chat -> generate fallback).
- Mongo logging and run-lock.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
import shutil
from pathlib import Path

import requests
import typer
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from rich.console import Console
from rich.markdown import Markdown
from rich.logging import RichHandler
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

# =========================
# Logging
# =========================

def _log_paths():
    base = os.getenv("AIC_AGENT_LOG_DIR", os.path.expanduser("~/.cache/aicontroller"))
    os.makedirs(base, exist_ok=True)
    file_log = os.path.join(base, os.getenv("AIC_AGENT_LOG_FILE", "llm_agent.log"))
    jsonl_log = os.path.join(base, os.getenv("AIC_AGENT_JSONL", "llm_agent_events.jsonl"))
    return file_log, jsonl_log


def init_logging(level: str = "INFO") -> logging.Logger:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("llm_agent")
    logger.setLevel(lvl)
    logger.handlers.clear()

    file_log, _ = _log_paths()

    ch = RichHandler(rich_tracebacks=False, show_time=False, show_path=False)
    ch.setLevel(lvl)
    ch.setFormatter(logging.Formatter("%(message)s"))

    fh = RotatingFileHandler(file_log, maxBytes=5_000_000, backupCount=5)
    fh.setLevel(lvl)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_event(event_type: str, **kwargs):
    _, jsonl_path = _log_paths()
    evt = {"ts": _now_iso(), "type": event_type} | kwargs
    if os.getenv("AIC_AGENT_JSONL_DISABLE", "0") != "1":
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(evt, default=str) + "\n")
        except Exception:
            pass
    if os.getenv("AIC_AGENT_LOG_TO_MONGO", "false").lower() == "true":
        try:
            db = _db()
            db["agent_logs"].insert_one(evt)
        except Exception:
            pass


logger = init_logging(os.getenv("AIC_AGENT_LOG_LEVEL", "INFO"))

# =========================
# Mongo / project helpers
# =========================

def _db():
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    dbn = os.getenv("MONGODB_DB", "flwr_runs")
    cli = MongoClient(uri, serverSelectionTimeoutMS=4000, connectTimeoutMS=4000)
    cli.admin.command("ping")
    return cli[dbn]


def _project_root(start: Optional[str] = None) -> str:
    d = Path(start or os.getcwd()).resolve()
    for parent in [d, *d.parents]:
        if (parent / "pyproject.toml").exists():
            return str(parent)
    return str(d)

# Recognized tool names (internal)
TOOL_NAMES = {
    "list_runs",
    "show_rounds",
    "summarize_run",
    "suggest_next",
    "compare_strategies",
    "run_flower",
}

TOOL_HELP = {
    "list_runs": "List the most recent runs from MongoDB (accepts: limit, strategy).",
    "show_rounds": "Show per-round metrics for a specific run (args: run_id).",
    "summarize_run": "Summarize a run (args: run_id | latest=true).",
    "suggest_next": "Propose a next config based on last run (args: run_id | latest=true).",
    "compare_strategies": "Compare recent runs grouped by strategy (args: strategies[], limit).",
    "run_flower": "Launch a run with --run-config (args: num-server-rounds, local-epochs, fraction-train, lr, strategy, label-mode, num-partitions, data-root, force).",
}

# =========================
# Ollama chat (chat ➜ generate fallback)
# =========================

def _ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        out.append(f"{role.capitalize()}: {content}\n")
    out.append("Assistant:")
    return "".join(out)


def _ollama_chat_try_chat(messages: List[Dict[str, str]], model: str, temperature: float) -> Optional[str]:
    url = f"{_ollama_host()}/api/chat"
    logger.debug("ollama_chat: trying /api/chat @ %s with %d messages", url, len(messages))
    resp = requests.post(
        url,
        json={"model": model, "messages": messages, "stream": False, "options": {"temperature": temperature}},
        timeout=120,
    )
    if resp.status_code in (404, 405):
        logger.info("/api/chat unavailable (%s), falling back to /api/generate", resp.status_code)
        return None
    resp.raise_for_status()
    data = resp.json()
    if "message" in data and "content" in data["message"]:
        out = data["message"]["content"]
    elif "messages" in data and data["messages"]:
        out = data["messages"][-1].get("content", "")
    else:
        out = data.get("content") or ""
    logger.debug("ollama_chat: /api/chat success, chars=%d", len(out))
    return out


def _ollama_chat_generate(messages: List[Dict[str, str]], model: str, temperature: float) -> str:
    url = f"{_ollama_host()}/api/generate"
    prompt = _messages_to_prompt(messages)
    logger.debug("ollama_chat: using /api/generate @ %s (prompt chars=%d)", url, len(prompt))
    resp = requests.post(
        url,
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    out = data.get("response", "")
    logger.debug("ollama_chat: /api/generate success, chars=%d", len(out))
    return out


def _ollama_chat(messages: List[Dict[str, str]], model: str, temperature: float = 0.2) -> str:
    try:
        content = _ollama_chat_try_chat(messages, model, temperature)
        if content is not None:
            return content
        return _ollama_chat_generate(messages, model, temperature)
    except requests.HTTPError as e:
        try:
            err = e.response.json()
            msg = err.get("error") or err
        except Exception:
            msg = str(e)
        logger.error("Ollama HTTP error: %s | details=%s", e, msg)
        _log_event("ollama_http_error", error=str(e), details=msg)
        raise RuntimeError(f"Ollama HTTP error: {e}\nDetails: {msg}\nCheck `ollama pull <model>`")
    except requests.ConnectionError as e:
        logger.error("Cannot connect to Ollama @ %s: %s", _ollama_host(), e)
        _log_event("ollama_connection_error", host=_ollama_host(), error=str(e))
        raise RuntimeError(f"Cannot connect to Ollama at {_ollama_host()}.")
    except Exception as e:
        logger.exception("Ollama call failed")
        _log_event("ollama_call_failed", error=str(e))
        raise RuntimeError(f"Ollama call failed: {e}")

# =========================
# Run lock + helpers
# =========================

def _lockfile_path() -> str:
    return os.path.join(_project_root(), ".flwr_run.lock")


def _acquire_run_lock(metadata: Dict[str, Any]) -> (bool, Dict[str, Any]):
    path = _lockfile_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {"path": path}
        return False, existing
    try:
        payload = {"ts": _now_iso(), "pid": os.getpid()} | metadata
        with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644), "w", encoding="utf-8") as f:
            json.dump(payload, f)
        logger.debug("run lock acquired: %s", payload)
        return True, payload
    except FileExistsError:
        return False, {"path": path, "note": "exists"}


def _active_run_doc() -> Optional[Dict[str, Any]]:
    try:
        db = _db()
        doc = db["runs"].find_one({"status": {"$in": ["running", "in_progress"]}}, sort=[("started_at", -1)])
        return doc
    except Exception:
        return None


def _release_run_lock() -> None:
    path = _lockfile_path()
    try:
        os.remove(path)
        logger.debug("run lock released")
    except FileNotFoundError:
        pass

# =========================
# Tools (internal)
# =========================

def _format_run_config(d: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k, v in d.items():
        if isinstance(v, bool):
            parts.append(f"{k}={'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}={v}")
        elif v is None:
            continue
        else:
            s = str(v).replace('"', '\\"')
            parts.append(f'{k}="{s}"')
    return " ".join(parts)


def tool_run(run_config: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
    cfg = _format_run_config(run_config)
    cmd = ["flwr", "run", ".", "--run-config", cfg]

    flwr_path = shutil.which("flwr")
    if not flwr_path:
        return {"error": "flwr_cli_not_found", "command": " ".join(cmd)}

    cwd = _project_root()

    # Optional active-run gate via Mongo
    if os.getenv("AIC_AGENT_MONGO_LOCK_ENABLE", "true").lower() == "true" and not force:
        active = _active_run_doc()
        if active:
            info = {"_id": str(active.get("_id")), "status": active.get("status"), "started_at": str(active.get("started_at"))}
            logger.warning("tool_run: active run in Mongo, refusing new run: %s", info)
            return {"error": "active_run_in_mongo", "active": info}

    ok, info = _acquire_run_lock({"cwd": cwd, "cmd": " ".join(cmd)})
    if not ok:
        logger.warning("tool_run: run lock present, refusing to start new run: %s", info)
        return {"error": "run_in_progress", "lock": info}

    logger.info("tool_run: executing -> %s", " ".join(shlex.quote(c) for c in cmd))
    _log_event("tool_call", name="run_flower", run_config=run_config, cwd=cwd)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
        result = {
            "command": " ".join(shlex.quote(c) for c in cmd),
            "cwd": cwd,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout.splitlines()[-60:],
            "stderr_tail": proc.stderr.splitlines()[-60:],
        }
        logger.info("tool_run: returncode=%s", proc.returncode)
        _log_event("tool_result", name="run_flower", returncode=proc.returncode)
        return result
    finally:
        _release_run_lock()


def tool_list_runs(limit: int = 20, strategy: Optional[str] = None) -> Dict[str, Any]:
    logger.debug("tool_list_runs: limit=%d strategy=%s", limit, strategy)
    _log_event("tool_call", name="list_runs", limit=limit)
    db = _db()
    q: Dict[str, Any] = {}
    if strategy:
        q["run_config.strategy"] = strategy
    docs = list(db["runs"].find(q).sort("started_at", -1).limit(int(limit)))
    for d in docs:
        d["_id"] = str(d["_id"])
    out = {"runs": docs}
    _log_event("tool_result", name="list_runs", count=len(docs))
    return out


def tool_show_rounds(run_id: str) -> Dict[str, Any]:
    logger.debug("tool_show_rounds: run_id=%s", run_id)
    _log_event("tool_call", name="show_rounds", run_id=run_id)
    db = _db()
    oid = ObjectId(run_id)
    rounds = list(db["rounds"].find({"run_id": oid}).sort([("round", 1), ("phase", 1)]))
    for r in rounds:
        r["_id"], r["run_id"] = str(r["_id"]), str(r["run_id"])
    _log_event("tool_result", name="show_rounds", count=len(rounds))
    return {"rounds": rounds}


def _summarize_rounds(round_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    fits  = [r for r in round_docs if r.get("phase") == "fit"]
    evals = [r for r in round_docs if r.get("phase") == "eval"]
    summary: Dict[str, Any] = {}

    avg_train = []
    for r in fits:
        cl = [cm.get("metrics", {}).get("train_loss") for cm in r.get("client_metrics", [])]
        cl = [x for x in cl if isinstance(x, (int, float))]
        avg_train.append(sum(cl)/len(cl) if cl else None)
    if avg_train:
        summary["avg_train_loss_by_round"] = avg_train

    eval_acc = []
    for r in evals:
        agg = r.get("agg_metrics") or r.get("agg_metrics_server") or {}
        acc = agg.get("eval_acc")
        if acc is None:
            cl = [cm.get("metrics", {}).get("eval_acc") for cm in r.get("client_metrics", [])]
            cl = [x for x in cl if isinstance(x, (int, float))]
            acc = sum(cl)/len(cl) if cl else None
        eval_acc.append(acc)
    if eval_acc:
        summary["eval_acc_by_round"] = eval_acc
        summary["best_eval_acc"] = max([a for a in eval_acc if a is not None], default=None)

    tips: List[str] = []
    if eval_acc and len(eval_acc) >= 3:
        deltas = []
        for i in range(len(eval_acc) - 1):
            a, b = eval_acc[i], eval_acc[i + 1]
            if a is None or b is None:
                continue
            deltas.append(b - a)
        if len(deltas) >= 2 and all(abs(d) < 0.002 for d in deltas[-2:]):
            tips.append("Validation plateau: increase num-server-rounds (+3–5) or set local-epochs=2.")
    last_train = next((x for x in reversed(avg_train or []) if x is not None), None)
    last_acc = next((x for x in reversed(eval_acc or []) if x is not None), None)
    if isinstance(last_train, (int, float)) and isinstance(last_acc, (int, float)):
        if last_train < 0.2 and last_acc < 0.9:
            tips.append("Possible overfit: reduce lr (0.01→0.005) or try strategy=FedAdam/FedYogi.")
        elif last_train > 0.8 and last_acc < 0.7:
            tips.append("Underfitting: raise local-epochs to 2–3 or lr to 0.02.")
    if not tips:
        tips.append("Try fraction-train=0.6–0.8 so more clients participate each round.")
    summary["suggestions"] = tips
    return summary


def tool_summary(run_id: Optional[str] = None, latest: bool = False) -> Dict[str, Any]:
    logger.debug("tool_summary: run_id=%s latest=%s", run_id, latest)
    _log_event("tool_call", name="summarize_run", run_id=run_id, latest=latest)
    db = _db()
    if latest:
        run = list(db["runs"].find({}).sort("started_at", -1).limit(1))
        if not run:
            return {"error": "no_runs"}
        oid = run[0]["_id"]
    else:
        if not run_id:
            return {"error": "missing_run_id"}
        oid = ObjectId(run_id)
    rounds = list(db["rounds"].find({"run_id": oid}).sort([("round", 1), ("phase", 1)]))
    summ = _summarize_rounds(rounds)
    _log_event("tool_result", name="summarize_run", round_count=len(rounds))
    return {"summary": summ, "run_id": str(oid)}


def tool_suggest(run_id: Optional[str] = None, latest: bool = False) -> Dict[str, Any]:
    logger.debug("tool_suggest: run_id=%s latest=%s", run_id, latest)
    _log_event("tool_call", name="suggest_next", run_id=run_id, latest=latest)
    db = _db()
    if latest:
        run = list(db["runs"].find({}).sort("started_at", -1).limit(1))
        if not run:
            return {"error": "no_runs"}
        run_doc = run[0]
    else:
        if not run_id:
            return {"error": "missing_run_id"}
        run_doc = db["runs"].find_one({"_id": ObjectId(run_id)})
        if not run_doc:
            return {"error": "run_not_found"}
    rounds = list(db["rounds"].find({"run_id": run_doc["_id"]}).sort([("round", 1), ("phase", 1)]))
    summ = _summarize_rounds(rounds)
    rc = dict(run_doc.get("run_config", {}))
    strat = rc.get("_chosen_strategy") or rc.get("strategy") or "FedAvg"
    lr = float(rc.get("lr", 0.01))
    local_epochs = int(rc.get("local-epochs", 1))
    num_rounds = int(rc.get("num-server-rounds", 3))
    proposal = {"strategy": strat, "lr": lr, "local-epochs": local_epochs, "num-server-rounds": num_rounds}
    tips = summ.get("suggestions", [])
    for tip in tips:
        low = tip.lower()
        if "plateau" in low:
            proposal["num-server-rounds"] = num_rounds + 3
            break
        if "overfit" in low:
            proposal["lr"] = max(lr * 0.5, 1e-4)
            if strat == "FedAvg":
                proposal["strategy"] = "FedAdam"
            break
        if "underfitting" in low:
            proposal["local-epochs"] = max(local_epochs, 2)
            proposal["lr"] = min(lr * 2.0, 0.05)
            break
    _log_event("tool_result", name="suggest_next", proposal=proposal)
    return {"proposal": proposal, "summary": summ, "source_run_id": str(run_doc["_id"])}

# =========================
# Parsing / normalization
# =========================

_RUNCFG_PATTERNS = {
    "num-server-rounds": r"(?:num[- ]?server[- ]?rounds|rounds?)\s*[:= ]\s*(\d+)",
    "local-epochs": r"(?:local[- ]?epochs?)\s*[:= ]\s*(\d+)",
    "fraction-train": r"(?:fraction[- ]?train)\s*[:= ]\s*([0-9]*\.?[0-9]+)",
    "lr": r"(?:lr|learning[- ]?rate)\s*[:= ]\s*([0-9]*\.?[0-9]+)",
    "strategy": r"(?:strategy|with)\s*(Fed[A-Za-z0-9]+)",
}

def _heuristic_parse_run_config(text: str) -> Dict[str, Any]:
    rc: Dict[str, Any] = {}
    for key, pat in _RUNCFG_PATTERNS.items():
        m = re.search(pat, text, re.IGNORECASE)
        if not m:
            continue
        val = m.group(1)
        if key in ("num-server-rounds", "local-epochs"):
            rc[key] = int(val)
        elif key in ("fraction-train", "lr"):
            rc[key] = float(val)
        elif key == "strategy":
            rc[key] = val
    return rc


def _extract_json(s: str) -> Dict[str, Any] | None:
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


def _normalize_action(js: Dict[str, Any] | None, last_user: str) -> Dict[str, Any] | None:
    """Accept compact actions and JSON stuffed into final.text."""
    if not js:
        return None
    act = js.get("action")

    if act == "final":
        txt = js.get("text", "")
        inner = _extract_json(txt)
        if inner and inner.get("action"):
            js = inner
            act = js.get("action")

    if act in TOOL_NAMES:
        name = act
        args = js.get("args") or {}
        return {"action": "tool", "name": name, "args": args}

    if act == "tool":
        if js.get("name") == "run_flower":
            args = js.get("args") or {}
            rc = args.get("run_config") or {}
            if not rc:
                rc = _heuristic_parse_run_config(last_user)
                if rc:
                    args["run_config"] = rc
                    js["args"] = args
        return js

    return js

# =========================
# NEW: local "run ..." shortcut
# =========================

def _maybe_local_run(user: str) -> bool:
    """If the user typed a 'run ...' command, parse and launch immediately."""
    text = user.strip()
    if not text.lower().startswith("run"):
        return False

    rc = _heuristic_parse_run_config(text)
    # Defaults (can be overridden via env)
    rc.setdefault("strategy", "FedAvg")
    rc.setdefault("num-server-rounds", 1)
    rc.setdefault("local-epochs", 1)
    rc.setdefault("fraction-train", 0.6)
    rc.setdefault("lr", 0.01)
    rc.setdefault("data-root", os.getenv("AIC_DATA_ROOT", "/Users/spartan/Projects/FlowerHackathon/data"))
    rc.setdefault("label-mode", os.getenv("AIC_LABEL_MODE", "binary"))
    rc.setdefault("num-partitions", int(os.getenv("AIC_NUM_PARTITIONS", "5")))

    console.print(Markdown(
        "**Starting Flower run** with config:\n\n" +
        "\n".join(f"- `{k}` = `{v}`" for k, v in rc.items())
    ))
    res = tool_run(rc, force=False)
    code = res.get("returncode")
    if code == 0:
        console.print("[green]Run launched[/green]")
    else:
        console.print("[red]Run failed to launch[/red]")
        tail = "\n".join(res.get("stderr_tail", [])[-20:])
        if tail:
            console.print(Panel(tail, title="stderr tail"))
    return True

# =========================
# Render helpers (human UI)
# =========================

def _print_status_line():
    lf = _lockfile_path()
    active = _active_run_doc()
    if os.path.exists(lf) or active:
        console.print("[yellow]Running...[/yellow]")
    else:
        console.print("[green]Idle[/green]")


def _render_tool_list():
    table = Table(title="Available Tools")
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("What it does")
    for k, v in TOOL_HELP.items():
        table.add_row(k, v)
    console.print(table)


def _render_runs(docs: Sequence[Dict[str, Any]]):
    if not docs:
        console.print("[dim]No runs found[/dim]")
        return
    table = Table(title="Recent Runs")
    table.add_column("Run ID", style="cyan")
    table.add_column("Strategy")
    table.add_column("Status")
    table.add_column("Started")
    table.add_column("Finished")
    for d in docs:
        rc = d.get("run_config", {}) or {}
        table.add_row(
            str(d.get("_id")),
            rc.get("strategy") or d.get("strategy") or "?",
            str(d.get("status")),
            str(d.get("started_at")),
            str(d.get("finished_at") or d.get("ended_at") or ""),
        )
    console.print(table)


def _render_summary(payload: Dict[str, Any]):
    summ = payload.get("summary", {}) or {}
    rid = payload.get("run_id")
    lines = []
    best = summ.get("best_eval_acc")
    if best is not None:
        lines.append(f"Best eval acc: **{best:.4f}**")
    if "avg_train_loss_by_round" in summ:
        last_train = next((x for x in reversed(summ["avg_train_loss_by_round"]) if x is not None), None)
        if last_train is not None:
            lines.append(f"Last train loss: {last_train:.4f}")
    tips = summ.get("suggestions") or []
    if tips:
        lines.append("\n**Suggestions**")
        for t in tips:
            lines.append(f"• {t}")
    console.print(Panel.fit(Markdown(f"### Run {rid}\n" + "\n".join(lines)), title="Summary"))


def _render_rounds(rounds: List[Dict[str, Any]]):
    if not rounds:
        console.print("[dim]No rounds logged[/dim]")
        return
    table = Table(title="Rounds")
    table.add_column("Round", justify="right")
    table.add_column("Phase")
    table.add_column("Eval Acc")
    table.add_column("Train Loss")
    for r in rounds:
        agg = (r.get("agg_metrics") or r.get("agg_metrics_server") or {}) or {}
        acc = agg.get("eval_acc")
        cl = [cm.get("metrics", {}).get("train_loss") for cm in (r.get("client_metrics") or [])]
        cl = [x for x in cl if isinstance(x, (int, float))]
        tl = f"{(sum(cl)/len(cl)):.4f}" if cl else ""
        table.add_row(str(r.get("round")), r.get("phase", ""), "" if acc is None else f"{acc:.4f}", tl)
    console.print(table)


def _render_compare(groups: Dict[str, Any]):
    if not groups:
        console.print("[dim]No runs to compare[/dim]")
        return
    for strat, runs in groups.items():
        table = Table(title=f"Strategy: {strat}")
        table.add_column("Run ID", style="cyan")
        table.add_column("Best Acc")
        table.add_column("Last Acc")
        table.add_column("Last Train Loss")
        table.add_column("Rounds")
        table.add_column("LR")
        for r in runs:
            table.add_row(
                r.get("run_id", ""),
                _fmtf(r.get("best_eval_acc")),
                _fmtf(r.get("last_eval_acc")),
                _fmtf(r.get("last_train_loss")),
                str(r.get("num-server-rounds")),
                _fmtf(r.get("lr")),
            )
        console.print(table)


def _fmtf(x):
    return "" if x is None else (f"{x:.4f}" if isinstance(x, (int, float)) else str(x))


def _brief_for_tool(name: str, result: Dict[str, Any]) -> str:
    """Short textual summary to feed back to the model (not printed)."""
    if name == "list_runs":
        return f"{len(result.get('runs', []))} runs listed."
    if name == "show_rounds":
        return f"{len(result.get('rounds', []))} rounds."
    if name == "summarize_run":
        s = result.get("summary", {})
        return f"summary: best_acc={s.get('best_eval_acc')}"
    if name == "suggest_next":
        p = result.get("proposal", {})
        return f"proposal: {p}"
    if name == "compare_strategies":
        return f"{len(result)} strategy groups."
    if name == "run_flower":
        rc = result.get("returncode")
        return f"run_exit={rc}"
    return "ok"

# =========================
# Chat loop
# =========================

SYSTEM_PROMPT = (
    "You are a pragmatic ML Ops agent for Flower federated experiments. "
    "Use tools when helpful but reply in natural language. If you need a tool, "
    "emit a JSON object like {\"action\":\"tool\",\"name\":\"list_runs\",\"args\":{...}}; "
    "otherwise just answer. After a tool runs you'll receive a short text summary."
)

def _chat_impl(model: str = "llama3.2:3b", log_level: str = "INFO"):
    global logger
    logger = init_logging(log_level)

    try:
        _db()  # quick health check
    except ServerSelectionTimeoutError as e:
        console.print(f"[red]MongoDB not reachable[/red]\n{e}")
        _log_event("startup_error", error=str(e))
        raise typer.Exit(2)

    _log_event("agent_start", model=model, ollama_host=_ollama_host())
    logger.info("Agent started with model=%s (Ollama host=%s)", model, _ollama_host())

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Say 'Ready' plainly."},
    ]
    content = _ollama_chat(messages, model=model)
    console.print(Markdown(content.strip() or "Ready"))

    while True:
        try:
            user = input("agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _log_event("agent_exit", reason="interrupt")
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            _log_event("agent_exit", reason="user_quit")
            break

        # Local shortcuts (no LLM)
        if user.lower() == "status":
            _print_status_line()
            continue
        if user.lower() in {"tool list", "tools"}:
            _render_tool_list()
            continue
        if user.lower() in {"list runs", "list"}:
            _render_runs(tool_list_runs(limit=20)["runs"])
            continue
        if user.lower() == "unlock":
            try:
                os.remove(_lockfile_path())
                console.print("[green]Removed run lock[/green]")
            except FileNotFoundError:
                console.print("[green]No lock present[/green]")
            continue
        # NEW: deterministic local run
        if _maybe_local_run(user):
            _print_status_line()
            continue

        _log_event("user_input", text=user)
        messages.append({"role": "user", "content": user})

        # Inner loop: allow the model to call tools; we show only the final narration.
        while True:
            content = _ollama_chat(messages, model=model)
            js = _normalize_action(_extract_json(content), last_user=user)

            # Natural reply
            if not js or js.get("action") == "final" and js.get("name") is None:
                text = js.get("text") if js else content
                messages.append({"role": "assistant", "content": text})
                console.print(Markdown(text))
                break

            if js.get("action") == "tool":
                name = js.get("name")
                args = js.get("args") or {}
                logger.debug("tool dispatch: %s args=%s", name, args)
                _log_event("tool_dispatch", name=name, args=args)

                # Execute tool and render
                try:
                    if name == "list_runs":
                        result = tool_list_runs(
                            limit=int(args.get("limit", 20)),
                            strategy=args.get("strategy"),
                        )
                        _render_runs(result.get("runs", []))

                    elif name == "show_rounds":
                        result = tool_show_rounds(run_id=args["run_id"])
                        _render_rounds(result.get("rounds", []))

                    elif name == "summarize_run":
                        result = tool_summary(run_id=args.get("run_id"), latest=bool(args.get("latest", False)))
                        if "error" in result:
                            console.print(f"[red]{result['error']}[/red]")
                        else:
                            _render_summary(result)

                    elif name == "suggest_next":
                        result = tool_suggest(run_id=args.get("run_id"), latest=bool(args.get("latest", False)))
                        if "proposal" in result:
                            p = result["proposal"]
                            console.print(Panel.fit(
                                Markdown(
                                    f"**Suggested next run**\n\n"
                                    f"- strategy: `{p.get('strategy')}`\n"
                                    f"- num-server-rounds: `{p.get('num-server-rounds')}`\n"
                                    f"- local-epochs: `{p.get('local-epochs')}`\n"
                                    f"- lr: `{p.get('lr')}`"
                                ), title="Suggest Next"
                            ))
                        else:
                            console.print(f"[red]{result.get('error','no suggestion')}[/red]")

                    elif name == "compare_strategies":
                        result = tool_compare(strategies=args.get("strategies"), limit=int(args.get("limit", 50)))
                        _render_compare(result)

                    elif name == "run_flower":
                        allowed = {"num-server-rounds","local-epochs","fraction-train","lr","strategy","label-mode","num-partitions","data-root"}
                        raw = args.get("run_config")
                        if isinstance(raw, dict) and raw:
                            rc = {k: v for k, v in raw.items() if k in allowed}
                        else:
                            rc = {k: v for k, v in args.items() if k in allowed}
                            if not rc:
                                rc = _heuristic_parse_run_config(user)
                        if not rc:
                            result = {"error": "empty_run_config", "received": args}
                            console.print("[red]Missing run configuration[/red]")
                        else:
                            console.print(Markdown(
                                f"**Starting run** with config:\n\n"
                                + "\n".join([f"- `{k}` = `{v}`" for k, v in rc.items()])
                            ))
                            result = tool_run(rc, force=bool(args.get("force", False)))
                            rc_code = result.get("returncode")
                            if rc_code == 0:
                                console.print("[green]Run launched[/green]")
                            else:
                                console.print("[red]Run failed to launch[/red]")
                                err_tail = "\n".join(result.get("stderr_tail", [])[-10:])
                                if err_tail:
                                    console.print(Panel(err_tail, title="stderr tail"))

                    else:
                        result = {"error": f"unknown_tool:{name}"}
                        console.print(f"[red]Unknown tool: {name}[/red]")

                except Exception as e:
                    logger.exception("tool execution failed: %s", name)
                    result = {"error": str(e)}
                    console.print(f"[red]{e}[/red]")

                # Feed a SHORT summary of the tool result back to the model (not printed)
                brief = _brief_for_tool(name, result)
                messages.append({"role": "user", "content": f"tool_result {name}: {brief}"})

                # Ask the model for a short narration/next step
                content = _ollama_chat(messages, model=model)
                messages.append({"role": "assistant", "content": content})
                console.print(Markdown(content))
                break

            # Unknown JSON — print original
            messages.append({"role": "assistant", "content": content})
            console.print(Markdown(content))
            break

# =========================
# Typer CLI
# =========================

@app.command("status")
def status_cmd():
    _print_status_line()

@app.command("unlock")
def unlock_cmd():
    path = _lockfile_path()
    try:
        os.remove(path)
        console.print(f"[green]Removed lock[/green]: {path}")
    except FileNotFoundError:
        console.print("[green]No lock to remove[/green]")

@app.command("chat")
def chat_cmd(
    model: str = typer.Option("llama3.2:3b", help="Ollama model name"),
    log_level: str = typer.Option("INFO", help="Log level: DEBUG, INFO, WARNING, ERROR"),
):
    _chat_impl(model=model, log_level=log_level)

@app.callback(invoke_without_command=True)
def _default(
    ctx: typer.Context,
    model: str = typer.Option("llama3.2:3b", "--model", help="Ollama model name"),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level: DEBUG, INFO, WARNING, ERROR"),
):
    if ctx.invoked_subcommand is None:
        return _chat_impl(model=model, log_level=log_level)

if __name__ == "__main__":
    app()
