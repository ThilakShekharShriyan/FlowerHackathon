"""
Shared tools and helpers for the Flower AIController agent.
Moved out of llm_agent_cli to enable reuse with Google ADK-based agents.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from bson import ObjectId
from pymongo import MongoClient

# ===============
# Core helpers
# ===============

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_root(start: Optional[str] = None) -> str:
    d = Path(start or os.getcwd()).resolve()
    # 1) Search upwards from CWD
    for parent in [d, *d.parents]:
        if (parent / "pyproject.toml").exists():
            return str(parent)
    # 2) Common layout: repo root with app under aicontroller/
    ai = d / "aicontroller"
    if (ai / "pyproject.toml").exists():
        return str(ai)
    # 3) Fallback to CWD
    return str(d)


def _db():
    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    dbn = os.getenv("MONGODB_DB", "flwr_runs")
    cli = MongoClient(uri, serverSelectionTimeoutMS=4000, connectTimeoutMS=4000)
    cli.admin.command("ping")
    return cli[dbn]

# ===============
# Run lock
# ===============

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
        payload = {"ts": _now_iso(), **metadata}
        with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644), "w", encoding="utf-8") as f:
            json.dump(payload, f)
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
    except FileNotFoundError:
        pass

# ===============
# Config parsing/formatting
# ===============

_RUNCFG_PATTERNS = {
    "num-server-rounds": r"(?:num[- ]?server[- ]?rounds|rounds?)\s*[:= ]\s*(\d+)",
    "local-epochs": r"(?:local[- ]?epochs?)\s*[:= ]\s*(\d+)",
    "fraction-train": r"(?:fraction[- ]?train)\s*[:= ]\s*([0-9]*\.?[0-9]+)",
    "lr": r"(?:lr|learning[- ]?rate)\s*[:= ]\s*([0-9]*\.?[0-9]+)",
    "strategy": r"(?:strategy|with)\s*(Fed[A-Za-z0-9]+)",
}

def _heuristic_parse_run_config(text: str) -> Dict[str, Any]:
    import re
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

# ===============
# Summaries
# ===============

def _summarize_rounds(round_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    from statistics import mean
    fits  = [r for r in round_docs if r.get("phase") == "fit"]
    evals = [r for r in round_docs if r.get("phase") == "eval"]
    summary: Dict[str, Any] = {}

    avg_train = []
    for r in fits:
        cl = [cm.get("metrics", {}).get("train_loss") for cm in r.get("client_metrics", [])]
        cl = [x for x in cl if isinstance(x, (int, float))]
        avg_train.append(mean(cl) if cl else None)
    if avg_train:
        summary["avg_train_loss_by_round"] = avg_train

    eval_acc = []
    for r in evals:
        agg = (r.get("agg_metrics") or r.get("agg_metrics_server") or {}) or {}
        acc = agg.get("eval_acc") if isinstance(agg, dict) else None
        if acc is None:
            cl = [cm.get("metrics", {}).get("eval_acc") for cm in r.get("client_metrics", [])]
            cl = [x for x in cl if isinstance(x, (int, float))]
            acc = mean(cl) if cl else None
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

# ===============
# Tool functions (business logic)
# ===============

def tool_run(run_config: Dict[str, Any], *, force: bool = False) -> Dict[str, Any]:
    cfg = _format_run_config(run_config)
    fed = os.getenv("AIC_FEDERATION") or os.getenv("FLWR_FEDERATION")
    cmd = ["flwr", "run", "."]
    if fed:
        cmd.append(fed)
    cmd += ["--run-config", cfg]

    if not shutil.which("flwr"):
        return {"error": "flwr_cli_not_found", "command": " ".join(cmd)}

    cwd = _project_root()

    if os.getenv("AIC_AGENT_MONGO_LOCK_ENABLE", "true").lower() == "true" and not force:
        active = _active_run_doc()
        if active:
            info = {"_id": str(active.get("_id")), "status": active.get("status"), "started_at": str(active.get("started_at"))}
            return {"error": "active_run_in_mongo", "active": info}

    ok, info = _acquire_run_lock({"cwd": cwd, "cmd": " ".join(cmd)})
    if not ok:
        return {"error": "run_in_progress", "lock": info}

    try:
        import subprocess
        # Do not stream logs into the agent console; run quietly and return.
        proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
        return {
            "command": " ".join(shlex.quote(c) for c in cmd),
            "cwd": cwd,
            "returncode": proc.returncode,
        }
    finally:
        _release_run_lock()


def tool_list_runs(limit: int = 20, strategy: Optional[str] = None) -> Dict[str, Any]:
    db = _db()
    q: Dict[str, Any] = {}
    if strategy:
        q["run_config.strategy"] = strategy
    docs = list(db["runs"].find(q).sort("started_at", -1).limit(int(limit)))
    for d in docs:
        d["_id"] = str(d["_id"])
    return {"runs": docs}


def tool_show_rounds(run_id: str) -> Dict[str, Any]:
    db = _db()
    oid = ObjectId(run_id)
    rounds = list(db["rounds"].find({"run_id": oid}).sort([("round", 1), ("phase", 1)]))
    for r in rounds:
        r["_id"], r["run_id"] = str(r["_id"]), str(r["run_id"])
    return {"rounds": rounds}


def tool_summary(run_id: Optional[str] = None, latest: bool = False) -> Dict[str, Any]:
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
    return {"summary": summ, "run_id": str(oid)}


def tool_suggest(run_id: Optional[str] = None, latest: bool = False) -> Dict[str, Any]:
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
    return {"proposal": proposal, "summary": summ, "source_run_id": str(run_doc["_id"])}


def tool_compare(strategies: Optional[List[str]] = None, limit: int = 50) -> Dict[str, Any]:
    db = _db()
    q: Dict[str, Any] = {}
    if strategies:
        q["run_config.strategy"] = {"$in": strategies}
    runs = list(db["runs"].find(q).sort("started_at", -1).limit(int(limit)))
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        rid = run["_id"]
        rc = run.get("run_config") or {}
        strat = rc.get("strategy") or run.get("strategy") or "?"
        rounds = list(db["rounds"].find({"run_id": rid}).sort([("round", 1), ("phase", 1)]))
        summ = _summarize_rounds(rounds)
        eval_accs = summ.get("eval_acc_by_round", [])
        last_eval = next((x for x in reversed(eval_accs or []) if isinstance(x, (int, float))), None)
        train_losses = summ.get("avg_train_loss_by_round", [])
        last_train = next((x for x in reversed(train_losses or []) if isinstance(x, (int, float))), None)
        item = {
            "run_id": str(rid),
            "best_eval_acc": summ.get("best_eval_acc"),
            "last_eval_acc": last_eval,
            "last_train_loss": last_train,
            "num-server-rounds": rc.get("num-server-rounds"),
            "lr": rc.get("lr"),
        }
        groups.setdefault(strat, []).append(item)
    return groups
