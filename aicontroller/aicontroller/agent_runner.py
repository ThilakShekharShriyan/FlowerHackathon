from __future__ import annotations

import json
import os
import shlex
import subprocess
from statistics import mean
from typing import Any, Dict, List, Optional

from bson import ObjectId
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("MONGODB_DB", "flwr_runs")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCKFILE = os.path.join(PROJECT_ROOT, ".flwr_run.lock")

app = FastAPI(title="Flower Agent Runner")

def _db():
    cli = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000, connectTimeoutMS=4000)
    cli.admin.command("ping")
    return cli[DB_NAME]

class RunRequest(BaseModel):
    run_config: Dict[str, Any] = {}
    force: bool = False

def _format_run_config(d: Dict[str, Any]) -> str:
    parts = []
    for k, v in d.items():
        if isinstance(v, bool):
            parts.append(f"{k}={'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            parts.append(f"{k}={v}")
        else:
            s = str(v).replace('"', '\\"')
            parts.append(f'{k}="{s}"')
    return " ".join(parts)

def _acquire_lock() -> bool:
    if os.path.exists(LOCKFILE):
        return False
    try:
        with open(LOCKFILE, "w", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "cwd": PROJECT_ROOT}))
        return True
    except Exception:
        return False

def _release_lock():
    try:
        os.remove(LOCKFILE)
    except FileNotFoundError:
        pass

def _summarize_rounds(round_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    fits  = [r for r in round_docs if r.get("phase") == "fit"]
    evals = [r for r in round_docs if r.get("phase") == "eval"]
    summary: Dict[str, Any] = {}

    avg_train = []
    for r in fits:
        cl = [cm.get("metrics", {}).get("train_loss") for cm in r.get("client_metrics", [])]
        cl = [x for x in cl if isinstance(x, (int, float))]
        avg_train.append(mean(cl) if cl else None)
    if avg_train: summary["avg_train_loss_by_round"] = avg_train

    eval_acc = []
    for r in evals:
        agg = r.get("agg_metrics", {})
        acc = agg.get("eval_acc")
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
            if a is None or b is None: continue
            deltas.append(b - a)
        if len(deltas) >= 2 and all(abs(d) < 0.002 for d in deltas[-2:]):
            tips.append("Validation plateau: increase num-server-rounds (+3–5) or set local-epochs=2.")
    last_train = next((x for x in reversed(avg_train or []) if x is not None), None)
    last_acc = next((x for x in reversed(eval_acc or []) if x is not None), None)
    if isinstance(last_train, (int, float)) and isinstance(last_acc, (int, float)):
        if last_train < 0.2 and last_acc < 0.9:
            tips.append("Potential overfit: lower lr (0.01→0.005) or try FedAdam/FedYogi.")
        elif last_train > 0.8 and last_acc < 0.7:
            tips.append("Underfitting: raise local-epochs to 2–3 or lr to 0.02.")
    if not tips:
        tips.append("Try fraction-train=0.6–0.8 so more clients join each round.")
    summary["suggestions"] = tips
    return summary

def _sort_spec():
    return [("finished_at", -1), ("ended_at", -1), ("started_at", -1)]

# --- Endpoints ---

@app.post("/run")
def launch_run(req: RunRequest):
    if not req.force and not _acquire_lock():
        raise HTTPException(status_code=409, detail="run_in_progress (lock present)")

    cfg = _format_run_config(req.run_config)
    cmd = ["flwr", "run", ".", "--run-config", cfg]
    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        return {
            "command": " ".join(shlex.quote(c) for c in cmd),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout.splitlines()[-80:],
            "stderr_tail": proc.stderr.splitlines()[-80:],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _release_lock()

@app.get("/runs")
def list_runs(limit: int = 20):
    db = _db()
    docs = list(db["runs"].find({}).sort(_sort_spec()).limit(int(limit)))
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs

@app.get("/runs/{run_id}/rounds")
def get_rounds(run_id: str):
    db = _db()
    try:
        oid = ObjectId(run_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    rounds = list(db["rounds"].find({"run_id": oid}).sort([("round", 1), ("phase", 1)]))
    for r in rounds:
        r["_id"] = str(r["_id"]); r["run_id"] = str(r["run_id"])
    return rounds

@app.get("/runs/{run_id}/summary")
def run_summary(run_id: str):
    db = _db()
    try:
        oid = ObjectId(run_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid run_id")
    rounds = list(db["rounds"].find({"run_id": oid}).sort([("round", 1), ("phase", 1)]))
    for r in rounds:
        r["_id"] = str(r["_id"]); r["run_id"] = str(r["run_id"])
    return _summarize_rounds(rounds)

@app.get("/compare")
def compare_strategies(strategies: List[str] = Query(default=[]), limit: int = 50):
    db = _db()
    runs = list(db["runs"].find({}).sort(_sort_spec()).limit(int(limit)))
    if strategies:
        sset = set(strategies)
        runs = [r for r in runs if (r.get("run_config", {}).get("_chosen_strategy") or r.get("run_config", {}).get("strategy")) in sset]

    out: Dict[str, Any] = {}
    for r in runs:
        rid = str(r["_id"])
        rc  = r.get("run_config", {})
        strat = rc.get("_chosen_strategy") or rc.get("strategy") or "UNKNOWN"
        rnds = list(db["rounds"].find({"run_id": r["_id"]}).sort([("round", 1), ("phase", 1)]))
        summ = _summarize_rounds(rnds)
        accs = summ.get("eval_acc_by_round", []) or []
        last_acc = next((x for x in reversed(accs) if x is not None), None)
        last_train = next((x for x in reversed(summ.get("avg_train_loss_by_round", []) or []) if x is not None), None)
        grp = out.setdefault(strat, {"runs": []})
        grp["runs"].append({
            "run_id": rid,
            "status": r.get("status"),
            "best_eval_acc": summ.get("best_eval_acc"),
            "last_eval_acc": last_acc,
            "last_train_loss": last_train,
            "num-server-rounds": rc.get("num-server-rounds"),
            "lr": rc.get("lr"),
            "fraction-train": rc.get("fraction-train"),
            "local-epochs": rc.get("local-epochs"),
        })
    return out
