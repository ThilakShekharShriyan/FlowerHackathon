# aicontroller/tracking_strategy.py
"""
Tracking wrapper for Flower Server strategies.

Key fixes:
- Prefer calling hooks with the official signature FIRST:
    aggregate_train(rnd, results, failures)
    aggregate_evaluate(rnd, results, failures)
  Only fall back to legacy orders if a TypeError occurs.
- aggregate_train -> returns (agg_arrays, agg_train_metrics)
- aggregate_evaluate -> returns metrics only (dict/float/None)
- Robust logging: dict or (server_metrics, clientapp_metrics) tuples.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pymongo import MongoClient


def _now():
    return datetime.now(timezone.utc)


def _to_plain(obj: Any) -> Any:
    try:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {str(k): _to_plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_plain(x) for x in obj]
        if is_dataclass(obj):
            return {k: _to_plain(v) for k, v in asdict(obj).items()}
        if hasattr(obj, "to_dict"):
            try:
                return _to_plain(obj.to_dict())  # type: ignore[attr-defined]
            except Exception:
                pass
        if hasattr(obj, "items"):
            try:
                return {str(k): _to_plain(v) for k, v in obj.items()}  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass
    return str(obj)


def _unpack_args(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Optional[int], Any, Any]:
    """Normalize to (rnd, results, failures) from common signatures."""
    rnd = kwargs.get("rnd")
    results = kwargs.get("results")
    failures = kwargs.get("failures")

    a = list(args)

    if len(a) == 3:
        # Either (rnd, results, failures) OR (results, failures, rnd)
        if rnd is None and isinstance(a[0], int):
            rnd, results, failures = a[0], a[1], a[2]
        elif rnd is None and isinstance(a[2], int):
            results, failures, rnd = a[0], a[1], a[2]
        else:
            results = a[0] if results is None else results
            failures = a[1] if failures is None else failures
            rnd = a[2] if rnd is None else rnd

    elif len(a) == 2:
        # (rnd, results) OR (results, rnd) OR (results, failures)
        if isinstance(a[0], int):
            rnd = a[0] if rnd is None else rnd
            results = a[1] if results is None else results
        elif isinstance(a[1], int):
            results = a[0] if results is None else results
            rnd = a[1] if rnd is None else rnd
        else:
            results = a[0] if results is None else results
            failures = a[1] if failures is None else failures

    elif len(a) == 1:
        results = a[0] if results is None else results

    return rnd, results, failures


def _call_with_fallbacks(fn, rnd, results, failures):
    """
    Try official order first; only fall back if a TypeError occurs.
    This avoids silently passing the wrong order to functions that accept any args.
    """
    attempts = [
        (rnd, results, failures),   # ✅ official
        (results, failures, rnd),   # legacy variant
        (rnd, results),             # 2-arg variants seen historically
        (results, rnd),
        (results, failures),
        (results,),                 # 1-arg
        {"kwargs": {"rnd": rnd, "results": results, "failures": failures}},  # kwargs last
    ]
    last_err: Exception | None = None
    for attempt in attempts:
        try:
            if isinstance(attempt, tuple):
                return fn(*attempt)
            else:
                return fn(**attempt["kwargs"])
        except TypeError as e:
            last_err = e
            continue
    if last_err:
        raise last_err
    raise RuntimeError("aggregate_* call failed with all fallbacks")


class TrackingStrategy:
    """Wrap a Flower server strategy and log runs/rounds into MongoDB."""

    def __init__(
        self,
        base_strategy: Any,
        *,
        mongo_uri: str = "mongodb://localhost:27017",
        mongo_db: str = "flwr_runs",
        run_meta: Optional[Dict[str, Any]] = None,
        app_name: str = "aicontroller",
    ) -> None:
        self.base = base_strategy
        self.run_meta = run_meta or {}
        self.app_name = app_name
        self._run_id = None

        # Mongo setup (best effort)
        self.db = None
        try:
            cli = MongoClient(mongo_uri, serverSelectionTimeoutMS=4000, connectTimeoutMS=4000)
            cli.admin.command("ping")
            self.db = cli[mongo_db]
        except Exception:
            self.db = None  # tracking disabled if Mongo unreachable

        # Save original hooks
        self._orig_agg_train = getattr(self.base, "aggregate_train", None)
        self._orig_agg_eval = getattr(self.base, "aggregate_evaluate", None)

    # --------- internal helpers ---------

    def _log_round(self, phase: str, round_id: Optional[int], client_results: Any, agg_metrics: Any):
        if self.db is None or self._run_id is None:
            return

        # Extract client metrics (best-effort)
        cl_metrics: List[Dict[str, Any]] = []
        seq: Sequence[Any] = client_results if isinstance(client_results, (list, tuple)) else []
        for res in seq:
            m = None
            try:
                if isinstance(res, tuple) and len(res) >= 2:
                    m = res[1]
                elif hasattr(res, "content"):
                    m = getattr(res.content, "get", lambda *_: None)("metrics")
                elif hasattr(res, "get"):
                    m = res.get("metrics")  # type: ignore[index]
            except Exception:
                m = None
            cl_metrics.append({"metrics": _to_plain(m)})

        # Metrics might be dict OR (server_metrics, clientapp_metrics) tuple
        metrics_doc: Dict[str, Any] = {}
        if isinstance(agg_metrics, tuple):
            if len(agg_metrics) >= 1:
                metrics_doc["agg_metrics_server"] = _to_plain(agg_metrics[0])
            if len(agg_metrics) >= 2:
                metrics_doc["agg_metrics_clientapp"] = _to_plain(agg_metrics[1])
        else:
            metrics_doc["agg_metrics"] = _to_plain(agg_metrics)

        doc = {
            "run_id": self._run_id,
            "phase": str(phase),
            "round": int(round_id) if round_id is not None else None,
            **metrics_doc,
            "client_metrics": cl_metrics,
            "ts": _now(),
        }
        try:
            self.db.rounds.insert_one(doc)  # type: ignore[union-attr]
        except Exception:
            pass

    # --------- wrapped hooks ---------

    def aggregate_train(self, *args, **kwargs):
        rnd, results, failures = _unpack_args(args, kwargs)
        ret = _call_with_fallbacks(self._orig_agg_train, rnd, results, failures)  # type: ignore[misc]

        # Normalize return to (arrays, metrics)
        if isinstance(ret, tuple):
            if len(ret) >= 2:
                agg_arrays, agg_train_metrics = ret[0], ret[1]
            elif len(ret) == 1:
                agg_arrays, agg_train_metrics = ret[0], None
            else:
                agg_arrays, agg_train_metrics = None, None
        else:
            agg_arrays, agg_train_metrics = ret, None

        self._log_round("fit", rnd, results, agg_train_metrics)
        return agg_arrays, agg_train_metrics

    def aggregate_evaluate(self, *args, **kwargs):
        if self._orig_agg_eval is None:
            return None  # some strategies don't implement evaluate

        rnd, results, failures = _unpack_args(args, kwargs)
        ret = _call_with_fallbacks(self._orig_agg_eval, rnd, results, failures)  # type: ignore[misc]

        # Normalize to metrics only (your Flower version expects a single return)
        if isinstance(ret, tuple):
            agg_eval_metrics = ret[1] if len(ret) >= 2 else ret[0]
        else:
            agg_eval_metrics = ret  # dict/float/None

        self._log_round("eval", rnd, results, agg_eval_metrics)
        return agg_eval_metrics

    # --------- public API ---------

    def start(self, *args, **kwargs):
        # Attach wrappers
        if self._orig_agg_train is not None:
            setattr(self.base, "aggregate_train", self.aggregate_train)
        if self._orig_agg_eval is not None:
            setattr(self.base, "aggregate_evaluate", self.aggregate_evaluate)

        # Create run doc
        if self.db is not None:
            try:
                run_doc = {
                    "app": self.app_name,
                    "started_at": _now(),
                    "status": "running",
                    "strategy": type(self.base).__name__,
                    "run_config": _to_plain(self.run_meta),
                    "final_arrays_saved": False,
                }
                self._run_id = self.db.runs.insert_one(run_doc).inserted_id  # type: ignore[union-attr]
            except Exception:
                self._run_id = None

        # Execute
        err = None
        result = None
        try:
            result = self.base.start(*args, **kwargs)
        except Exception as e:
            err = e

        # Finalize
        if self.db is not None and self._run_id is not None:
            try:
                done = _now()
                upd = {
                    "status": "failed" if err else "completed",
                    "finished_at": done,
                    "ended_at": done,
                }
                self.db.runs.update_one({"_id": self._run_id}, {"$set": upd})  # type: ignore[union-attr]
            except Exception:
                pass

        if err:
            raise err
        return result