"""
UI render helpers for the AIController agent (Rich-based), parameterized by Console.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown


def render_tool_list(console, tool_help: Dict[str, str]):
    table = Table(title="Available Tools")
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("What it does")
    for k, v in tool_help.items():
        table.add_row(k, v)
    console.print(table)


def render_runs(console, docs: Sequence[Dict[str, Any]]):
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


def render_summary(console, payload: Dict[str, Any]):
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


def render_rounds(console, rounds: List[Dict[str, Any]]):
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
        acc = agg.get("eval_acc") if isinstance(agg, dict) else None
        cl = [cm.get("metrics", {}).get("train_loss") for cm in (r.get("client_metrics") or [])]
        cl = [x for x in cl if isinstance(x, (int, float))]
        tl = f"{(sum(cl)/len(cl)):.4f}" if cl else ""
        table.add_row(str(r.get("round")), r.get("phase", ""), "" if acc is None else f"{acc:.4f}", tl)
    console.print(table)


def render_compare(console, groups: Dict[str, Any]):
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


def print_status_line(console, lockfile_path: str, active_run: Dict[str, Any] | None):
    import os
    if os.path.exists(lockfile_path) or active_run:
        console.print("[yellow]Running...[/yellow]")
    else:
        console.print("[green]Idle[/green]")


def _fmtf(x):
    return "" if x is None else (f"{x:.4f}" if isinstance(x, (int, float)) else str(x))
