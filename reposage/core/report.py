"""Reporting utilities for reposage (stub)."""
from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any, Dict, List

from rich.console import Console

from . import scan as scan_module


console = Console()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_run(target_path: str | Path, label: str = "after") -> Dict[str, Any]:  # noqa: D401
    """Run pytest and capture pass/fail stats."""

    res = scan_module.run_pytest(target_path)
    artifact_path = Path(target_path) / ".reposage" / "artifacts" / f"test_{label}.json"
    artifact_path.write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------------------
# Markdown reporting
# ---------------------------------------------------------------------------


def _hit_rate_mrr(candidates: List[Dict[str, Any]], buggy_file: str) -> Dict[str, Any]:
    hit = 0
    rank_pos = None
    for idx, cand in enumerate(candidates, 1):
        if cand.get("path") == buggy_file:
            hit = 1
            rank_pos = idx
            break
    mrr = 1.0 / rank_pos if rank_pos else 0.0
    return {"hit": hit, "mrr": mrr}


def write_markdown_report(path: Path, data: Dict[str, Any]) -> None:  # noqa: D401
    """Write markdown report summarising a run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Reposage Run Report – {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        fh.write("## Test Results\n")
        fh.write(
            f"Before failures: **{data['before_fail']}**, After failures: **{data['after_fail']}**\n\n"
        )

        fh.write("## Retrieval Metrics\n")
        fh.write(
            f"Baseline Hit@{data['k']}: {data['baseline']['hit']}, MRR: {data['baseline']['mrr']:.2f}\n\n"
        )
        if data.get("learned"):
            fh.write(
                f"Learned Hit@{data['k']}: {data['learned']['hit']}, MRR: {data['learned']['mrr']:.2f}\n\n"
            )

        fh.write("## Token Stats\n")
        fh.write(
            f"Baseline tokens used: {data.get('tokens_baseline', 'n/a')}, Learned tokens used: {data.get('tokens_learned', 'n/a')}\n\n"
        )

        fh.write("## Ablation\n")
        fh.write("Compare baseline vs learned retrieval effectiveness above.\n")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_report(repo_path: Path, *args: Any, **kwargs: Any) -> None:  # noqa: D401
    """Generate markdown report based on artifacts in repo."""

    console.print(f"[green]Generating report for {repo_path}")

    artifacts = repo_path / ".reposage" / "artifacts"
    baseline_path = artifacts / "retrieve_baseline.json"
    learned_path = artifacts / "retrieve_learned.json"
    context_path = artifacts / "context.json"
    scan_before = artifacts / "scan.json"
    test_after = artifacts / "test_after.json"

    if not (baseline_path.exists() and scan_before.exists() and test_after.exists()):
        console.print("[red]Required artifacts missing. Run scan, retrieve, and test first.")
        return

    baseline = json.loads(baseline_path.read_text())
    scan_before_art = json.loads(scan_before.read_text())
    test_after_art = json.loads(test_after.read_text())

    buggy_file = None
    if scan_before_art.get("failed_tests"):
        buggy_file = scan_before_art["failed_tests"][0].get("file")

    metrics = {
        "k": len(baseline),
        "before_fail": scan_before_art.get("failed_count", 0),
        "after_fail": test_after_art.get("failed_count", 0),
        "baseline": _hit_rate_mrr(baseline, buggy_file) if buggy_file else {"hit": 0, "mrr": 0},
    }

    # learned metrics
    if learned_path.exists():
        learned = json.loads(learned_path.read_text())
        metrics["learned"] = _hit_rate_mrr(learned, buggy_file) if buggy_file else {"hit": 0, "mrr": 0}

    # token stats
    if context_path.exists():
        ctx = json.loads(context_path.read_text())
        metrics["tokens_learned"] = ctx.get("tokens_used")

    report_dir = repo_path / "reports"
    report_path = report_dir / f"run_{int(time.time())}.md"
    write_markdown_report(report_path, metrics)
    console.print(f"[green]Report written to {report_path}")