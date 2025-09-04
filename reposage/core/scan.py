"""Test-suite execution and failure capture.

TODO: integrate with pytest programmatically and serialize failures in a structured
format (e.g., JSON). For now this is a no-op stub.
"""
from __future__ import annotations

from pathlib import Path
import json
import re
import subprocess
from typing import List, Optional, Dict, Any

from rich.console import Console


console = Console()


FAILURE_HEADER_RE = re.compile(r"_{2,}\s+(?P<nodeid>.+?)\s+_{2,}")
SUMMARY_RE = re.compile(
    r"(?P<failed>\d+) failed.*?(?P<passed>\d+) passed|(?P<passed_only>\d+) passed"
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def _build_query(scan_artifact: Dict[str, Any]) -> str:  # noqa: D401
    """Build textual query from pytest scan artifact.

    Concatenate failed test nodeids, messages, and tail of tracebacks to feed
    downstream retrieval components.
    """

    if not scan_artifact:
        return ""

    failed = scan_artifact.get("failed_tests") or []
    parts: List[str] = []
    for ft in failed:
        nodeid = ft.get("nodeid") or ""
        msg = ft.get("message") or ""
        traceback_text = ft.get("traceback_text") or ""

        if nodeid:
            parts.append(nodeid)
        if msg:
            parts.append(msg)
        if traceback_text:
            short_tb = "\n".join(traceback_text.strip().splitlines()[-3:])
            parts.append(short_tb)

    return " \n".join(p for p in parts if p)


def _parse_failures(output: str) -> List[Dict[str, Any]]:
    """Extract basic failure info from pytest output using regex heuristics."""
    lines = output.splitlines()
    failures: List[Dict[str, Any]] = []
    current: List[str] = []
    nodeid: Optional[str] = None

    for line in lines:
        header_match = FAILURE_HEADER_RE.match(line.strip())
        if header_match:
            # flush previous
            if nodeid is not None:
                failures.append(
                    {
                        "nodeid": nodeid,
                        "traceback_text": "\n".join(current),
                    }
                )
                current = []
            nodeid = header_match.group("nodeid").strip()
            continue
        if nodeid is not None:
            current.append(line)

    # final flush
    if nodeid is not None:
        failures.append(
            {
                "nodeid": nodeid,
                "traceback_text": "\n".join(current),
            }
        )

    # Derive file, line, message heuristically from traceback text
    for entry in failures:
        tb_lines = entry["traceback_text"].splitlines()
        file_line_re = re.compile(r"^>\s+(.+):(\d+).*")
        msg_re = re.compile(r"E\s+(.+)")
        file_: str = ""
        line_no: int = -1
        message: str = ""
        for l in tb_lines:
            m = file_line_re.match(l.strip())
            if m:
                file_, line_no = m.group(1), int(m.group(2))
            m2 = msg_re.match(l.strip())
            if m2:
                message = m2.group(1)
        entry.update({"file": file_, "line": line_no, "message": message})
    return failures


def run_pytest(target_path: str | Path, pytest_args: Optional[str] | None = None) -> Dict[str, Any]:
    """Execute pytest in *target_path* and return structured results."""

    cmd = ["pytest", "-q"]
    if pytest_args:
        cmd.extend(pytest_args.split())

    result = subprocess.run(
        cmd,
        cwd=str(target_path),
        capture_output=True,
        text=True,
    )

    output = result.stdout + "\n" + result.stderr

    # Parse summary
    failed_count = 0
    passed_count = 0
    for match in SUMMARY_RE.finditer(output):
        if match.group("passed_only"):
            passed_count = int(match.group("passed_only"))
        else:
            failed_count = int(match.group("failed"))
            passed_count = int(match.group("passed"))

    failures = _parse_failures(output) if failed_count else []

    return {
        "returncode": result.returncode,
        "failed_tests": failures,
        "passed_count": passed_count,
        "failed_count": failed_count,
    }


def run_scan(repo_path: Path, pytest_args: Optional[str] | None = None) -> None:
    """Run pytest on *repo_path* and persist structured scan results."""

    console.print(f"[green]Running pytest in {repo_path}...")
    scan_result = run_pytest(repo_path, pytest_args)

    # Persist JSON
    artifacts_dir = repo_path / ".reposage" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    json_path = artifacts_dir / "scan.json"
    json_path.write_text(json.dumps(scan_result, indent=2))

    # Summary
    console.print(
        f"[bold]Pytest completed[/bold] rc={scan_result['returncode']} "
        f"| passed={scan_result['passed_count']} failed={scan_result['failed_count']}"
    )

    if scan_result["failed_count"]:
        console.print("[red]Some tests failed. See scan.json for details.")
    else:
        console.print("[green]All tests passed!")


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "run_pytest",
    "run_scan",
    "_build_query",
]