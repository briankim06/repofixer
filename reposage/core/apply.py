"""Safe application of unified diffs with backup/rollback support (stub)."""
from __future__ import annotations

from pathlib import Path
import json
import shutil
import time
from typing import Any, Dict, List, Tuple

from rich.console import Console

from unidiff import PatchSet  # type: ignore

from . import scan as scan_module


console = Console()


def _apply_patched_file(file_path: Path, patched_file) -> Tuple[bool, str]:  # noqa: D401
    """Apply hunks to *file_path*; return (success, reason)."""
    if not file_path.exists():
        return False, "File does not exist"

    try:
        original = file_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        new_lines: List[str] = []
        idx = 0
        for hunk in patched_file:
            # copy unchanged section before hunk
            while idx < hunk.source_start - 1:
                new_lines.append(original[idx])
                idx += 1

            for line in hunk:
                if line.is_context:
                    new_lines.append(original[idx])
                    idx += 1
                elif line.is_removed:
                    idx += 1  # skip line in source
                elif line.is_added:
                    new_lines.append(line.value)

        # append the tail
        new_lines.extend(original[idx:])

        file_path.write_text("".join(new_lines), encoding="utf-8")
        return True, "applied"
    except Exception as exc:  # pragma: no cover
        return False, str(exc)


def apply_unified_diff(target_path: str | Path, diff_text: str) -> Dict[str, Any]:  # noqa: D401
    """Apply unified diff to files under *target_path* with backups."""

    target_path = Path(target_path)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = target_path / ".reposage" / "backups" / timestamp

    patch = PatchSet(diff_text)

    applied: List[str] = []
    rejected: List[str] = []

    for patched_file in patch:
        rel_path = Path(patched_file.path)
        abs_path = target_path / rel_path

        # backup
        backup_path = backup_root / rel_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if abs_path.exists():
            shutil.copy2(abs_path, backup_path)

        success, _ = _apply_patched_file(abs_path, patched_file)
        if success:
            applied.append(str(rel_path))
        else:
            rejected.append(str(rel_path))

    summary = {
        "applied_files": applied,
        "rejected_files": rejected,
        "backup_dir": str(backup_root),
    }
    return summary


def rollback_on_worse(target_path: Path, before_fail_count: int, after_fail_count: int) -> None:  # noqa: D401
    """Restore latest backup if failures increased."""

    if after_fail_count <= before_fail_count:
        return

    backups_dir = target_path / ".reposage" / "backups"
    if not backups_dir.exists():
        console.print("[yellow]No backups found; cannot rollback.")
        return

    latest = sorted(backups_dir.iterdir())[-1]
    console.print(f"[yellow]Rolling back changes from {latest} due to increased failures")

    for backup_file in latest.rglob("*"):
        if backup_file.is_file():
            rel = backup_file.relative_to(latest)
            dest = target_path / rel
            shutil.copy2(backup_file, dest)


def apply_patch(diff_file: Path, repo_path: Path, backup: bool = True, *args: Any, **kwargs: Any) -> None:  # noqa: D401
    """High-level patch apply: backup, apply diff, run tests, rollback if worse."""

    diff_text = diff_file.read_text(encoding="utf-8")

    # before count
    before = scan_module.run_pytest(repo_path)
    before_fail = before.get("failed_count", 0)

    summary = apply_unified_diff(repo_path, diff_text)

    after = scan_module.run_pytest(repo_path)
    after_fail = after.get("failed_count", 0)

    rollback_on_worse(repo_path, before_fail, after_fail)

    console.print(
        f"[green]Patch summary: applied={len(summary['applied_files'])} rejected={len(summary['rejected_files'])}"
    )

    if after_fail > before_fail:
        console.print(
            f"[red]Failures increased from {before_fail} to {after_fail}. Rollback performed if backups available."
        )
    else:
        console.print(f"[green]Failures reduced from {before_fail} to {after_fail}.")