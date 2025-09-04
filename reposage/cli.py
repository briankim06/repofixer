"""Command-line interface for the reposage package.

Provides convenient access to the core functionality via `typer` commands.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .core import apply as apply_module
from .core import patch as patch_module
from .core import ranker as ranker_module
from .core import report as report_module
from .core import retrieve as retrieve_module
from .core import scan as scan_module

__all__ = ["app"]

console = Console()

app = typer.Typer(
    help="reposage – repository scanning, patching, and reporting toolkit.",
    add_completion=False,
)


@app.command()
def scan(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Path to repository to scan"),
    pytest_args: Optional[str] = typer.Option(None, help="Extra arguments passed to pytest"),
) -> None:
    """Run the test suite and capture structured failures."""
    console.rule("[bold blue]Scan")
    scan_module.run_scan(path, pytest_args)


@app.command()
def retrieve(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Repository path to search for fixes"),
    k: int = typer.Option(6, help="Number of top candidate files to retrieve"),
    use_ranker: bool = typer.Option(False, help="Use learned ranker to rerank results"),
) -> None:
    """Retrieve candidate code snippets for fixing failures."""
    console.rule("[bold blue]Retrieve")
    retrieve_module.retrieve_candidates(path, k, use_ranker)


@app.command()
def patch(
    repo_path: Path = typer.Argument(..., exists=True, readable=True, help="Target repository path"),
    dry_run: bool = typer.Option(True, help="Only show patch diff without applying"),
) -> None:
    """Generate a unified diff patch using an LLM prompt template."""
    console.rule("[bold blue]Patch")
    diff_text = patch_module.generate_patch(repo_path)
    console.print(diff_text)
    if not dry_run:
        console.print("[yellow]Dry run disabled — you may apply the patch with `apply`.")


@app.command()
def apply(
    diff_file: Path = typer.Argument(..., exists=True, readable=True, help="Unified diff file to apply"),
    repo_path: Path = typer.Option(Path.cwd(), exists=True, readable=True, help="Repository path where to apply the patch"),
    backup: bool = typer.Option(True, help="Keep backup files for rollback"),
) -> None:
    """Apply a unified diff to the target repository safely."""
    console.rule("[bold blue]Apply")
    apply_module.apply_patch(diff_file, repo_path, backup)


@app.command()
def test(
    repo_path: Path = typer.Argument(..., exists=True, readable=True, help="Repository path to run tests in"),
) -> None:
    """Run tests and store artifact."""
    console.rule("[bold blue]Test")
    from .core import report as report_module

    res = report_module.evaluate_run(repo_path, label="after")
    console.print(
        f"[green]Tests completed – passed={res['passed_count']} failed={res['failed_count']}"
    )


@app.command()
def report(
    repo_path: Path = typer.Argument(..., exists=True, readable=True, help="Repository path"),
) -> None:
    """Generate markdown report from existing artifacts."""
    console.rule("[bold blue]Report")
    report_module.generate_report(repo_path)


# -------------------------------------------------
# Training command
# -------------------------------------------------


@app.command("train-ranker")
def train_ranker_cmd(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Repository path (baseline fixed repo)"),
    n_variants: int = typer.Option(30, help="Number of synthetic variants to generate"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Generate synthetic data and train simple ranker model."""
    console.rule("[bold blue]Train Ranker")

    # Build synthetic dataset
    examples = ranker_module.build_synthetic_training_set(path, n_variants=n_variants, seed=seed)
    model_dir = path / ".reposage" / "models"
    ranker_module.train_ranker(examples, model_dir)


def main() -> None:  # pragma: no cover
    """Entry-point for the `python -m reposage` or `reposage` command."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()