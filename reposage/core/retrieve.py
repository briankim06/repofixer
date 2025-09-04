"""Candidate retrieval module.

Combines heuristic search with a learned ranker to select promising code snippets
for automated patch generation.
"""
from __future__ import annotations

from pathlib import Path
import ast
import json
import re
from typing import Any, Dict, List, Set

from rich.console import Console

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .scan import _build_query
from . import context as context_module


console = Console()

EXCLUDE_DIRS = {"venv", "env", "__pycache__"}


def _iter_py_files(base: Path) -> List[Path]:
    files: List[Path] = []
    for path in base.rglob("*.py"):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def _gather_imports(py_file: Path) -> Set[str]:
    """Return module names imported in *py_file* (best-effort)."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except Exception:
        return set()

    imports: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def _relative_module_name(base: Path, file_path: Path) -> str:
    rel = file_path.relative_to(base).with_suffix("")
    return ".".join(rel.parts)


def _build_query(scan_artifact: Dict[str, Any]) -> str:
    parts: List[str] = []
    for failure in scan_artifact.get("failed_tests", []):
        parts.extend([failure.get("nodeid", ""), failure.get("message", "")])
    return " ".join(parts)


def _make_snippet(file_path: Path, focus_line: int | None = None) -> str:
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if focus_line and 1 <= focus_line <= len(lines):
        start = max(0, focus_line - 31)
        end = min(len(lines), focus_line + 30)
        snippet_lines = lines[start:end]
    else:
        snippet_lines = lines[:120]
    return "\n".join(snippet_lines)


def baseline_retrieve(target_path: str | Path, scan_artifact: Dict[str, Any], k: int = 6) -> List[Dict[str, Any]]:
    """Score & rank files heuristically based on scan results."""

    base = Path(target_path)
    referenced_files: Set[Path] = {
        base / failure.get("file") for failure in scan_artifact.get("failed_tests", []) if failure.get("file")
    }

    # Walk files
    candidates = _iter_py_files(base)

    # Precompute imports for trace files
    trace_imports: Set[str] = set()
    for pf in referenced_files:
        trace_imports.update(_gather_imports(pf))

    module_name_map = {c: _relative_module_name(base, c) for c in candidates}

    # TF-IDF similarity
    query_text = _build_query(scan_artifact)
    corpus = [query_text] + [c.read_text(encoding="utf-8", errors="replace") for c in candidates]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    ranked: List[Dict[str, Any]] = []
    for idx, file_path in enumerate(candidates):
        score = 0.0
        if file_path in referenced_files:
            score += 3.0
        if module_name_map[file_path] in trace_imports:
            score += 1.0
        score += sims[idx]  # similarity in [0,1]

        focus_line = None
        for failure in scan_artifact.get("failed_tests", []):
            if (base / failure.get("file")) == file_path and failure.get("line", -1) > 0:
                focus_line = int(failure["line"])
                break

        snippet = _make_snippet(file_path, focus_line)

        ranked.append({
            "path": str(file_path.relative_to(base)),
            "score": score,
            "snippet": snippet,
        })

    ranked.sort(key=lambda d: d["score"], reverse=True)
    return ranked[:k]


# ---------------------------------------------------------------------------
# Learned re-ranking
# ---------------------------------------------------------------------------


def learned_rerank(
    candidates: List[Dict[str, Any]],
    query_text: str,
    model_path: Path,
    feats_cfg_path: Path,
    repo_base: Path,
    top_k: int = 6,
) -> List[Dict[str, Any]]:  # noqa: D401
    """Combine heuristic score with learned ranker probability."""

    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError:
        console.print("[red]PyTorch not installed – cannot use learned reranker.")
        return candidates[:top_k]

    if not (model_path.exists() and feats_cfg_path.exists()):
        console.print("[yellow]Model files not found – fallback to heuristic.")
        return candidates[:top_k]

    feats_cfg = json.loads(feats_cfg_path.read_text())
    feat_names = feats_cfg.get("features", [])

    # Build feature matrix
    X: List[List[float]] = []
    for cand in candidates:
        row: List[float] = []
        for name in feat_names:
            if name == "query_len":
                row.append(len(query_text))
            elif name == "file_len":
                file_path = repo_base / cand["path"]
                try:
                    row.append(len(file_path.read_text(encoding="utf-8", errors="replace")))
                except Exception:
                    row.append(0.0)
            else:
                row.append(0.0)
        X.append(row)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Define model architecture consistent with training (2->16->1)
    model = nn.Sequential(nn.Linear(len(feat_names), 16), nn.ReLU(), nn.Linear(16, 1))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(X_tensor).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()

    for cand, prob in zip(candidates, probs):
        cand["model_prob"] = float(prob)
        cand["combined_score"] = 0.5 * cand["score"] + 0.5 * cand["model_prob"]

    candidates.sort(key=lambda d: d["combined_score"], reverse=True)
    return candidates[:top_k]


def retrieve_candidates(repo_path: Path, k: int = 6, use_ranker: bool = False) -> List[Dict[str, Any]]:  # noqa: D401
    """High-level wrapper for CLI. Baseline then optional learned rerank."""

    console.print(f"[green]Retrieving candidates from {repo_path} (k={k}, use_ranker={use_ranker})")

    scan_path = repo_path / ".reposage" / "artifacts" / "scan.json"
    if not scan_path.exists():
        console.print("[red]Scan artifact not found. Run `reposage scan` first.")
        return []

    scan_artifact = json.loads(scan_path.read_text())
    baseline = baseline_retrieve(repo_path, scan_artifact, k=max(k, 20))  # compute more for rerank

    artifacts_dir = repo_path / ".reposage" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "retrieve_baseline.json").write_text(json.dumps(baseline, indent=2))

    final_candidates = baseline[:k]

    if use_ranker:
        model_dir = repo_path / ".reposage" / "models"
        model_path = model_dir / "ranker.pt"
        feats_cfg_path = model_dir / "feats.json"
        query_text = _build_query(scan_artifact)
        final_candidates = learned_rerank(
            baseline,
            query_text,
            model_path,
            feats_cfg_path,
            repo_base=repo_path,
            top_k=k,
        )
        (artifacts_dir / "retrieve_learned.json").write_text(json.dumps(final_candidates, indent=2))

    console.print(
        "\n".join(
            f"[cyan]{item['path']}[/cyan] score={item.get('combined_score', item['score']):.3f}" for item in final_candidates
        )
    )

    # Build prompt context and save
    context_dict = context_module.build_prompt_context(
        {"candidates": final_candidates, "scan": scan_artifact},
        token_budget=2500,
    )
    (artifacts_dir / "context.json").write_text(json.dumps(context_dict, indent=2))

    return final_candidates