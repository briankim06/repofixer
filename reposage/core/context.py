"""Snippet and context management utilities.

Handles token budgeting and code snippet extraction for LLM prompts.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

from rich.console import Console


console = Console()


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    """Return approximate token count (tiktoken if installed, else chars/4)."""

    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except Exception:
        # Simple heuristic: average 4 chars per token
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------


def make_context(file_paths: List[Path], max_tokens: int = 8000) -> str:  # noqa: D401
    """Concatenate files into a string under token budget (simple fallback)."""
    console.print(
        f"[green]Building raw context up to {max_tokens} tokens from {len(file_paths)} files"
    )
    chunks: List[str] = []
    tokens_used = 0
    for fp in file_paths:
        text = fp.read_text(encoding="utf-8", errors="replace")
        t = _count_tokens(text)
        if tokens_used + t > max_tokens:
            break
        chunks.append(f"# File: {fp}\n" + text)
        tokens_used += t
    return "\n\n".join(chunks)


def build_prompt_context(retrieve_artifact: Dict[str, Any], token_budget: int = 2500) -> Dict[str, Any]:  # noqa: D401
    """Select snippets to fit into *token_budget* for LLM prompt building.

    Parameters
    ----------
    retrieve_artifact : dict
        Expected keys: "candidates" (List[dict]) and "scan" (failure details).
    token_budget : int, default 2500

    Returns
    -------
    dict
        Structured prompt context information.
    """

    candidates: List[Dict[str, Any]] = retrieve_artifact.get("candidates", [])
    failures = retrieve_artifact.get("scan", {}).get("failed_tests", [])

    failure_summary = "\n".join(
        f"{item.get('nodeid', '')}: {item.get('message', '')}" for item in failures
    )

    selected: List[Dict[str, Any]] = []
    tokens_used = _count_tokens(failure_summary)

    for cand in candidates:
        if tokens_used >= token_budget:
            break

        snippet = cand.get("snippet", "")
        snip_tokens = _count_tokens(snippet)
        if tokens_used + snip_tokens > token_budget:
            continue
        selected.append({
            "path": cand.get("path"),
            "tokens": snip_tokens,
            "snippet": snippet,
            "score": cand.get("combined_score", cand.get("score")),
        })
        tokens_used += snip_tokens

    return {
        "failure_summary": failure_summary,
        "selected": selected,
        "token_budget": token_budget,
        "tokens_used": tokens_used,
    }