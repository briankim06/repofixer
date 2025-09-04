"""Tiny learned ranker implemented in PyTorch.

This is a placeholder demonstrating where a future featurizer and training loop
will live. The class currently does nothing substantial to avoid unnecessary
heavy dependencies when installing the package without `torch`.
"""
from __future__ import annotations

import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rich.console import Console

from .scan import run_pytest

ROOT_DIR = Path(__file__).resolve().parents[2]  # reposage/

console = Console()

try:
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    console.print("[yellow]PyTorch not found – Ranker is disabled.")
    torch = None  # type: ignore  # noqa: N816


class Ranker:
    """Placeholder ranker model."""

    def __init__(self) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for Ranker but is not installed.")
        import torch.nn as nn  # local import to avoid global dependency when torch missing

        self.input_dim = 2  # query_len, file_len
        self.model: nn.Module = nn.Sequential(
            nn.Linear(self.input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def train(self, data: List[Any]) -> None:  # noqa: D401
        """Train internal model on provided tensors."""
        # Not used – training handled by train_ranker below.
        return

    def rank(self, items: List[Any]) -> List[Any]:
        """Return items ordered by relevance (stub)."""
        # TODO: forward pass and sorting.
        return items

    def save(self, path: Path) -> None:
        """Save model weights to disk (stub)."""
        # TODO: torch.save(...)
        pass

    @classmethod
    def load(cls, path: Path) -> "Ranker":
        """Load model weights from disk (stub)."""
        # TODO: torch.load(...)
        return cls()


# -------------------------------------------------
# Synthetic dataset generation utilities
# -------------------------------------------------


def get_clean_baseline_path() -> Path:  # noqa: D401
    """Return absolute Path to the clean baseline repo."""
    return ROOT_DIR / "examples" / "target_repo_fixed"


_MUTATIONS: List[Tuple[str, str]] = [
    (">", ">="),
    ("<", "<="),
    ("+ 1", ""),  # off-by-one removal
    ("[1:]", "[0:]"),
]


def _apply_random_mutation(py_file: Path, rng: random.Random) -> bool:
    """Apply a single mutation in-place; return True if file mutated."""
    text = py_file.read_text()
    mutations = _MUTATIONS.copy()
    rng.shuffle(mutations)
    for old, new in mutations:
        if old in text:
            mutated = text.replace(old, new, 1)
            py_file.write_text(mutated)
            return True
    return False


def build_synthetic_training_set(target_path: str | Path, n_variants: int = 30, seed: int = 42) -> List[Dict[str, Any]]:  # noqa: D401
    """Generate synthetic training examples and persist them.

    Returns a list of {features, label} dicts.
    """

    rng = random.Random(seed)
    baseline_path = get_clean_baseline_path()

    # Quick sanity check
    baseline_result = run_pytest(baseline_path)
    assert baseline_result["failed_count"] == 0, "Baseline repo must have 0 failures"

    dataset: List[Dict[str, Any]] = []

    for variant_idx in range(n_variants):
        tmp_dir = Path(tempfile.mkdtemp(prefix="reposage_variant_"))
        work_dir = tmp_dir / "repo"
        shutil.copytree(baseline_path, work_dir)

        # pick random .py file
        py_files = list(work_dir.rglob("*.py"))
        rng.shuffle(py_files)
        mutated_file: Path | None = None
        for pf in py_files:
            if _apply_random_mutation(pf, rng):
                mutated_file = pf
                break

        if mutated_file is None:
            console.print(f"[yellow]Variant {variant_idx}: mutation failed, skipping.")
            continue

        # Run tests and collect failures
        res = run_pytest(work_dir)
        if res["failed_count"] == 0:
            console.print(f"[yellow]Variant {variant_idx}: no failing tests after mutation, skipping.")
            continue

        query = " ".join(
            failure.get("nodeid", "") + " " + failure.get("message", "") for failure in res["failed_tests"]
        )

        # Simple feature: query length and file length
        mutated_rel = mutated_file.relative_to(work_dir)
        mut_text = mutated_file.read_text(encoding="utf-8", errors="replace")
        pos_example = {
            "features": {
                "query_len": len(query),
                "file_len": len(mut_text),
            },
            "label": 1,
        }
        dataset.append(pos_example)

        # Negative samples
        negatives = [pf for pf in py_files if pf != mutated_file]
        rng.shuffle(negatives)
        for neg_pf in negatives[:5]:
            neg_text = neg_pf.read_text(encoding="utf-8", errors="replace")
            dataset.append(
                {
                    "features": {
                        "query_len": len(query),
                        "file_len": len(neg_text),
                    },
                    "label": 0,
                }
            )

        # cleanup tmp dir
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Persist dataset
    data_dir = Path(target_path) / ".reposage" / "data" / "ranker"
    data_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = data_dir / "synthetic_dataset.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for ex in dataset:
            fh.write(json.dumps(ex) + "\n")

    console.print(f"[green]Synthetic dataset saved to {jsonl_path} ({len(dataset)} examples)")
    return dataset


def _examples_to_tensors(examples: List[Dict[str, Any]]):
    import torch

    X = torch.tensor([[ex["features"]["query_len"], ex["features"]["file_len"]] for ex in examples], dtype=torch.float32)
    y = torch.tensor([[ex["label"]] for ex in examples], dtype=torch.float32)
    return X, y


def train_ranker(examples: List[Dict[str, Any]], model_dir: Path, epochs: int = 5, lr: float = 1e-3) -> None:  # noqa: D401
    """Train tiny MLP ranker and save artifacts."""

    if torch is None:
        raise RuntimeError("PyTorch not available – cannot train ranker.")

    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import roc_auc_score

    rng = random.Random(0)
    rng.shuffle(examples)
    split = int(0.8 * len(examples))
    train_ex = examples[:split]
    val_ex = examples[split:]

    X_train, y_train = _examples_to_tensors(train_ex)
    X_val, y_val = _examples_to_tensors(val_ex)

    ranker = Ranker()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(ranker.model.parameters(), lr=lr)

    for epoch in range(epochs):
        ranker.model.train()
        optimizer.zero_grad()
        logits = ranker.model(X_train).squeeze(1)
        loss = criterion(logits, y_train.squeeze(1))
        loss.backward()
        optimizer.step()

    # Validation
    ranker.model.eval()
    with torch.no_grad():
        val_logits = ranker.model(X_val).squeeze(1)
        preds = torch.sigmoid(val_logits).cpu().numpy()
        y_true = y_val.squeeze(1).cpu().numpy()
        try:
            auc = roc_auc_score(y_true, preds) if len(set(y_true)) > 1 else None
        except Exception:
            auc = None
        acc = ((preds > 0.5) == y_true).mean()

    console.print(f"[green]Validation accuracy: {acc:.3f} AUC: {auc if auc is not None else 'n/a'}")

    # Save model
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ranker.model.state_dict(), model_dir / "ranker.pt")
    (model_dir / "feats.json").write_text(json.dumps({"features": ["query_len", "file_len"]}))

    console.print(f"[green]Saved model to {model_dir / 'ranker.pt'}")