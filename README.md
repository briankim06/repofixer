# Reposage – Automated Repository Sage

**Reposage** is a research prototype that scans a Python code-base, identifies failing tests, retrieves the most relevant code snippets, drafts patches with an LLM, applies them safely with backup/rollback, and produces rich markdown reports.  It demonstrates an end-to-end loop for _self-healing_ software using a mix of heuristics and learned ranking.

---

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .  # editable install of this package (requires Python ≥3.9)
```
Additional optional dependencies:
* `torch` – for the tiny learned ranker
* `scikit-learn` – TF-IDF features
* `tiktoken`, `unidiff`, `rich`, `typer`

---

## Quick Demo
```bash
python -m reposage.cli scan examples/target_repo
python -m reposage.cli retrieve examples/target_repo --k 6 --use-ranker
python -m reposage.cli patch examples/target_repo --model gpt-4o-mini
python -m reposage.cli apply examples/target_repo --diff .reposage/artifacts/patch.diff
python -m reposage.cli test examples/target_repo
python -m reposage.cli report examples/target_repo
```
Each step produces JSON/markdown artifacts under `.reposage/artifacts/` or `reports/` so you can inspect intermediate results.

---

## Learned Retrieval
The baseline heuristic ranks files by simple signals such as stack-trace presence and lexical TF-IDF similarity.  The **learned ranker** fine-tunes a lightweight MLP on synthetic bugs generated from a clean baseline.  By combining the heuristic score with the model’s relevance probability we achieve higher Hit@K and MRR, meaning the buggy file is surfaced earlier, giving the LLM better context and improving patch quality.

---

## Limitations & Future Work
* Only trivial features are used; richer program analysis and embeddings could improve ranking.
* Patch generation is a stub – integrates with an LLM but prompt/template tuning is pending.
* Apply logic handles text hunks; structural edits (AST) are out of scope.
* Evaluation focuses on test outcomes; semantic regressions may slip through.
* No concurrency or large-scale optimisation yet – suited for small repos.

---

## Ethics & Attribution
AI-generated code and suggestions produced by Reposage are _clearly labelled_ in artifacts (`patch.diff`, reports).  **Human review is required** before merging patches into production.  Use responsibly and respect licence terms of any external models or datasets.  The authors take no liability for generated code.
