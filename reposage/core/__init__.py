"""Core functionality stubs for reposage.

Each sub-module contains a thin wrapper around future logic. All heavy-lifting will
be implemented incrementally.
"""

from .scan import run_scan  # noqa: F401
from .retrieve import retrieve_candidates  # noqa: F401
from .patch import generate_patch  # noqa: F401
from .apply import apply_patch  # noqa: F401
from .report import generate_report  # noqa: F401

__all__ = [
    "run_scan",
    "retrieve_candidates",
    "generate_patch",
    "apply_patch",
    "generate_report",
]