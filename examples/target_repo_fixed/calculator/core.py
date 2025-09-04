"""Core calculator operations (clean)."""
from __future__ import annotations

from typing import Iterable


def add_many(numbers: Iterable[int]) -> int:
    """Return the sum of *numbers*."""
    return sum(numbers)