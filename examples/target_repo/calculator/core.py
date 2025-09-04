"""Core calculator operations with an intentional off-by-one bug."""
from __future__ import annotations

from typing import Iterable


def add_many(numbers: Iterable[int]) -> int:
    """Return the sum of *numbers* (bug: skips first element)."""
    nums = list(numbers)
    # BUG: should sum(nums) but slices from index 1
    return sum(nums[1:])