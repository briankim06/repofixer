"""Tests for the buggy calculator."""
from __future__ import annotations

import pytest

from calculator import add_many


def test_add_many_simple():
    assert add_many([1, 2, 3]) == 6  # will fail (returns 5)


def test_add_many_empty():
    assert add_many([]) == 0


def test_add_many_single():
    assert add_many([42]) == 42  # will fail (returns 0)