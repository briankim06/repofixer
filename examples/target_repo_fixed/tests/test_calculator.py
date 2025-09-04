"""Tests for clean calculator."""
from __future__ import annotations

from calculator import add_many


def test_add_many_simple():
    assert add_many([1, 2, 3]) == 6


def test_add_many_empty():
    assert add_many([]) == 0


def test_add_many_single():
    assert add_many([42]) == 42