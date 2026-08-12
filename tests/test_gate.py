"""Tests for the eval gate (mock mode — no API needed)."""
from __future__ import annotations

from evaluator import Evaluator
from golden_set import GoldenSet


def test_golden_set_loads_30_cases() -> None:
    assert len(GoldenSet()) == 30


def test_mock_scores_are_within_range() -> None:
    gs = GoldenSet()
    ev = Evaluator(mode="mock")
    for case in gs.cases:
        s = ev.score(case)
        assert 0.0 <= s["faithfulness"] <= 1.0
        assert 0.0 <= s["answer_relevancy"] <= 1.0


def test_mock_gate_passes() -> None:
    gs = GoldenSet()
    ev = Evaluator(mode="mock")
    scores = [ev.score(c) for c in gs.cases]
    mean_f = sum(s["faithfulness"] for s in scores) / len(scores)
    assert mean_f >= 0.90, f"mock faithfulness {mean_f:.3f} below bar"
