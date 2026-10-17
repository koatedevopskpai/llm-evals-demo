"""golden_set.py — load and validate the human-curated golden dataset."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class GoldenCase:
    """One evaluation case.

    ``reference_answer`` is the human-authored ground truth. It is NOT the thing
    that gets scored. ``generated_answer`` is what the application under test
    actually produced, and that is what the scorer must see.

    Keeping both as separate fields is the point of E01: the previous
    implementation passed ``reference_answer`` to RAGAS as the answer, so a
    perfect score was guaranteed regardless of what the application produced.
    """

    question: str
    reference_answer: str
    retrieved_contexts: list[str]
    generated_answer: str | None = None


class GoldenSet:
    """Loads evals/golden.json and exposes cases + quality bar."""

    def __init__(self, path: str | None = None) -> None:
        path = path or os.path.join(os.path.dirname(__file__), "evals", "golden.json")
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.meta = data.get("_meta", {})
        self.quality_bar = self.meta.get("quality_bar", {"faithfulness": 0.90, "answer_relevancy": 0.85})
        self.cases: list[GoldenCase] = [
            GoldenCase(
                question=c["question"],
                reference_answer=c["reference_answer"],
                retrieved_contexts=c.get("retrieved_contexts", []),
                generated_answer=c.get("generated_answer"),
            )
            for c in data.get("cases", [])
        ]

    def __len__(self) -> int:
        return len(self.cases)


if __name__ == "__main__":  # sanity check
    gs = GoldenSet()
    assert len(gs) == 30, len(gs)
    print(f"golden set OK: {len(gs)} cases; bar={gs.quality_bar}")
