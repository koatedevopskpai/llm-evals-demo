"""Offline placeholder evaluator; live scoring is disabled pending E01."""

from __future__ import annotations

import os
from typing import Any

from golden_set import GoldenCase


class Evaluator:
    """Scores faithfulness + answer relevancy per golden case.

    Mock mode returns deterministic placeholders for wiring checks. The legacy
    RAGAS code below substitutes the reference answer and is deliberately
    inaccessible through supported construction until E01 repairs measurement.
    """

    def __init__(self, mode: str = "mock") -> None:
        if mode not in {"mock", "real"}:
            raise ValueError(f"Unsupported evaluation mode: {mode}")
        if mode == "real":
            # E01 is not implemented: current scoring substitutes the reference.
            # Fail before loading SDKs or issuing a provider request.
            raise RuntimeError(
                "Live quality evaluation unavailable until E01 evaluates generated answers."
            )
        self.mode = mode
        self._ragas = None

    @staticmethod
    def _load_ragas() -> Any:
        try:
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, faithfulness

            return {"evaluate": evaluate, "faithfulness": faithfulness, "answer_relevancy": answer_relevancy}
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "RAGAS not installed — run `pip install -r requirements.txt` "
                "or use `--mode mock`."
            ) from exc

    def score(self, case: GoldenCase) -> dict[str, float]:
        if self.mode == "mock":
            return self._mock_score(case)
        return self._real_score(case)

    def _mock_score(self, case: GoldenCase) -> dict[str, float]:
        # Deterministic placeholder: a "good" pipeline scores near the bar.
        # Used to prove the gate wiring; replace with real numbers via --mode real.
        support = min(1.0, 0.92 + 0.004 * len(case.retrieved_contexts))
        if case.question.startswith("How"):
            support = max(0.90, support - 0.01)
        return {
            "faithfulness": round(support, 3),
            "answer_relevancy": round(min(1.0, support + 0.02), 3),
        }

    def _real_score(self, case: GoldenCase) -> dict[str, float]:
        # Build a single-sample HF dataset for RAGAS.
        from datasets import Dataset

        data = {
            "question": [case.question],
            "answer": [case.reference_answer],  # reference answer stands in for the generated answer
            "contexts": [case.retrieved_contexts],
        }
        ds = Dataset.from_dict(data)
        result = self._ragas["evaluate"](
            ds,
            metrics=[self._ragas["faithfulness"], self._ragas["answer_relevancy"]],
        )
        s = result.to_pandas().iloc[0]
        return {
            "faithfulness": round(float(s.get("faithfulness", 0.0)), 3),
            "answer_relevancy": round(float(s.get("answer_relevancy", 0.0)), 3),
        }


def provider_hint() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("note: OPENAI_API_KEY not set — real mode will fail; use --mode mock in CI.")
