"""evaluator.py - score the answer the application produced, not the reference.

E01 contract:

  * ``reference_answer`` is ground truth and is never scored.
  * ``generated_answer`` is what the application output, and is what the scorer
    receives. If it is absent, scoring raises rather than substituting anything.
  * Real mode requires a provider credential and refuses to load an SDK without
    one, so a missing key cannot produce a passing live-quality result.

Mock mode is unchanged: deterministic placeholders for wiring checks only.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from golden_set import GoldenCase

#: A scoring function: GoldenCase -> {"faithfulness": float, "answer_relevancy": float}
Scorer = Callable[[GoldenCase], dict[str, float]]


class MissingGeneratedAnswer(RuntimeError):
    """Raised when real scoring is attempted without a generated answer.

    This is deliberately fatal. The failure mode it prevents is the one E01
    exists to fix: quietly scoring the reference answer and reporting a pass.
    """


class Evaluator:
    """Scores faithfulness + answer relevancy per golden case."""

    def __init__(self, mode: str = "mock", scorer: Scorer | None = None) -> None:
        if mode not in {"mock", "real"}:
            raise ValueError(f"Unsupported evaluation mode: {mode}")
        if mode == "real" and not os.getenv("OPENAI_API_KEY"):
            # Fail before loading SDKs or issuing a provider request. A missing
            # credential must not be able to yield a passing live-quality result.
            raise RuntimeError(
                "real mode requires OPENAI_API_KEY; no provider call was attempted"
            )
        self.mode = mode
        # Injectable so tests can exercise the contract without network access.
        self._scorer = scorer
        self._ragas: dict[str, Any] | None = None

    @property
    def mocks_provider_calls(self) -> bool:
        """True when scoring does not touch a provider (mock mode or injected scorer)."""
        return self.mode == "mock" or self._scorer is not None

    @staticmethod
    def _load_ragas() -> dict[str, Any]:
        try:
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, faithfulness
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "RAGAS not installed - run `pip install -r requirements.txt` "
                "or use `--mode mock`."
            ) from exc
        return {
            "evaluate": evaluate,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
        }

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
        generated = (case.generated_answer or "").strip()
        if not generated:
            raise MissingGeneratedAnswer(
                f"case {case.question!r} has no generated_answer; refusing to score "
                "the reference answer in its place"
            )
        if self._scorer is not None:
            return self._scorer(case)
        return self._ragas_score(case, generated)

    @staticmethod
    def ragas_payload(case: GoldenCase, generated: str) -> dict[str, list]:
        """The exact RAGAS input for a case.

        Separated from the transport so the contract can be asserted directly,
        without the datasets/ragas dependency chain. Note what is absent:
        ``reference_answer`` is never placed in the payload, because faith-
        fulness and answer_relevancy must be computed against the generated
        output.
        """
        return {
            "question": [case.question],
            "answer": [generated],
            "contexts": [case.retrieved_contexts],
        }

    def _ragas_score(self, case: GoldenCase, generated: str) -> dict[str, float]:
        if self._ragas is None:
            self._ragas = self._load_ragas()
        from datasets import Dataset

        result = self._ragas["evaluate"](
            Dataset.from_dict(self.ragas_payload(case, generated)),
            metrics=[self._ragas["faithfulness"], self._ragas["answer_relevancy"]],
        )
        row = result.to_pandas().iloc[0]
        return {
            "faithfulness": round(float(row.get("faithfulness", 0.0)), 3),
            "answer_relevancy": round(float(row.get("answer_relevancy", 0.0)), 3),
        }


def provider_hint() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("note: OPENAI_API_KEY not set - real mode will refuse; use --mode mock in CI.")
