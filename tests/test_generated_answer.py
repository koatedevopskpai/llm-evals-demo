"""E01 acceptance tests: the scorer must see the generated answer.

These run fully offline. The real-mode path is exercised by injecting the
generator and the scorer, so what is under test is the contract - which text
reaches the scorer - rather than a provider's behaviour.

The bug being guarded against (found in review, `evaluator.py:60` before E01):

    "answer": [case.reference_answer],  # reference answer stands in for the
                                        # generated answer

With that line present, a perfect score was guaranteed no matter what the
application produced.
"""

from __future__ import annotations

import json

import pytest

from evaluator import Evaluator, MissingGeneratedAnswer
from golden_set import GoldenCase
from run_evals import main

REFERENCE = "The 2024 revenue was $391.0 billion."


@pytest.fixture(autouse=True)
def _credential(monkeypatch):
    """Real-mode construction requires a key. No call is made by these tests:
    every one of them injects ``complete_fn`` and/or ``scorer``."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")


def _case(generated: str | None) -> GoldenCase:
    return GoldenCase(
        question="What was the 2024 revenue?",
        reference_answer=REFERENCE,
        retrieved_contexts=["FY2024 total net sales were $391.0 billion."],
        generated_answer=generated,
    )


def _scorer(scores: dict[str, float]):
    """A scorer that records the case it was given, then returns fixed scores."""

    class _Recorder:
        def __init__(self) -> None:
            self.seen: list[GoldenCase] = []

        def __call__(self, case: GoldenCase) -> dict[str, float]:
            self.seen.append(case)
            return dict(scores)

    return _Recorder()


class TestGeneratedAnswerContract:
    def test_scorer_receives_the_generated_answer_not_the_reference(self):
        rec = _scorer({"faithfulness": 0.5, "answer_relevancy": 0.5})
        evaluator = Evaluator(mode="real", scorer=rec)
        case = _case("Revenue was $391.0 billion in 2024.")

        evaluator.score(case)

        assert rec.seen[0].generated_answer == "Revenue was $391.0 billion in 2024."
        assert rec.seen[0].generated_answer != REFERENCE

    def test_scoring_refuses_when_there_is_no_generated_answer(self):
        evaluator = Evaluator(mode="real", scorer=_scorer({}))
        with pytest.raises(MissingGeneratedAnswer):
            evaluator.score(_case(None))

    def test_blank_generated_answer_is_also_refused(self):
        evaluator = Evaluator(mode="real", scorer=_scorer({}))
        with pytest.raises(MissingGeneratedAnswer):
            evaluator.score(_case("   "))

    def test_ragas_input_contains_the_generated_answer_and_no_reference(self, monkeypatch):
        """The RAGAS payload must carry the generated answer and nothing else.

        Asserted directly on the payload builder, so the offline environment
        does not need datasets or ragas installed.
        """
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
        case = _case("Revenue was $391.0 billion in 2024.")

        payload = Evaluator.ragas_payload(case, case.generated_answer or "")

        assert payload["answer"] == ["Revenue was $391.0 billion in 2024."]
        assert payload["answer"] != [REFERENCE]
        assert REFERENCE not in json.dumps(payload)
        assert "generated_answer" not in payload  # only RAGAS field names are sent


class TestGateOutcome:
    """The acceptance criterion: a wrong generated answer fails even when the
    reference answer is correct."""

    def test_wrong_generated_answer_fails_while_reference_is_correct(self, tmp_path, monkeypatch):
        """main() loads the real golden set, so assert on the case it actually used:
        a reference answer was available and correct, yet the wrong generated
        answer still failed."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
        report = tmp_path / "wrong.md"
        rec = _scorer({"faithfulness": 0.10, "answer_relevancy": 0.20})

        code = main(
            ["--mode", "real", "--max-cases", "1", "--report", str(report)],
            complete_fn=lambda messages: "Revenue was $999 billion in 2024.",  # noqa: ARG005
            scorer=rec,
        )

        payload = json.loads(report.with_suffix(".json").read_text())
        assert code == 1, "a wrong answer must fail the gate"
        assert payload["status"] == "live_fail"
        assert payload["means"]["faithfulness"] < 0.90
        # Ground truth was present and was not used as the scored answer.
        assert rec.seen[0].reference_answer
        assert rec.seen[0].reference_answer != rec.seen[0].generated_answer
        assert rec.seen[0].generated_answer == "Revenue was $999 billion in 2024."
        assert "999" in payload["rows"][0]["generated_answer"]

    def test_correct_generated_answer_passes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
        report = tmp_path / "good.md"

        code = main(
            ["--mode", "real", "--max-cases", "1", "--report", str(report)],
            complete_fn=lambda messages: "Revenue was $391.0 billion in 2024.",  # noqa: ARG005
            scorer=_scorer({"faithfulness": 0.97, "answer_relevancy": 0.95}),
        )

        payload = json.loads(report.with_suffix(".json").read_text())
        assert code == 0
        assert payload["status"] == "live_pass"
        assert payload["live_quality_established"] is True

    def test_generated_answer_is_persisted_in_the_report(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
        report = tmp_path / "persist.md"

        main(
            ["--mode", "real", "--max-cases", "1", "--report", str(report)],
            complete_fn=lambda messages: "Answer text produced by the application.",  # noqa: ARG005
            scorer=_scorer({"faithfulness": 0.99, "answer_relevancy": 0.99}),
        )

        payload = json.loads(report.with_suffix(".json").read_text())
        assert payload["rows"][0]["generated_answer"] == "Answer text produced by the application."
        assert payload["rows"][0]["question"]

    def test_unanswerable_case_scores_zero_rather_than_being_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
        report = tmp_path / "empty.md"

        code = main(
            ["--mode", "real", "--max-cases", "1", "--report", str(report)],
            complete_fn=lambda messages: "",  # the application produced nothing
            scorer=_scorer({"faithfulness": 0.99, "answer_relevancy": 0.99}),
        )

        payload = json.loads(report.with_suffix(".json").read_text())
        assert code == 1
        assert payload["status"] == "live_fail"
        assert payload["unanswerable_cases"] == 1
        assert payload["rows"][0]["faithfulness"] == 0.0
