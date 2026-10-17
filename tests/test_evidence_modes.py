"""No paid inference: exercise the distinction between checks and evidence.

Updated for the E01 contract. Two things are preserved from the earlier
fail-closed design and must stay true:

  * a mock pass can never satisfy a live-quality requirement;
  * a real run without a credential must refuse *before* loading an SDK.
"""

from __future__ import annotations

import json

import pytest

from evaluator import Evaluator
from run_evals import main


def test_mock_success_is_explicitly_not_live_quality(tmp_path):
    report = tmp_path / "mock.md"
    assert main(["--mode", "mock", "--report", str(report)]) == 0
    result = json.loads(report.with_suffix(".json").read_text())
    assert result["status"] == "offline_pass"
    assert result["evaluator_identity"] == "deterministic-placeholder-v1"
    assert result["live_quality_established"] is False
    assert "cannot authorise" in report.read_text()


def test_mock_can_never_satisfy_a_live_quality_requirement(tmp_path):
    report = tmp_path / "mock-release.md"
    report.write_text("stale successful report")
    assert main(["--mode", "mock", "--require-live-quality", "--report", str(report)]) == 2
    result = json.loads(report.with_suffix(".json").read_text())
    assert result["status"] == "blocked"
    assert result["rows"] == []
    assert result["live_quality_established"] is False


def test_real_mode_without_a_key_fails_closed_before_loading_sdk(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden():
        pytest.fail("RAGAS must not load without a credential")

    monkeypatch.setattr(Evaluator, "_load_ragas", staticmethod(forbidden))
    report = tmp_path / "release.md"
    report.write_text("stale successful report")
    assert main(["--mode", "real", "--require-live-quality", "--report", str(report)]) == 2
    result = json.loads(report.with_suffix(".json").read_text())
    assert result["status"] == "blocked"
    assert result["rows"] == []
    assert result["live_quality_established"] is False
    assert "OPENAI_API_KEY" in report.read_text()


def test_evaluator_real_mode_requires_a_credential(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        Evaluator("real")


def test_real_run_with_no_generated_answers_cannot_pass(tmp_path, monkeypatch):
    """A credential alone is not enough: without generated answers there is no pass.

    The injected scorer would return a perfect score if it were ever asked, so
    this fails only because the unanswerable cases are scored as failures.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    report = tmp_path / "noanswers.md"
    code = main(
        ["--mode", "real", "--max-cases", "2", "--report", str(report)],
        complete_fn=lambda messages: "",  # the application produced nothing
        scorer=lambda case: {"faithfulness": 1.0, "answer_relevancy": 1.0},
    )
    result = json.loads(report.with_suffix(".json").read_text())
    assert code == 1
    assert result["status"] == "live_fail"
    assert result["unanswerable_cases"] == 2
    assert result["live_quality_established"] is True


def test_mock_threshold_failure_remains_a_failure(tmp_path):
    assert main(["--threshold", "1", "--report", str(tmp_path / "failed.md")]) == 1


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.1"])
def test_invalid_threshold_cannot_authorise_a_pass(value):
    with pytest.raises(SystemExit) as exc:
        main(["--threshold", value])
    assert exc.value.code == 2
