"""Known failures must be rejected independently of aggregate scores."""
from copy import deepcopy
import json
import math

import pytest

from gate_policy import assess_rows, json_safe, load_policy
from run_evals import main


def row(score=0.96):
    return {"question": "When is payment due?", "retrieved_contexts": ["Payment is due in 30 days."],
            "generated_answer": "Payment is due in 30 days. [1]", "faithfulness": score, "answer_relevancy": score}


def test_acceptable_development_fixture_passes():
    policy = load_policy()
    rows = [{**row(), **scores} for scores in policy['baseline_fixture_scores']]
    assert assess_rows(rows, policy, len(rows))['passed']


@pytest.mark.parametrize('change,reason', [
    ({'retrieved_contexts': []}, 'missing_context'),
    ({'retrieved_contexts': 42}, 'missing_context'),
    ({'generated_answer': 'Payment is due in 30 days. [99]'}, 'unknown_citation'),
    ({'generated_answer': 'Payment is due in 30 days. [source invented]'}, 'unknown_citation'),
    ({'generated_answer': ''}, 'missing_generated_answer'),
    ({'judge_error': 'TimeoutError'}, 'judge_error'),
    ({'faithfulness': float('nan')}, 'invalid_metric:faithfulness'),
    ({'faithfulness': None}, 'invalid_metric:faithfulness'),
    ({'faithfulness': True}, 'invalid_metric:faithfulness'),
    ({'answer_relevancy': float('inf')}, 'invalid_metric:answer_relevancy'),
    ({'answer_relevancy': 1.1}, 'invalid_metric:answer_relevancy'),
])
def test_structured_failure_cannot_be_hidden_by_high_scores(change, reason):
    result = assess_rows([{**row(), **change}], load_policy(), 1)
    assert not result['passed']
    assert reason in {f['reason'] for f in result['hard_failures']}


def test_missing_metric_is_rejected():
    candidate = row()
    del candidate['faithfulness']
    assert not assess_rows([candidate], load_policy(), 1)['passed']


def test_malformed_row_is_rejected_without_crashing():
    assert not assess_rows([None], load_policy(), 1)['passed']


def test_harmful_case_cannot_be_diluted_by_many_good_cases():
    rows = [row(1.0) for _ in range(20)] + [row(0.0)]
    result = assess_rows(rows, load_policy(), len(rows))
    assert result['means']['faithfulness'] > 0.92
    assert not result['passed']


@pytest.mark.parametrize('rows,count', [([], 0), ([], 1), ([row()], 2)])
def test_empty_or_incomplete_runs_fail(rows, count):
    assert not assess_rows(rows, load_policy(), count)['passed']


def test_rejected_nan_is_serialised_as_null():
    assert json.dumps(json_safe({'score': math.nan}), allow_nan=False) == '{"score": null}'


def test_cli_judge_failure_writes_failed_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only')
    def broken(case):
        raise TimeoutError('private provider message must not be published')
    report = tmp_path / 'failed.md'
    assert main(['--mode', 'real', '--max-cases', '1', '--report', str(report)],
                complete_fn=lambda messages: 'A generated answer.', scorer=broken) == 1
    result = json.loads(report.with_suffix('.json').read_text())
    assert result['status'] == 'live_fail'
    assert result['rows'][0]['judge_error'] == 'TimeoutError'
    assert 'private provider message' not in json.dumps(result)


def test_cli_malformed_metric_writes_failed_standard_json(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only')
    report = tmp_path / 'nan.md'
    assert main(['--mode', 'real', '--max-cases', '1', '--report', str(report)],
                complete_fn=lambda messages: 'A generated answer.',
                scorer=lambda case: {'faithfulness': float('nan'), 'answer_relevancy': 1.0}) == 1
    assert 'NaN' not in report.with_suffix('.json').read_text()


def test_cli_zero_selected_cases_is_a_failed_report(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only')
    import run_evals
    class EmptySet:
        cases = []
    monkeypatch.setattr(run_evals, 'GoldenSet', EmptySet)
    report = tmp_path / 'empty.md'
    assert main(['--mode', 'real', '--report', str(report)],
                complete_fn=lambda messages: 'unused', scorer=lambda case: {}) == 1
    assert json.loads(report.with_suffix('.json').read_text())['case_count'] == 0
