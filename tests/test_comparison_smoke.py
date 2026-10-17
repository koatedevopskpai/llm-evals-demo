import json

import pytest

from compare_smoke import compare, percentile


def test_percentiles_have_a_named_small_sample_interpretation():
    assert percentile([3, 1, 2], .5) == 2
    assert percentile([3, 1, 2], .95) == pytest.approx(2.9)
    with pytest.raises(ValueError):
        percentile([], .95)


class FixtureProvider:
    def __init__(self):
        self.records = []

    def complete(self, messages, model, purpose, json_output=False):
        rec = {'purpose': purpose, 'latency_seconds': 1 if purpose == 'generation' else 2,
               'estimated_usd': .001, 'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}
        self.records.append(rec)
        text = json.dumps({'faithfulness': 1, 'answer_relevancy': 1}) if purpose == 'judge' else 'A fixture answer. [1]'
        return text, rec


def test_same_questions_both_configurations_with_separate_judge_cost(tmp_path):
    provider = FixtureProvider()
    report = compare(provider, tmp_path/'report.json')
    assert len(provider.records) == 12
    a, b = report['results'].values()
    assert [r['case_id'] for r in a['rows']] == [r['case_id'] for r in b['rows']]
    for result in (a, b):
        assert result['summary']['case_count'] == 3
        assert result['summary']['generation_estimated_usd'] == pytest.approx(.003)
        assert result['summary']['judge_estimated_usd'] == pytest.approx(.003)
        assert result['summary']['p95_end_to_end_seconds'] == 3
        assert result['summary']['automatic_retry_count'] == 0


def test_failed_attempts_remain_in_the_population_and_usage_is_unknown(tmp_path):
    class Broken(FixtureProvider):
        def complete(self, messages, model, purpose, json_output=False):
            self.records.append({'purpose': purpose, 'latency_seconds': 4})
            raise TimeoutError('provider failed')
    result = compare(Broken(), tmp_path/'failed.json')
    for value in result['results'].values():
        assert value['summary']['failure_count'] == 3
        assert value['summary']['p95_end_to_end_seconds'] == 4
        assert value['summary']['attempts_with_unknown_usage'] == 3
    assert result['provisional_lower_generation_cost_option'] is None
