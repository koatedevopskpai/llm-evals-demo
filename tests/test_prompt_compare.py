import io
import json
from pathlib import Path

import pytest

from bounded_provider import BoundedProvider
from prompt_compare import ROOT, compare, load_cases, reassess_saved


def test_same_thirty_questions_actual_answers_and_failure_publication(tmp_path):
    calls = []

    def respond(request, **kwargs):
        body = json.loads(request.data)
        calls.append(body)
        assert 'reference_answer' not in request.data.decode()
        judge = 'Assess the supplied generated answer' in body['messages'][0]['content']
        if judge:
            # The wrong generation, not a reference or variant ID, must reach the judge.
            assert 'unsupported-999' in body['messages'][1]['content']
        content = '{"faithfulness":0.0,"answer_relevancy":0.0}' if judge else 'unsupported-999'
        return io.BytesIO(json.dumps({
            'model': 'fixture-model',
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20},
            'choices': [{'finish_reason': 'stop', 'message': {'content': content}}],
        }).encode())

    provider = BoundedProvider('private-fixture', tmp_path/'usage.json', 120, 1.0, opener=respond)
    result = compare(provider, tmp_path/'report.json')
    assert result['run_complete'] and result['sample_size_per_variant'] == 30
    assert result['paired_outcomes']['both_fail'] == 30
    assert len(calls) == 120
    assert result['results']['baseline-v1']['summary']['failure_count'] == 30
    assert result['results']['constrained-v2']['summary']['generation_input_tokens'] == 3000
    assert result['results']['baseline-v1']['summary']['judge_output_tokens'] == 600
    assert not result['results']['baseline-v1']['aggregate_quality_gate']['passed']
    assert 'private-fixture' not in (tmp_path/'report.json').read_text()
    first = result['results']['baseline-v1']['rows']
    second = result['results']['constrained-v2']['rows']
    assert [row['case_id'] for row in first] == [row['case_id'] for row in second]
    # First question A then B, second B then A: generation prompts show actual order.
    generation_calls = [body for body in calls if 'response_format' not in body]
    assert generation_calls[0]['messages'][0]['content'] == generation_calls[3]['messages'][0]['content']
    assert generation_calls[1]['messages'][0]['content'] == generation_calls[2]['messages'][0]['content']


def test_transport_failure_remains_a_failure_and_unknown_usage(tmp_path):
    def fail(*args, **kwargs):
        raise TimeoutError('private-provider-message')
    provider = BoundedProvider('dummy-secret', tmp_path/'usage.json', 120, 1.0, opener=fail)
    report = compare(provider, tmp_path/'report.json')
    assert len(provider.records) == 60  # No judges without a generated answer; no retries.
    summary = report['results']['baseline-v1']['summary']
    assert summary['failure_count'] == 30 and summary['attempts_with_unknown_usage'] == 30
    assert summary['failure_categories']['generation_provider_or_parse_failure'] == 30
    text = (tmp_path/'report.json').read_text() + (tmp_path/'usage.json').read_text()
    assert 'private-provider-message' not in text and 'dummy-secret' not in text


def test_cases_cannot_silently_shrink_or_duplicate(tmp_path):
    cases_path = ROOT/'evals/P01-cases.json'
    assert len(load_cases(cases_path)) == 30
    data = json.loads(cases_path.read_text())
    data['cases'] = data['cases'][:3]
    path = tmp_path/'small.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='30-50'):
        load_cases(path)


def test_positive_gate_and_unchanged_reassessment(tmp_path):
    def respond(request, **kwargs):
        body = json.loads(request.data)
        judge = 'response_format' in body
        content = ('{"faithfulness":1.0,"answer_relevancy":1.0}' if judge
                   else 'The context does not state the requested information. [1]')
        return io.BytesIO(json.dumps({
            'model':'fixture', 'usage':{'prompt_tokens':10,'completion_tokens':5},
            'choices':[{'finish_reason':'stop','message':{'content':content}}],
        }).encode())
    provider = BoundedProvider('fixture',tmp_path/'usage.json',120,1.0,opener=respond)
    path = tmp_path/'raw.json'
    original = compare(provider,path)
    assert original['paired_outcomes']['both_pass'] == 30
    assert original['results']['baseline-v1']['aggregate_quality_gate']['passed']
    original_bytes = path.read_bytes()
    # Reproduce the adapter defect on frozen outputs; semantic evidence is untouched.
    for value in original['results'].values():
        for row in value['rows']:
            del row['question'], row['retrieved_contexts']
    path.write_text(json.dumps(original))
    corrected = reassess_saved(path,tmp_path/'corrected.json')
    assert corrected['paired_outcomes']['both_pass'] == 30
    assert corrected['reassessment_provider_calls'] == 0 and len(provider.records) == 120
    assert path.read_bytes() != original_bytes
    frozen = path.read_bytes()
    assert frozen == json.dumps(original).encode()
    assert corrected['execution_mode'] == 'offline_reassessment_of_saved_live_outputs'
    # Changing prompts/cases/thresholds cannot masquerade as a metadata-only repair.
    original['case_sha256'] = 'changed'
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError,match='changed cases'):
        reassess_saved(path,tmp_path/'invalid.json')
