import io
import json

import pytest

from bounded_provider import BoundedProvider, ProviderFailure, judge_answer


def response(content='answer'):
    return io.BytesIO(json.dumps({'model': 'recorded-model', 'usage': {'prompt_tokens': 100, 'completion_tokens': 20,
        'prompt_tokens_details': {'cached_tokens': 10}},
        'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]}).encode())


def test_attempt_cap_and_usage_are_recorded_without_key(tmp_path):
    provider = BoundedProvider('sensitive-dummy', tmp_path/'ledger.json', 1, opener=lambda *a, **k: response())
    assert provider.complete([{'role': 'user', 'content': 'hi'}], 'gpt-5-nano', 'generation')[0] == 'answer'
    with pytest.raises(ProviderFailure):
        provider.complete([], 'gpt-5-nano', 'generation')
    text = (tmp_path/'ledger.json').read_text()
    result = json.loads(text)
    assert 'sensitive-dummy' not in text
    assert len(result['records']) == 1
    assert result['records'][0]['usage']['completion_tokens'] == 20
    assert result['records'][0]['estimated_usd'] > 0


def test_failed_call_remains_reserved_and_has_no_automatic_retry(tmp_path):
    attempts = []
    def broken(*args, **kwargs):
        attempts.append(1)
        raise TimeoutError('private credentials must not be logged')
    provider = BoundedProvider('private-key', tmp_path/'ledger.json', 1, opener=broken)
    with pytest.raises(ProviderFailure):
        provider.complete([], 'gpt-5-nano', 'generation')
    assert len(attempts) == 1
    text = (tmp_path/'ledger.json').read_text()
    assert 'private credentials' not in text
    assert json.loads(text)['reserved_upper_bound_usd'] > 0


def test_spend_cap_blocks_before_call(tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail('No request should be made')
    provider = BoundedProvider('dummy', tmp_path/'ledger.json', 1, spend_limit_usd=0, opener=forbidden)
    with pytest.raises(ProviderFailure):
        provider.complete([], 'gpt-5-nano', 'generation')


def test_judge_never_receives_reference_answer(tmp_path):
    def inspect(req, **kwargs):
        assert b'private-reference' not in req.data
        assert b'generated-answer' in req.data
        return response('{"faithfulness": 0.9, "answer_relevancy": 0.95}')
    provider = BoundedProvider('dummy', tmp_path/'ledger.json', 1, opener=inspect)
    scores, _ = judge_answer(provider, {'question': 'q', 'retrieved_contexts': ['c'], 'reference_answer': 'private-reference'}, 'generated-answer')
    assert scores['faithfulness'] == 0.9


def test_malformed_judge_result_is_rejected(tmp_path):
    provider = BoundedProvider('dummy', tmp_path/'ledger.json', 1, opener=lambda *a, **k: response('{"faithfulness": 9}'))
    with pytest.raises(ProviderFailure):
        judge_answer(provider, {'question': 'q', 'retrieved_contexts': ['c']}, 'answer')
