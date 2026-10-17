"""Small explicit OpenAI transport for smoke checks, with no automatic retries.

This is a custom rubric transport, not the RAGAS scorer. Usage is priced from
the dated supplied rate card; estimates are not provider invoices.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib import error, request

from gate_policy import finite_score

RATES = {
    'gpt-5-nano': {'input': 0.05, 'cached_input': 0.005, 'output': 0.40},
    'gpt-5.4-mini': {'input': 0.75, 'cached_input': 0.075, 'output': 4.50},
}
PRICING = {'currency': 'USD', 'date_checked': '2026-10-17', 'per': '1000000 tokens',
           'sources': ['https://developers.openai.com/api/docs/models/gpt-5-nano',
                       'https://developers.openai.com/api/docs/models/gpt-5.4-mini']}


class ProviderFailure(RuntimeError):
    pass


class BoundedProvider:
    def __init__(self, api_key, ledger, max_calls, spend_limit_usd=0.25, opener=None):
        if not api_key:
            raise ValueError('API key required')
        self._key = api_key
        self.ledger = Path(ledger)
        if self.ledger.exists():
            raise ValueError('usage ledger already exists; choose a new run path')
        self.max_calls = max_calls
        self.limit = spend_limit_usd
        self._open = opener or request.urlopen
        self.records = []

    def _save(self):
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self.ledger.write_text(json.dumps({'pricing': PRICING, 'rates': RATES,
            'spend_limit_usd': self.limit, 'max_calls': self.max_calls,
            'reserved_upper_bound_usd': sum(r['reserved_usd'] for r in self.records),
            'estimated_known_usage_usd': sum(r.get('estimated_usd', 0) for r in self.records),
            'records': self.records}, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    def complete(self, messages, model, purpose, json_output=False):
        if model not in RATES:
            raise ValueError('model missing from dated rate card')
        payload = {'model': model, 'messages': messages, 'max_completion_tokens': 1024,
                   'reasoning_effort': 'minimal' if model == 'gpt-5-nano' else 'none'}
        if json_output:
            payload['response_format'] = {'type': 'json_object'}
        body = json.dumps(payload).encode('utf-8')
        # One byte per input token is a conservative allowance for this short
        # text-only request, plus 256 tokens of protocol overhead.
        if len(body) > 6000:
            raise ValueError('request exceeds smoke-test input allowance')
        rate = RATES[model]
        reserved = ((len(body) + 256) * rate['input'] + 1024 * rate['output']) / 1_000_000
        if len(self.records) >= self.max_calls or sum(r['reserved_usd'] for r in self.records) + reserved > self.limit:
            raise ProviderFailure('call or spend allowance exhausted before request')
        rec = {'index': len(self.records) + 1, 'requested_model': model, 'purpose': purpose,
               'created_at_utc': datetime.now(timezone.utc).isoformat(), 'reserved_usd': reserved,
               'status': 'attempting', 'attempt': 1, 'automatic_retries': 0,
               'request_sha256': hashlib.sha256(body).hexdigest()}
        self.records.append(rec)
        self._save()  # reserve before the call; an interrupted request stays reserved
        started = time.perf_counter()
        req = request.Request('https://api.openai.com/v1/chat/completions', body,
                              {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self._key})
        try:
            with self._open(req, timeout=45) as response:
                result = json.load(response)
            usage = result.get('usage')
            if not isinstance(usage, dict):
                raise ProviderFailure('response has no usage record')
            input_tokens = usage.get('prompt_tokens')
            output_tokens = usage.get('completion_tokens')
            cached = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
            if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in [input_tokens, output_tokens, cached]) or cached > input_tokens:
                raise ProviderFailure('response has invalid usage')
            rec.update(status='received', actual_model=result.get('model'), usage=usage,
                       estimated_usd=((input_tokens-cached)*rate['input'] + cached*rate['cached_input'] + output_tokens*rate['output'])/1_000_000)
            choice = result['choices'][0]
            if choice.get('finish_reason') != 'stop' or not isinstance(choice['message'].get('content'), str) or not choice['message']['content'].strip():
                raise ProviderFailure('incomplete or empty completion')
            return choice['message']['content'], rec
        except Exception as exc:
            # Never log raw provider bodies, authorization headers or keys.
            rec.update(status='failed', error_type=type(exc).__name__)
            if isinstance(exc, error.HTTPError):
                rec['http_status'] = exc.code
            raise ProviderFailure('provider request failed; inspect sanitised ledger') from None
        finally:
            rec['latency_seconds'] = time.perf_counter() - started
            self._save()


def judge_answer(provider, case, answer):
    # No reference-answer field is sent to the judge.
    text, receipt = provider.complete([
        {'role': 'system', 'content': 'Assess the supplied generated answer against only the question and contexts. Return JSON with faithfulness and answer_relevancy numbers from 0 to 1. Unsupported facts or contradictions must lower faithfulness. Give no explanation.'},
        {'role': 'user', 'content': json.dumps({'question': case['question'], 'contexts': case['retrieved_contexts'], 'generated_answer': answer})},
    ], 'gpt-5.4-mini', 'judge', json_output=True)
    result = json.loads(text)
    if not isinstance(result, dict) or any(not finite_score(result.get(m)) for m in ('faithfulness', 'answer_relevancy')):
        raise ProviderFailure('judge returned malformed scores')
    return {m: result[m] for m in ('faithfulness', 'answer_relevancy')}, receipt
