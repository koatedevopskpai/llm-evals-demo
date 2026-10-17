"""Compare two configurations on three fixed cases; explicitly not a benchmark."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess

from bounded_provider import BoundedProvider, PRICING, judge_answer
from calibration_smoke import CASES
from gate_policy import assess_rows, load_policy

CONFIGURATIONS = [
    {'id': 'nano-grounded', 'model': 'gpt-5-nano'},
    {'id': 'mini-grounded', 'model': 'gpt-5.4-mini'},
]
PROMPT = 'Answer using only the supplied context, in one sentence. Cite the numbered context as [1]. If unsupported, say you do not know. Do not invent dates, amounts or conditions.'


def percentile(values, fraction):
    if not values or not 0 <= fraction <= 1:
        raise ValueError('non-empty values and fraction in [0,1] required')
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def summary(rows):
    # Failed attempts remain in the population; do not report success-only latency.
    timings = [r['end_to_end_seconds'] for r in rows]
    return {'case_count': len(rows), 'failure_count': sum(not r['passed'] for r in rows),
            'mean_faithfulness': statistics.fmean(r['faithfulness'] for r in rows),
            'mean_answer_relevancy': statistics.fmean(r['answer_relevancy'] for r in rows),
            'p50_end_to_end_seconds': percentile(timings, .50),
            'p95_end_to_end_seconds': percentile(timings, .95),
            'p50_generation_seconds': percentile([r['generation_seconds'] for r in rows], .50),
            'p95_generation_seconds': percentile([r['generation_seconds'] for r in rows], .95),
            'generation_estimated_usd': sum(r['generation_estimated_usd'] for r in rows),
            'judge_estimated_usd': sum(r['judge_estimated_usd'] for r in rows),
            'retry_estimated_usd': 0.0,
            'automatic_retry_count': 0,
            'input_tokens': sum(r['input_tokens'] for r in rows),
            'output_tokens': sum(r['output_tokens'] for r in rows),
            'attempts_with_unknown_usage': sum(r['attempts_with_unknown_usage'] for r in rows)}


def compare(provider, output):
    policy = load_policy()
    grouped = {c['id']: [] for c in CONFIGURATIONS}
    # Alternate configurations per question to reduce simple ordering bias.
    for index, case in enumerate(CASES):
        configurations = CONFIGURATIONS if index % 2 == 0 else list(reversed(CONFIGURATIONS))
        for configuration in configurations:
            row = {'case_id': case['id'], 'question': case['question'],
                   'retrieved_contexts': case['retrieved_contexts'], 'generated_answer': '',
                   'faithfulness': 0.0, 'answer_relevancy': 0.0}
            initial_attempts = len(provider.records)
            try:
                answer, _ = provider.complete([
                    {'role': 'system', 'content': PROMPT},
                    {'role': 'user', 'content': json.dumps({'question': case['question'], 'context': '[1] ' + case['retrieved_contexts'][0]})},
                ], configuration['model'], 'generation')
                row['generated_answer'] = answer
                row.update(judge_answer(provider, case, answer)[0])
            except Exception as exc:
                row['judge_error'] = type(exc).__name__
            receipt = provider.records[initial_attempts:]
            row['generation_seconds'] = sum(r['latency_seconds'] for r in receipt if r['purpose'] == 'generation')
            row['judge_seconds'] = sum(r['latency_seconds'] for r in receipt if r['purpose'] == 'judge')
            row['end_to_end_seconds'] = row['generation_seconds'] + row['judge_seconds']
            row['generation_estimated_usd'] = sum(r.get('estimated_usd', 0) for r in receipt if r['purpose'] == 'generation')
            row['judge_estimated_usd'] = sum(r.get('estimated_usd', 0) for r in receipt if r['purpose'] == 'judge')
            row['input_tokens'] = sum(r.get('usage', {}).get('prompt_tokens', 0) for r in receipt)
            row['output_tokens'] = sum(r.get('usage', {}).get('completion_tokens', 0) for r in receipt)
            row['attempts_with_unknown_usage'] = sum('usage' not in r for r in receipt)
            row['outcome'] = assess_rows([row], policy, 1)
            row['passed'] = row['outcome']['passed']
            grouped[configuration['id']].append(row)
    results = {name: {'summary': summary(rows), 'rows': rows,
                      'aggregate_gate': assess_rows(rows, policy, len(CASES))} for name, rows in grouped.items()}
    accepted = [name for name, value in results.items() if value['aggregate_gate']['passed']]
    preference = min(accepted, key=lambda name: results[name]['summary']['generation_estimated_usd']) if accepted else None
    report = {'scope': 'three-question paired development smoke comparison; not a benchmark',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'candidate_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'python_version': platform.python_version(), 'pricing': PRICING,
              'configurations': CONFIGURATIONS, 'generation_prompt': PROMPT,
              'fixture_sha256': hashlib.sha256(json.dumps(CASES, sort_keys=True).encode()).hexdigest(),
              'policy': policy, 'results': results, 'provisional_lower_generation_cost_option': preference,
              'selection_rule': 'Among configurations passing this tiny fixture, prefer the lower measured generation cost. No claim of statistical superiority or production suitability.',
              'limitations': ['n=3 paired questions per configuration; interpolated p95 is descriptive, not an SLO or stable tail estimate.',
                              'Custom gpt-5.4-mini JSON judge; self-preference bias and unreviewed semantic errors remain possible.',
                              'No automatic retries. Unknown usage on failed calls is reported; reserved allowance remains in ledger.',
                              'Dated usage-based cost estimates, not provider invoices. B01 held-out benchmark remains deferred.']}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('Comparison written. Per-configuration failures:', {k:v['summary']['failure_count'] for k,v in results.items()})
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='artifacts/E05-live.json')
    args = ap.parse_args()
    provider = BoundedProvider(os.getenv('OPENAI_API_KEY'), Path(args.output).with_name('E05-usage.json'), max_calls=12)
    report = compare(provider, args.output)
    raise SystemExit(0 if report['provisional_lower_generation_cost_option'] else 1)
