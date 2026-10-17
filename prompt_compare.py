"""P01: paired prompt comparison; reuses E04's gate and E05's bounded transport.

Thirty authored development cases are not a held-out benchmark. All failures and
null results remain in the report. No provider/model quality claim follows.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess

from bounded_provider import BoundedProvider, PRICING, RATES, judge_answer
from compare_smoke import summary
from gate_policy import assess_rows, load_policy

ROOT = Path(__file__).resolve().parent
MODEL = 'gpt-5.4-mini'
VARIANTS = ['baseline-v1', 'constrained-v2']
ABSTENTION = re.compile(
    r"not (?:stated|specified|provided|available|mentioned|in |supported)|"
    r"does not|do not|cannot|can't|don't|doesn't|unknown|no information|insufficient|not contain",
    re.I,
)


def load_cases(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    cases = data['cases']
    if data['scope'] != 'self-authored-development' or not 30 <= len(cases) <= 50:
        raise ValueError('Expected 30-50 explicitly authored development cases')
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case IDs')
    for case in cases:
        if not case['question'] or not case['retrieved_contexts']:
            raise ValueError('Question and context required')
        if type(case['answerable']) is not bool:
            raise ValueError('Explicit answerability required')
        if any(not isinstance(context, str) or not context for context in case['retrieved_contexts']):
            raise ValueError('Nonempty context text required')
    return cases


def classify(case, row, policy):
    categories = []
    categories.extend('quality_gate:' + fail['reason'] for fail in row['outcome']['hard_failures'])
    if row.get('provider_error'):
        categories.append(row['provider_error_stage'] + '_provider_or_parse_failure')
    if row['faithfulness'] < policy['mean_floors']['faithfulness']:
        categories.append('judge_faithfulness_below_floor')
    if row['answer_relevancy'] < policy['mean_floors']['answer_relevancy']:
        categories.append('judge_relevancy_below_floor')
    if case['answerable'] and not re.search(r'\[[1-9][0-9]*\]', row['generated_answer']):
        categories.append('citation_marker_missing')
    if not case['answerable'] and not ABSTENTION.search(row['generated_answer']):
        categories.append('expected_abstention_phrase_missing')
    return categories


def summarise(rows):
    result = summary(rows)  # Same failure-inclusive timing/cost population as E05.
    result['failure_categories'] = dict(Counter(c for r in rows for c in r['failure_categories']))
    for purpose in ('generation', 'judge'):
        for token in ('input', 'output'):
            result[f'{purpose}_{token}_tokens'] = sum(r[f'{purpose}_{token}_tokens'] for r in rows)
    return result


def compare(provider, output, cases_path=ROOT/'evals/P01-cases.json'):
    output = Path(output)
    if output.exists():
        raise ValueError('Comparison output already exists; choose a fresh run path')
    cases = load_cases(cases_path)
    prompts = {name: (ROOT/f'prompts/P01-{name}.txt').read_text(encoding='utf-8') for name in VARIANTS}
    policy = load_policy()
    grouped = {name: [] for name in VARIANTS}
    report = {
        'scope': 'paired self-authored development comparison, not a benchmark',
        'execution_mode': 'live_openai_generation_and_custom_json_judge',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'candidate_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'python_version': platform.python_version(), 'model': MODEL, 'judge_model': MODEL,
        'reasoning_effort': 'none', 'max_completion_tokens': 1024,
        'pricing': PRICING, 'rates': RATES, 'sample_size_per_variant': len(cases),
        'case_sha256': hashlib.sha256(Path(cases_path).read_bytes()).hexdigest(),
        'prompts': prompts,
        'prompt_sha256': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in prompts.items()},
        'policy': policy,
        'policy_sha256': hashlib.sha256((ROOT/'evals/gate-policy.json').read_bytes()).hexdigest(),
        'run_complete': False,
        'limitations': [
            'Thirty authored cases per variant, one completion each; no repeated trials or significance claim.',
            'Both prompts use the same model, fixed questions and fixed retrieval contexts; retrieval is not evaluated.',
            'Same-family custom JSON judge, not RAGAS/human review; bias and semantic scoring errors remain possible.',
            'Citation/abstention categories are lexical contract heuristics, not semantic proof.',
            'p95 interpolated from 30 observations is descriptive, not an SLO or stable tail estimate.',
            'Dated usage estimates, not invoices; unknown usage remains reserved. No automatic retries.',
            'OpenAI prompts only: this does not establish Claude model or Claude Code performance.',
            'B01 remains deferred; do not tune prompts against this report and call the same set held-out.',
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        report['results'] = {
            name: {'rows': rows, 'summary': summarise(rows) if rows else None,
                   'aggregate_quality_gate': assess_rows(rows, policy, len(cases)),
                   'all_contracts_passed': len(rows) == len(cases) and all(r['passed'] for r in rows)}
            for name, rows in grouped.items()
        }
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    save()
    for index, case in enumerate(cases):
        # Fixed alternation limits a simple all-A-then-all-B ordering effect.
        order = VARIANTS if index % 2 == 0 else list(reversed(VARIANTS))
        for name in order:
            row = {'case_id': case['id'], 'category': case['category'],
                   'question': case['question'], 'retrieved_contexts': case['retrieved_contexts'],
                   'generated_answer': '', 'faithfulness': 0.0, 'answer_relevancy': 0.0}
            start = len(provider.records)
            stage = 'generation'
            try:
                context = '\n'.join(f'[{i}] {text}' for i, text in enumerate(case['retrieved_contexts'], 1))
                answer, _ = provider.complete([
                    {'role': 'system', 'content': prompts[name]},
                    {'role': 'user', 'content': json.dumps({'question': case['question'], 'context': context})},
                ], MODEL, 'generation')
                row['generated_answer'] = answer
                stage = 'judge'
                # E05 judge sends only question, contexts, generated answer. No gold answer/variant label.
                row.update(judge_answer(provider, case, answer)[0])
            except Exception as exc:
                row.update(provider_error=type(exc).__name__, provider_error_stage=stage)
            attempts = provider.records[start:]
            for purpose in ('generation', 'judge'):
                receipt = [r for r in attempts if r['purpose'] == purpose]
                row[purpose+'_seconds'] = sum(r['latency_seconds'] for r in receipt)
                row[purpose+'_estimated_usd'] = sum(r.get('estimated_usd', 0) for r in receipt)
                row[purpose+'_input_tokens'] = sum(r.get('usage', {}).get('prompt_tokens', 0) for r in receipt)
                row[purpose+'_output_tokens'] = sum(r.get('usage', {}).get('completion_tokens', 0) for r in receipt)
            row['end_to_end_seconds'] = row['generation_seconds'] + row['judge_seconds']
            row['input_tokens'] = row['generation_input_tokens'] + row['judge_input_tokens']
            row['output_tokens'] = row['generation_output_tokens'] + row['judge_output_tokens']
            row['attempts_with_unknown_usage'] = sum('usage' not in r for r in attempts)
            row['attempt_indexes'] = [r['index'] for r in attempts]
            row['outcome'] = assess_rows([row], policy, 1)
            row['failure_categories'] = classify(case, row, policy)
            row['passed'] = row['outcome']['passed'] and not row['failure_categories']
            grouped[name].append(row)
            save()
        print(f'Paired cases completed: {index+1}/{len(cases)}', flush=True)
    first, second = [summarise(grouped[name]) for name in VARIANTS]
    pairs = list(zip(grouped[VARIANTS[0]], grouped[VARIANTS[1]], strict=True))
    assert all(a['case_id'] == b['case_id'] for a,b in pairs)
    report['paired_outcomes'] = {
        'v2_pass_v1_fail': sum(b['passed'] and not a['passed'] for a,b in pairs),
        'v1_pass_v2_fail': sum(a['passed'] and not b['passed'] for a,b in pairs),
        'both_pass': sum(a['passed'] and b['passed'] for a,b in pairs),
        'both_fail': sum(not a['passed'] and not b['passed'] for a,b in pairs),
    }
    report['v2_minus_v1'] = {key: second[key]-first[key] for key in (
        'mean_faithfulness', 'mean_answer_relevancy', 'failure_count',
        'p50_generation_seconds', 'p95_generation_seconds', 'generation_estimated_usd')}
    report['run_complete'] = True
    report['provider_failure_count'] = sum(
        bool(row.get('provider_error')) for rows in grouped.values() for row in rows
    )
    report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    save()
    return report


def reassess_saved(original_path, output):
    """Re-evaluate frozen live outputs, without generating or judging again."""
    original_path, output = Path(original_path), Path(output)
    if output.exists():
        raise ValueError('Reassessment output exists; never overwrite original evidence')
    report = json.loads(original_path.read_text(encoding='utf-8'))
    cases_path = ROOT/'evals/P01-cases.json'
    cases = load_cases(cases_path)
    policy = load_policy()
    if not report['run_complete'] or report['case_sha256'] != hashlib.sha256(cases_path.read_bytes()).hexdigest():
        raise ValueError('Incomplete run or changed cases')
    if report['policy_sha256'] != hashlib.sha256((ROOT/'evals/gate-policy.json').read_bytes()).hexdigest():
        raise ValueError('Policy changed since live run')
    raw_fields = ['generated_answer', 'faithfulness', 'answer_relevancy',
                  'generation_seconds', 'judge_seconds', 'generation_estimated_usd',
                  'judge_estimated_usd', 'input_tokens', 'output_tokens', 'attempt_indexes']
    raw_before = []
    raw_after = []
    grouped = {}
    for name in VARIANTS:
        prompt = (ROOT/f'prompts/P01-{name}.txt').read_text(encoding='utf-8')
        if report['prompt_sha256'][name] != hashlib.sha256(prompt.encode()).hexdigest():
            raise ValueError('Prompt changed since live run')
        rows = report['results'][name]['rows']
        if [row['case_id'] for row in rows] != [case['id'] for case in cases]:
            raise ValueError('Saved cases changed or incomplete')
        for case, row in zip(cases, rows, strict=True):
            raw_before.append({key: row[key] for key in raw_fields})
            row.update(question=case['question'], retrieved_contexts=case['retrieved_contexts'])
            row['outcome'] = assess_rows([row], policy, 1)
            row['failure_categories'] = classify(case, row, policy)
            row['passed'] = row['outcome']['passed'] and not row['failure_categories']
            raw_after.append({key: row[key] for key in raw_fields})
        grouped[name] = rows
        report['results'][name] = {
            'rows': rows, 'summary': summarise(rows),
            'aggregate_quality_gate': assess_rows(rows, policy, len(cases)),
            'all_contracts_passed': all(row['passed'] for row in rows),
        }
    assert raw_before == raw_after
    pairs = list(zip(*(grouped[name] for name in VARIANTS), strict=True))
    report['paired_outcomes'] = {
        'v2_pass_v1_fail': sum(b['passed'] and not a['passed'] for a,b in pairs),
        'v1_pass_v2_fail': sum(a['passed'] and not b['passed'] for a,b in pairs),
        'both_pass': sum(a['passed'] and b['passed'] for a,b in pairs),
        'both_fail': sum(not a['passed'] and not b['passed'] for a,b in pairs),
    }
    first, second = [report['results'][name]['summary'] for name in VARIANTS]
    report['v2_minus_v1'] = {key: second[key]-first[key] for key in report['v2_minus_v1']}
    report.update(
        original_execution_mode=report['execution_mode'],
        execution_mode='offline_reassessment_of_saved_live_outputs',
        reassessed_at_utc=datetime.now(timezone.utc).isoformat(),
        reassessment_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        original_report_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
        unchanged_live_fields_sha256=hashlib.sha256(json.dumps(raw_after,sort_keys=True).encode()).hexdigest(),
        reassessment_provider_calls=0,
        reassessment_reason='Restore omitted question/context metadata before unchanged E04 gate; no prompt, case, threshold, answer or judge-score changes.',
    )
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='artifacts/P01/report.json')
    ap.add_argument('--reassess', help='Reassess a saved live report offline, without a provider key')
    args = ap.parse_args()
    if args.reassess:
        result = reassess_saved(args.reassess, args.output)
    else:
        provider = BoundedProvider(os.getenv('OPENAI_API_KEY'), Path(args.output).with_name('usage.json'),
                                   max_calls=120, spend_limit_usd=1.0)
        result = compare(provider, args.output)
    print('Paired comparison complete:', result['paired_outcomes'])
    raise SystemExit(1 if result['provider_failure_count'] else 0)
