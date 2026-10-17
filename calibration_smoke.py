"""Author-labelled development smoke checks, not the deferred benchmark."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from bounded_provider import BoundedProvider, judge_answer
from gate_policy import assess_rows, json_safe, load_policy

CASES = [
    {'id': 'payment', 'question': 'How many days are allowed for payment?', 'retrieved_contexts': ['Payment is due within 30 days.'], 'good': 'Payment is due within 30 days. [1]', 'bad': 'Payment is due within 90 days. [1]'},
    {'id': 'notice', 'question': 'How much notice is needed to terminate?', 'retrieved_contexts': ['Termination requires 14 days of written notice.'], 'good': 'Termination requires 14 days of written notice. [1]', 'bad': 'No notice is required to terminate. [1]'},
    {'id': 'currency', 'question': 'What is the annual subscription fee?', 'retrieved_contexts': ['The annual subscription fee is USD 120.'], 'good': 'The annual fee is USD 120. [1]', 'bad': 'The annual fee is USD 1200. [1]'},
]


def run(output, live=False, provider=None):
    policy = load_policy()
    samples = []
    for case, scores in zip(CASES, policy['baseline_fixture_scores']):
        for kind in ('good', 'bad'):
            row = {'question': case['question'], 'retrieved_contexts': case['retrieved_contexts'],
                   'generated_answer': case[kind], 'label': kind, 'case_id': case['id']}
            if live:
                try:
                    row.update(judge_answer(provider, case, case[kind])[0])
                except Exception as exc:
                    row.update(faithfulness=0.0, answer_relevancy=0.0, judge_error=type(exc).__name__)
            else:
                row.update(scores if kind == 'good' else {'faithfulness': 0.0, 'answer_relevancy': 0.0})
            samples.append(row)
    good = [r for r in samples if r['label'] == 'good']
    bad = [r for r in samples if r['label'] == 'bad']
    baseline = assess_rows(good, policy, 3)
    regression = assess_rows(bad, policy, 3)
    disagreement = [{'case_id': r['case_id'], 'author_label': r['label'],
                     'gate_passed': assess_rows([r], policy, 1)['passed']}
                    for r in samples if assess_rows([r], policy, 1)['passed'] != (r['label'] == 'good')]
    report = {'scope': 'six author-labelled development outputs, three questions; not a benchmark',
              'execution_mode': 'live_custom_rubric' if live else 'offline_contract_fixture',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'candidate_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'fixture_sha256': hashlib.sha256(json.dumps(CASES, sort_keys=True).encode()).hexdigest(),
              'policy': policy, 'baseline': baseline, 'known_regression': regression,
              'judge_author_disagreements': disagreement, 'rows': samples,
              'acceptance_met': baseline['passed'] and not regression['passed'] and not disagreement,
              'limitations': ['Labels authored for simple cases, not independent expert adjudication.',
                              'Custom JSON rubric; not RAGAS metrics and not comparable with prior RAGAS numbers.',
                              'Thresholds provisional; B01 held-out validation remains deferred.']}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('Development smoke acceptance:', report['acceptance_met'], '; mode:', report['execution_mode'])
    return 0 if report['acceptance_met'] else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', default='artifacts/E04-offline.json')
    ap.add_argument('--live', action='store_true')
    args = ap.parse_args()
    provider = BoundedProvider(os.getenv('OPENAI_API_KEY'), Path(args.output).with_name('E04-usage.json'), max_calls=6) if args.live else None
    raise SystemExit(run(args.output, args.live, provider))
