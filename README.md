# llm-evals-demo — offline evaluation scaffold

A small reference harness for loading golden cases, exercising deterministic
placeholder scores and checking aggregate thresholds. The mock run is a software
wiring check. It does not measure the quality of generated application answers.

## Implemented scope

- `golden_set.py` loads 30 supplied question/reference/context cases.
- `evaluator.py` offers deterministic mock scores and contains a legacy RAGAS
  path whose results do not establish valid application-quality evidence.
- `run_evals.py` checks mean faithfulness >= 0.90 and answer relevancy >= 0.85
  by default. These are configured demonstration thresholds, not validated
  acceptance criteria for a particular application.
- Tests exercise the mock scores and golden-set loading.

## Known measurement limitation

The legacy real-mode implementation sends `case.reference_answer` as the
answer to RAGAS. It therefore does not evaluate freshly generated application
answers. E01 must separate reference and generated answers before real-mode
results can support an application-quality claim. A separate custom rubric
judge is not implemented in this harness.

The workflow's ordinary mock job can pass without a live evaluation. Its green
status is not evidence that a deployment was quality-gated. This repository does
not establish a deployment integration, production reliability or commercial use.

## Offline quick start

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install pytest==8.4.2
.venv\Scripts\python golden_set.py
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python run_evals.py --mode mock --report mock-report.md
```

Installation needs a package source unless dependencies are cached. The listed
test/mock commands require no provider key or model download. Mock scores are
placeholders, regardless of how high they are.

## Planned improvements

Keep mock and live reports distinct (E02), evaluate actual application answers
(E01), version the benchmark and provenance (E03), then calibrate thresholds
against known failures (E04). No new live benchmark has been run for this
documentation correction. Paid execution remains disabled under the current
portfolio budget.

Authored by Koate Kpai: https://github.com/koatedevopskpai
