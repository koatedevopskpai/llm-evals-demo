# llm-evals-demo — a CI quality bar for LLM apps

[![eval-gate](https://github.com/koate-kpai/llm-evals-demo/actions/workflows/eval-gate.yml/badge.svg)](https://github.com/koate-kpai/llm-evals-demo/actions/workflows/eval-gate.yml)

A compact, reproducible eval harness that puts a **quality bar** on an LLM/RAG pipeline: golden dataset, RAGAS faithfulness + answer relevance, LLM-as-judge, and a CI gate that **blocks deploys below a threshold** (default: faithfulness ≥ 0.90).

This is the eval layer I reference in my portfolio — the same discipline as the eval platform in `agentic-platform-aws`, packaged so you can see the gate actually run.

## What it does

```
golden_set.py  ──►  evaluator.py  ──►  run_evals.py  ──►  EVAL_REPORT.md
 (ground truth)       (RAGAS metrics)    (threshold gate)   (evidence for interview/blog)
```

- **Golden dataset** — 30 hand-labelled cases: question, reference answer, and the retrieved context that SHOULD be used.
- **RAGAS metrics** — `faithfulness` (is the answer grounded in context?) and `answer_relevancy`, using an OpenAI-compatible endpoint (OpenRouter works — self-host/open-weights friendly).
- **LLM-as-judge** — a judge model scores open-ended quality on a rubric when you enable it.
- **CI gate** — a GitHub Actions workflow runs the suite on every push/PR and fails the build if the bar is missed.
- **Mock mode** — runs without any API key, so CI validates the pipeline for free and offline dev is instant.

## Quick start

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 1. Mock run (no API key) — validates the pipeline + gate
python run_evals.py --mode mock --threshold 0.90

# 2. Real run (OpenRouter/OpenAI-compatible)
#    set OPENAI_API_KEY (or OPENROUTER key + OPENAI_BASE_URL=https://openrouter.ai/api/v1)
python run_evals.py --threshold 0.90
```

A gate failure exits non-zero and prints a regression report:

```bash
EVAL GATE BLOCKED: faithfulness 0.84 (bar 0.90) — fix retrieval before shipping.
```

## The three questions I score

1. **Faithfulness** — is the answer supported by the provided context? (kills hallucination)
2. **Answer relevance** — does it answer the user's question? (kills noise)
3. **Judge quality** — tone/completeness (optional, on a rubric)

## CI

`.github/workflows/eval-gate.yml` runs mock-mode on every push/PR (zero cost) and real-mode whenever `OPENAI_API_KEY` is present in secrets. A PR that regresses faithfulness fails its check — evals are a deploy gate, not a vibe check.

## Tests

```bash
pytest tests/ -q      # mock-mode gate behaviour, no API needed
```

## Notes for reviewers

- `evals/golden.json` is the human-curated ground truth — the thing that makes the bar honest.
- Swap `OPENAI_BASE_URL` for any OpenAI-compatible provider (OpenRouter, vLLM, Ollama's OpenAI layer) — provider-agnostic by design.

Built by koate-kpai as part of an AI engineering portfolio. See the eval-platform design in `agentic-platform-aws`.
