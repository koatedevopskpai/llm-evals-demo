"""run_evals.py — CLI entry point: score the golden set, apply the quality bar, write EVAL_REPORT.md.

Usage:
  python run_evals.py --mode mock --threshold 0.90
  python run_evals.py --mode real --threshold 0.90
Exit code: 0 = gate passed, 1 = gate blocked.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from evaluator import Evaluator
from golden_set import GoldenSet


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the LLM eval gate over the golden set.")
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--threshold", type=float, default=0.90, help="min faithfulness to pass")
    ap.add_argument("--relevancy-bar", type=float, default=0.85)
    ap.add_argument("--report", default="EVAL_REPORT.md")
    args = ap.parse_args()

    golden = GoldenSet()
    evaluator = Evaluator(mode=args.mode)
    rows: list[dict] = []
    for i, case in enumerate(golden.cases, start=1):
        scores = evaluator.score(case)
        rows.append({"case": i, "question": case.question, **scores})

    faithfulness = [r["faithfulness"] for r in rows]
    relevancy = [r["answer_relevancy"] for r in rows]
    mean_f = statistics.fmean(faithfulness)
    mean_r = statistics.fmean(relevancy)
    worst = min(rows, key=lambda r: r["faithfulness"])

    report = Path(args.report)
    report.write_text(
        f"""# EVAL REPORT — {args.mode} mode

Date: (generated)
Golden set: {len(golden)} cases
Quality bar: faithfulness >= {args.threshold}  |  relevancy >= {args.relevancy_bar}

| Metric | Mean | Worst case |
|---|---|---|
| Faithfulness | {mean_f:.3f} | case {worst['case']}: {worst['faithfulness']:.3f} |
| Answer relevancy | {mean_r:.3f} | — |

## Detail
```json
{json.dumps(rows, indent=2, default=str)}
```
""",
        encoding="utf-8",
    )

    ok = mean_f >= args.threshold and mean_r >= args.relevancy_bar
    if ok:
        print(f"EVAL GATE PASSED: faithfulness {mean_f:.3f} (bar {args.threshold}) | "
              f"relevancy {mean_r:.3f} (bar {args.relevancy_bar})")
        print(f"report -> {report}")
        return 0

    print(f"EVAL GATE BLOCKED: faithfulness {mean_f:.3f} (bar {args.threshold}) | "
          f"relevancy {mean_r:.3f} (bar {args.relevancy_bar})")
    print(f"worst case #{worst['case']}: {worst['question'][:60]}...")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
