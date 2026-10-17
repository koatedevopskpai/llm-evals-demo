"""Offline scaffold CLI. A mock pass can never satisfy live release quality."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics

from evaluator import Evaluator
from golden_set import GoldenSet

def _bar(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("bar must be finite and between 0 and 1")
    return value

def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Evaluation evidence: " + payload["status"] + "\n\n"
        "Execution mode: " + payload["execution_mode"] + "\n\n"
        "Evaluator: " + payload["evaluator_identity"] + "\n\n"
        "Live quality established: false\n\n"
        "This artifact cannot authorise a live-quality release.\n\n"
        "```json\n" + json.dumps(payload, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--threshold", type=_bar, default=0.90)
    ap.add_argument("--relevancy-bar", type=_bar, default=0.85)
    ap.add_argument("--report", default="EVAL_REPORT.mock.md")
    ap.add_argument("--require-live-quality", action="store_true",
                    help="Block unless genuine application-quality evidence is available")
    args = ap.parse_args(argv)
    path = Path(args.report)
    if path.suffix.lower() != ".md":
        ap.error("--report must end in .md; a matching .json artifact is also written")
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_mode": args.mode,
        "evaluator_identity": "deterministic-placeholder-v1" if args.mode == "mock" else "unavailable-E01",
        "evidence_type": "offline_scaffold" if args.mode == "mock" else "unavailable",
        "live_quality_established": False,
        "live_quality_required": args.require_live_quality,
        "thresholds": {"faithfulness": args.threshold, "answer_relevancy": args.relevancy_bar},
        "rows": [],
    }
    # Check before evaluator construction, SDK loading, scoring or credential use.
    if args.require_live_quality or args.mode == "real":
        payload.update(status="blocked", reason="E01 generated-answer evaluation is not implemented")
        write_report(path, payload)
        print("LIVE QUALITY BLOCKED: E01 is required; no provider calls were made.")
        return 2

    golden = GoldenSet()
    evaluator = Evaluator(mode="mock")
    rows = [{"case": i, "question": c.question, **evaluator.score(c)}
            for i, c in enumerate(golden.cases, start=1)]
    if not rows:
        payload.update(status="blocked", reason="No golden cases")
        write_report(path, payload)
        return 2
    mean_f = statistics.fmean(r["faithfulness"] for r in rows)
    mean_r = statistics.fmean(r["answer_relevancy"] for r in rows)
    ok = mean_f >= args.threshold and mean_r >= args.relevancy_bar
    payload.update(status="offline_pass" if ok else "offline_fail", case_count=len(rows),
                   means={"faithfulness": mean_f, "answer_relevancy": mean_r}, rows=rows)
    write_report(path, payload)
    print(f"OFFLINE SCAFFOLD {'PASSED' if ok else 'FAILED'}: placeholder scores only; live quality unverified.")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
