"""Fail-closed checks over application outputs and evaluator results."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import statistics

METRICS = ("faithfulness", "answer_relevancy")
DEFAULT_POLICY = Path(__file__).parent / "evals/gate-policy.json"


def finite_score(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def load_policy(path=DEFAULT_POLICY):
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(policy.get("policy_id"), str) or not policy["policy_id"]:
        raise ValueError("policy_id is required")
    for metric in METRICS:
        if not finite_score(policy.get("mean_floors", {}).get(metric)):
            raise ValueError(f"invalid mean floor: {metric}")
        if not finite_score(policy.get("per_case_floors", {}).get(metric)):
            raise ValueError(f"invalid per-case floor: {metric}")
    return policy


def assess_rows(rows, policy, expected_count):
    failures = []
    totals = {metric: [] for metric in METRICS}
    if not rows or len(rows) != expected_count or expected_count <= 0:
        failures.append({"case": None, "reason": "missing_or_unexpected_cases"})
    for index, row in enumerate(rows, 1):
        reasons = []
        if not isinstance(row, dict):
            failures.append({"case": index, "reason": "malformed_row"})
            for values in totals.values():
                values.append(0.0)
            continue
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            reasons.append("missing_question")
        answer = row.get("generated_answer")
        if not isinstance(answer, str) or not answer.strip():
            reasons.append("missing_generated_answer")
        contexts = row.get("retrieved_contexts")
        if not isinstance(contexts, list) or not contexts or any(not isinstance(c, str) or not c.strip() for c in contexts):
            reasons.append("missing_context")
        if row.get("judge_error"):
            reasons.append("judge_error")
        if isinstance(answer, str):
            # Existence only, not citation entailment. Numeric references map to
            # E01's numbered context blocks; named citations need explicit IDs.
            cited = re.findall(r"\[(\d+)\]|\[source\s+([^\]]+)\]", answer)
            context_count = len(contexts) if isinstance(contexts, list) else 0
            valid = {str(i) for i in range(1, context_count + 1)}
            sources = row.get("source_ids", [])
            if not isinstance(sources, list):
                sources = []
                reasons.append("malformed_source_ids")
            valid.update(str(x) for x in sources)
            if any((number or named.strip()) not in valid for number, named in cited):
                reasons.append("unknown_citation")
        for metric in METRICS:
            value = row.get(metric)
            if not finite_score(value):
                reasons.append(f"invalid_metric:{metric}")
                totals[metric].append(0.0)
            else:
                totals[metric].append(float(value))
                if value < policy["per_case_floors"][metric]:
                    reasons.append(f"case_below_floor:{metric}")
        failures.extend({"case": index, "reason": reason} for reason in reasons)
    means = {metric: statistics.fmean(values) if values else 0.0 for metric, values in totals.items()}
    for metric, value in means.items():
        if value < policy["mean_floors"][metric]:
            failures.append({"case": None, "reason": f"mean_below_floor:{metric}"})
    return {"passed": not failures, "policy_id": policy["policy_id"], "means": means,
            "hard_failures": failures, "case_count": len(rows), "expected_count": expected_count}


def json_safe(value):
    """Preserve rejected metric evidence without emitting non-standard JSON NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value
