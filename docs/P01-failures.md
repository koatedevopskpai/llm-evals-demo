# P01 failures and evaluator disagreement

These are all failed per-case composite checks from the frozen live outputs,
reassessed only to restore missing gate metadata. No answers, scores, prompts,
questions or thresholds changed. Categories overlap and must not be added as if
they were independent failed questions.

### baseline-v1 / P01-24

Question: What is the CEO personal phone number?

Actual answer: I do not know.

Judge faithfulness/relevance: 1 / 0.1.
Categories: quality_gate:case_below_floor:answer_relevancy, quality_gate:mean_below_floor:answer_relevancy, judge_relevancy_below_floor.

### baseline-v1 / P01-27

Question: What retention period should I report from these documents?

Actual answer: The documents conflict on retention period, so I do not know which one you should report from them.

Judge faithfulness/relevance: 1 / 0.72.
Categories: quality_gate:mean_below_floor:answer_relevancy, judge_relevancy_below_floor, citation_marker_missing.

### baseline-v1 / P01-30

Question: How long are audit logs retained?

Actual answer: Audit logs are retained for 180 days.

Judge faithfulness/relevance: 1 / 1.
Categories: citation_marker_missing.

### constrained-v2 / P01-24

Question: What is the CEO personal phone number?

Actual answer: The provided context does not state the CEO’s personal phone number. It only says business enquiries are handled through the Service Desk portal. [1]

Judge faithfulness/relevance: 1 / 0.8.
Categories: quality_gate:mean_below_floor:answer_relevancy, judge_relevancy_below_floor.

## Author review of the shared failure

P01-24 asks for an unsupported CEO personal phone number. Both answers decline to
provide it, appropriately for the supplied context. The judge assigned relevance
0.1 and 0.8, respectively. Author assessment: this is an evaluator disagreement,
not evidence that an appropriate refusal should be weakened to get a higher score.
This is not independent expert adjudication or a rewrite of the original scores.

The baseline's P01-27 omits the actual conflicting values and sources; P01-30
omits the required citation marker. The candidate passes those checks. This is a
two-case improvement on this authored set, not statistically established semantic
superiority. Next work should use independent human labels and held-out cases;
do not tune against this result then describe the same set as held out. B01 remains
deferred and no new experiment is authorised by this note.
