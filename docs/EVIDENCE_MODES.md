# Evaluation evidence modes (E02)

`python run_evals.py --mode mock --report artifacts/mock.md` exercises placeholder
scores and thresholds. Exit 0 means an offline contract passed, not that an
application answer is faithful. JSON and Markdown identify mode, evaluator and
`live_quality_established: false`.

`--require-live-quality` returns exit 2 with a blocked artifact, for either mode.
`--mode real` also blocks independently of that flag, before importing SDKs or
making provider calls. The programmatic Evaluator('real') path rejects use too.
This intentionally disables the known reference-as-answer scoring until E01
implements and validates generated-answer evaluation. E02 does not implement E01.

The automatic CI job is named `offline-scaffold` and receives no provider secrets.
A manually requested live-quality check fails explicitly; it cannot quietly skip
and become green. Ordinary offline CI remains useful without pretending to be a
release-quality job. Actual deployment integration is outside this change.

No live result or benchmark has been produced. E01 remains budget-blocked under
the GBP 0 plan. To unblock live quality later requires both correct application
measurement and explicit budget authorisation; merely restoring credentials is
insufficient. See evidence/E02.md for local checks and pending remote checks.
