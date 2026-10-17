# artifacts/

This directory holds **generated evidence**, not source. Everything in it is
reproducible from the commands below.

## Nothing here is committed

`artifacts/*` is git-ignored, apart from this README. Per-run output used to be
tracked, and that caused a merge conflict on every pair of branches created from
the same base: on 17 October 2026 pull requests #1 and #2 both added
`artifacts/pytest.xml` and `artifacts/test-output.txt`, and the add/add conflict
had to be resolved by hand.

Committing a file whose name repeats on every run is the root cause. Regenerate
it instead.

## How these files are produced

```bash
# Offline test run, with machine-readable results
python -m pytest -p offline_guard tests -q --junitxml=artifacts/pytest.xml

# Mock evaluation report (writes both the .md and a matching .json)
python run_evals.py --mode mock --report artifacts/mock-report.md

# The required-live form, which fails closed until E01 is implemented
python run_evals.py --mode real --require-live-quality --report artifacts/live-report.md
```

## Where the durable evidence lives instead

| What | Where |
|---|---|
| Human-readable task evidence | `evidence/C*.md`, `evidence/E*.md` |
| How to reproduce a baseline | `docs/OFFLINE_BASELINE.md`, `docs/EVIDENCE_MODES.md` |
| Test results from a real run | Uploaded as workflow artifacts by `.github/workflows/` |
| Pinned dependency set | `requirements-offline.lock` |

`environment.json` is also written here by the test guard. It records the Python
version and installed package versions for a run; it is regenerated, not stored.

## Environment versions are not a stable claim

Because these files are regenerated, any version numbers in them describe **one
particular run on one machine**. Do not cite them as a repository-wide property.
The pinned set in `requirements-offline.lock` is the durable statement.
