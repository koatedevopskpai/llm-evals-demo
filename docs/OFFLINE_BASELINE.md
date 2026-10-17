# Clean-checkout offline baseline (C03)

Supported baseline: CPython 3.12. Install the committed lock into a fresh virtual
environment. Installation requires a package source; execution needs no cloud,
model or private .env. No cloud credentials should be configured for this job.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-offline.lock
.venv\Scripts\python -m pytest -p offline_guard tests -q
```

`offline_guard` disables python-dotenv loading and telemetry configuration and
blocks outbound Python socket connections before test collection. Internal
loopback socket-pair setup is allowed for Windows/asyncio plumbing; ordinary
loopback connections remain blocked. Tests use local fixtures
and patched application/provider behaviour. The guard is process-local, not an
operating-system security boundary; subprocesses are not isolated by it.

The fixture text and test expectations are in `tests/`. Test results establish
only the exercised local behaviour. They do not establish Azure connectivity,
retrieval quality, live-model faithfulness or a production deployment. Optional
integration tests remain unrun under the GBP 0 budget. See `evidence/C03.md` for
the recorded command, environment and results.
