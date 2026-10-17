"""No paid inference: exercise the distinction between checks and evidence."""
import json
import pytest
from evaluator import Evaluator
from run_evals import main

def test_mock_success_is_explicitly_not_live_quality(tmp_path):
    report = tmp_path / "mock.md"
    assert main(["--mode", "mock", "--report", str(report)]) == 0
    result = json.loads(report.with_suffix('.json').read_text())
    assert result['status'] == 'offline_pass'
    assert result['evaluator_identity'] == 'deterministic-placeholder-v1'
    assert result['live_quality_established'] is False
    assert 'cannot authorise' in report.read_text()

@pytest.mark.parametrize('mode', ['mock', 'real'])
@pytest.mark.parametrize('key', [None, 'dummy-not-a-real-key'])
def test_release_quality_fails_closed_before_loading_sdk(tmp_path, monkeypatch, mode, key):
    if key is None:
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    else:
        monkeypatch.setenv('OPENAI_API_KEY', key)
    def forbidden():
        pytest.fail('RAGAS must not load while live quality is unavailable')
    monkeypatch.setattr(Evaluator, '_load_ragas', staticmethod(forbidden))
    report = tmp_path / 'release.md'
    # A previous artifact must be overwritten, not left looking like a pass.
    report.write_text('stale successful report')
    assert main(['--mode', mode, '--require-live-quality', '--report', str(report)]) == 2
    result = json.loads(report.with_suffix('.json').read_text())
    assert result['status'] == 'blocked'
    assert result['rows'] == []
    assert result['live_quality_established'] is False

def test_real_mode_cannot_bypass_cli_release_flag(tmp_path):
    assert main(['--mode', 'real', '--report', str(tmp_path / 'real.md')]) == 2
    with pytest.raises(RuntimeError, match='until E01'):
        Evaluator('real')

def test_mock_threshold_failure_remains_a_failure(tmp_path):
    assert main(['--threshold', '1', '--report', str(tmp_path / 'failed.md')]) == 1

@pytest.mark.parametrize('value', ['nan', 'inf', '-0.1', '1.1'])
def test_invalid_threshold_cannot_authorise_a_pass(value):
    with pytest.raises(SystemExit) as exc:
        main(['--threshold', value])
    assert exc.value.code == 2
