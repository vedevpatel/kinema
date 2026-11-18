"""A bad pinned input must not publish a smaller, apparently passing study."""
import importlib.util
import sys
from pathlib import Path


def test_failed_pin_aborts_calibration(monkeypatch, capsys):
    path = Path(__file__).parents[1] / 'scripts/calibrate.py'
    spec = importlib.util.spec_from_file_location('calibrate_contract', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.setattr(sys, 'argv', ['calibrate.py', 'org/bad@missing'])
    def fail(*args, **kwargs):
        raise ValueError('revision not found')
    monkeypatch.setattr(m, 'load_dataset', fail)
    monkeypatch.setattr(m, 'run_calibration', lambda *a: (_ for _ in ()).throw(AssertionError('must not run')))
    assert m.main() == 1
    assert 'no partial calibration published' in capsys.readouterr().err
