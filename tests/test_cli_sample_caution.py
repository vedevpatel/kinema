"""Regression pin for the sampled-audit caution (finding #11 scale ladder).

Verdicts below ~250 sampled episodes were observed flipping both ways on
real data (robocasa stats-drift WARN->PASS, outlier-episodes INFO->WARN;
DROID stuck-actuator PASS->WARN from 100). The CLI must say so on a small
truncated sample, and stay quiet on a full load or an untruncated sample.
"""

from __future__ import annotations

from kinema.cli import main
from kinema.fixtures.synth import make_dataset


def test_caution_on_small_truncated_sample(tmp_path, capsys):
    ds = make_dataset(tmp_path / "ds", episodes=24)
    rc = main(["check", str(ds), "--max-episodes", "10"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "sampled audit of 10 episodes" in err
    assert "can flip either way" in err


def test_no_caution_on_full_load(tmp_path, capsys):
    ds = make_dataset(tmp_path / "ds", episodes=24)
    rc = main(["check", str(ds)])
    assert rc == 0
    assert "sampled audit" not in capsys.readouterr().err


def test_no_caution_when_sample_covers_dataset(tmp_path, capsys):
    """--max-episodes >= the dataset's episodes is not truncated: the load
    IS the full dataset, so the caution would be wrong."""
    ds = make_dataset(tmp_path / "ds", episodes=24)
    rc = main(["check", str(ds), "--max-episodes", "24"])
    assert rc == 0
    assert "sampled audit" not in capsys.readouterr().err
