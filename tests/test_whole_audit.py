"""Independent regressions for the whole-repository correctness review."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.checks.deployment import _spatial_size
from kinema.checks.stats_drift import _errors
from kinema.dataset import load_dataset
from kinema.fixtures.synth import make_dataset

CHECKS = {c.id: c for c in ALL_CHECKS}


def run(name, ctx):
    return CHECKS[name].run(ctx, CheckConfig())


def test_nan_dimension_does_not_hide_other_dimension_drift(tmp_path):
    ctx = load_dataset(str(make_dataset(tmp_path / 'd')))
    ctx.state = ctx.state.copy()
    ctx.state[:, 0] = np.nan
    ctx.state[:, 1] += 100
    result = run('stats-drift', ctx)
    assert any(f.severity == Severity.FAIL and f.details.get('dim') == 1
               for f in result.findings)


def test_high_mean_does_not_make_varying_dimension_dead():
    actual = {k: np.array([v]) for k, v in
              dict(mean=1000., std=.1, min=999.5, max=1000.5).items()}
    stored = {k: v.copy() for k, v in actual.items()}
    stored['std'][:] = .01
    errs, _ = _errors(actual, stored)
    assert errs['std'][0] > 8


def test_inf_does_not_disable_idle_for_other_episodes(tmp_path):
    ctx = load_dataset(str(make_dataset(tmp_path / 'd', defects=['idle-segments'])))
    before = {f.episode_index for f in run('idle-segments', ctx).findings
              if f.severity >= Severity.WARN}
    assert before
    ctx.action = ctx.action.copy()
    ctx.action[-1, 0] = np.inf
    after = {f.episode_index for f in run('idle-segments', ctx).findings
             if f.severity >= Severity.WARN}
    assert before - {ctx.episode_ids()[-1]} <= after


@pytest.mark.parametrize('spill', [False, True])
def test_width_one_action_stays_matrix(tmp_path, monkeypatch, spill):
    monkeypatch.setenv('KINEMA_FRAME_BUDGET_BYTES', '1' if spill else '0')
    root = make_dataset(tmp_path / 'd')
    for path in root.glob('data/**/*.parquet'):
        t = pq.read_table(path)
        idx = t.schema.get_field_index('action')
        t = t.set_column(idx, 'action', pa.array([[v[0]] for v in t['action'].to_pylist()]))
        pq.write_table(t, path)
    ctx = load_dataset(str(root))
    assert ctx.action.ndim == 2 and ctx.action.shape[1] == 1
    for name in ('calibration-drift', 'idle-segments', 'stuck-actuator', 'action-saturation',
                 'nan-and-normalization', 'outlier-episodes', 'duplicate-and-leakage'):
        run(name, ctx)  # no swallowed check crashes


def test_camera_shapes_follow_hwc_and_explicit_axes():
    assert _spatial_size({'shape': [480, 640, 3]}) == (640, 480)
    assert _spatial_size({'shape': [3, 480, 640]}) == (640, 480)
    assert _spatial_size({'shape': [3, 480, 640], 'names': ['channel', 'height', 'width']}) == (640, 480)


@pytest.mark.parametrize('spill', [False, True])
@pytest.mark.parametrize('missing_first', [False, True])
def test_mixed_shard_columns_fail_closed_instead_of_cross_episode_values(tmp_path, monkeypatch, spill, missing_first):
    from kinema.dataset import DatasetError
    monkeypatch.setenv('KINEMA_FRAME_BUDGET_BYTES', '1' if spill else '0')
    root = make_dataset(tmp_path / 'd', episodes=6, shards=2)
    files = sorted(root.glob('data/**/*.parquet'))
    path = files[0 if missing_first else -1]
    t = pq.read_table(path).drop(['observation.state'])
    pq.write_table(t, path)
    with pytest.raises(DatasetError, match='inconsistent.*columns'):
        load_dataset(str(root))


def test_null_done_is_not_true(tmp_path):
    root = make_dataset(tmp_path / 'd', episodes=4)
    for path in root.glob('data/**/*.parquet'):
        t = pq.read_table(path)
        t = t.append_column('next.done', pa.array([None] * len(t), type=pa.bool_()))
        pq.write_table(t, path)
    import json
    p = root / 'meta/info.json'
    info = json.loads(p.read_text())
    info['features']['next.done'] = {'dtype': 'bool', 'shape': [1]}
    p.write_text(json.dumps(info))
    ctx = load_dataset(str(root))
    assert np.isnan(ctx.done).all()
    res = run('reward-done', ctx)
    assert not any(f.severity >= Severity.WARN for f in res.findings)
    assert not any('terminate at birth' in f.message for f in res.findings)


def test_short_action_names_do_not_crash_gripper_message(tmp_path):
    ctx = load_dataset(str(make_dataset(tmp_path / 'd')))
    ctx.action = ctx.action.copy()
    ctx.action[:, 5] += 4.0
    ctx.action_names = ['only-one-name']
    CHECKS['calibration-drift']._gripper(ctx, CheckConfig(), (5, 5))
