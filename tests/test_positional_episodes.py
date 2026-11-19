"""Frames that number episodes by position while meta/episodes catalogues
source ids (lerobot/behavior1k-task0045: 450010.. vs 0..199)."""
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.dataset import DatasetError, load_dataset
from kinema.fixtures.synth import make_dataset

CHECKS = {c.id: c for c in ALL_CHECKS}


def _recatalogue(root, ids):
    mf = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m = pq.read_table(mf)
    i = m.column_names.index("episode_index")
    pq.write_table(m.set_column(i, "episode_index", pa.array(ids, type=pa.int64())), mf)


def test_positional_frames_are_matched_and_flagged(tmp_path):
    root = make_dataset(tmp_path / "p", episodes=12)
    _recatalogue(root, [450010 + 10 * k for k in range(12)])
    ctx = load_dataset(str(root))
    assert ctx.episode_ids_by_position[:2] == [450010, 450020]
    assert ctx.episode_ids()[:2] == [450010, 450020]  # frames carry catalogue ids now
    df = CHECKS["dropped-frames"].run(ctx, CheckConfig())
    assert any(f.details.get("kind") == "positional-episode-index" and f.severity == Severity.WARN
               for f in df.findings)
    # lengths line up by position: no metadata length mismatch, no FAIL
    assert df.verdict == Severity.WARN, [f.message for f in df.findings]
    ptr = CHECKS["episode-file-pointers"].run(ctx, CheckConfig())
    assert ptr.verdict < Severity.WARN, [f.message for f in ptr.findings]


def test_partial_overlap_is_not_matched_by_position(tmp_path):
    # ids that overlap the frames are a real disagreement, not positional
    root = make_dataset(tmp_path / "q", episodes=12)
    _recatalogue(root, [5 + k for k in range(12)])  # 5..16 vs frames 0..11
    ctx = load_dataset(str(root))
    assert ctx.episode_ids_by_position is None
    assert CHECKS["dropped-frames"].run(ctx, CheckConfig()).verdict == Severity.FAIL


def test_unordered_catalogue_is_not_matched_by_position(tmp_path):
    root = make_dataset(tmp_path / "r", episodes=12)
    _recatalogue(root, [450010 + 10 * k for k in reversed(range(12))])
    with pytest.raises(DatasetError):
        load_dataset(str(root))
