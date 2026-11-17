"""Leaderboard board-listing lanes: top downloads + newest-uploads probation."""

from __future__ import annotations

import importlib.util
import sys

import pytest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_leaderboard():
    spec = importlib.util.spec_from_file_location("leaderboard", SCRIPTS / "leaderboard.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["leaderboard"] = mod
    spec.loader.exec_module(mod)
    return mod


def _d(i, dl=0):
    return {"id": f"org/ds{i}", "downloads": dl, "likes": 0, "data_bytes": 1}


def test_merge_lanes_unions_without_duplicates():
    lb = _load_leaderboard()
    top = [_d(i, 100 - i) for i in range(5)]
    recent = [_d(3), _d(100), _d(101), _d(102)]
    out = lb._merge_lanes(top, recent, limit=4, new_lane=3)
    assert [d["id"] for d in out] == ["org/ds0", "org/ds1", "org/ds2", "org/ds3",
                                      "org/ds100", "org/ds101", "org/ds102"]
    assert [d["lane"] for d in out] == ["top"] * 4 + ["new"] * 3


def test_merge_lanes_caps_and_empty_lane():
    lb = _load_leaderboard()
    top = [_d(i) for i in range(10)]
    assert [d["lane"] for d in lb._merge_lanes(top, [], 3, 0)] == ["top"] * 3
    out = lb._merge_lanes(top, [_d(50)], 3, 5)  # lane bigger than supply
    assert len(out) == 4


def test_index_marks_new_lane_rows():
    lb = _load_leaderboard()
    entries = [{"id": "org/ds", "slug": "org--ds", "downloads": 1, "lane": "new",
                "report": {"verdict": "PASS", "score": 100, "score_model": "tiered-v2",
                           "episodes": 10, "truncated": False, "checks": []}}]
    html = lb.render_index(entries, "now")
    assert "(new)" in html
    entries[0]["lane"] = "top"
    assert "(new)" not in lb.render_index(entries, "now")


def test_video_subsample_picks_from_targets_under_cap():
    lb = _load_leaderboard()
    sizes = {f"videos/chunk-{i:03d}/f.mp4": 10 for i in range(100)}
    pick = lb._video_subsample(sorted(sizes), sizes, 24, 400)
    assert pick is not None and len(pick) <= 24
    assert set(pick) <= set(sizes)  # only files the targeting returned
    # spread across the range, not a prefix of early chunks
    assert pick[0].endswith("chunk-000/f.mp4") and pick[-1].endswith("chunk-099/f.mp4")


def test_video_subsample_refuses_over_cap_and_unknown_sizes():
    lb = _load_leaderboard()
    sizes = {"videos/a.mp4": 300, "videos/b.mp4": 200}
    assert lb._video_subsample(sorted(sizes), sizes, 24, 400) is None  # 500 > 400
    assert lb._video_subsample(sorted(sizes), sizes, 24, 500) == ["videos/a.mp4", "videos/b.mp4"]
    assert lb._video_subsample(["videos/missing.mp4"], sizes, 24, 500) is None
    assert lb._video_subsample([], sizes, 24, 500) is None


def test_video_subsample_expands_whole_tree_sentinel():
    lb = _load_leaderboard()
    sizes = {"videos/a.mp4": 10, "videos/b/c.mp4": 10, "data/x.parquet": 10, "meta/y.json": 1}
    pick = lb._video_subsample(["videos/**"], sizes, 24, 400)
    assert pick == ["videos/a.mp4", "videos/b/c.mp4"]  # only videos/ paths
    assert lb._video_subsample(["videos/**"], sizes, 24, 15) is None  # 20 > 15


def test_video_single_shard_picks_middle_and_respects_cap():
    lb = _load_leaderboard()
    targets = [f"videos/cam/chunk-{i:03d}/file-000.mp4" for i in range(10)]
    sizes = {t: 100 for t in targets}
    # middle of the sorted list - never the first (warm-up) shard
    assert lb._video_single_shard(targets, sizes, 400) == targets[5]
    assert lb._video_single_shard(targets, sizes, 99) is None  # one file > cap
    sizes2 = dict(sizes)
    del sizes2[targets[5]]
    assert lb._video_single_shard(targets, sizes2, 400) is None  # unknown size
    assert lb._video_single_shard([], sizes, 400) is None
    # sentinel expansion: only videos/ paths
    s3 = {"videos/a.mp4": 10, "videos/b.mp4": 20, "data/x.parquet": 5}
    assert lb._video_single_shard(["videos/**"], s3, 400) == "videos/b.mp4"


def test_audit_one_falls_back_to_single_shard_when_spread_over_cap(tmp_path, monkeypatch):
    """Over-cap datasets must still get a real visual-quality verdict from
    one middle shard instead of a SKIP."""
    av = pytest.importorskip("av")
    import json
    import shutil

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from kinema.fixtures.synth import make_dataset

    lb = _load_leaderboard()
    cam, fps, length, episodes, size = "observation.images.top", 10, 6, 6, 64
    root = make_dataset(tmp_path / "ds", episodes=episodes)
    vid = root / "videos" / cam / "chunk-000" / "file-000.mp4"
    vid.parent.mkdir(parents=True)
    with av.open(str(vid), "w") as out:
        st = out.add_stream("mpeg4", rate=fps)
        st.width = st.height = size
        st.pix_fmt = "yuv420p"
        st.bit_rate = 4_000_000
        img = np.full((size, size, 3), 140, dtype=np.uint8)
        img[16:48, 16:48] = 220
        for _ in range(episodes * length):
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    mf = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m = pq.read_table(mf).to_pydict()
    m[f"videos/{cam}/chunk_index"] = [0] * episodes
    m[f"videos/{cam}/file_index"] = [0] * episodes
    m[f"videos/{cam}/from_timestamp"] = [e * length / fps for e in m["episode_index"]]
    m[f"videos/{cam}/to_timestamp"] = [(e + 1) * length / fps for e in m["episode_index"]]
    pq.write_table(pa.table(m), mf)
    info = json.loads((root / "meta" / "info.json").read_text())
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    info["features"][cam] = {"dtype": "video", "shape": [size, size, 3]}
    (root / "meta" / "info.json").write_text(json.dumps(info))

    real_sizes = {str(p.relative_to(root)): p.stat().st_size
                  for p in root.rglob("*") if p.is_file()}
    # lie about the shard's size: under the single-file cap but the dataset
    # ALSO has phantom shards that push any 24-file spread far over the cap
    vid_rel = str(vid.relative_to(root))
    sizes = dict(real_sizes)
    sizes[vid_rel] = lb.VIDEO_CAP_BYTES - 1
    # 15 phantom shards sorting before + 15 after, so the deterministic
    # middle pick (index 15 of 31) lands on the real shard
    for i in range(15):
        sizes[f"videos/aaa.phantom/chunk-{i:03d}/file-000.mp4"] = lb.VIDEO_CAP_BYTES - 1
        sizes[f"videos/zzz.phantom/chunk-{i:03d}/file-000.mp4"] = lb.VIDEO_CAP_BYTES - 1
    downloaded = []

    def fake_download_sample(repo_id, cap_bytes, dest):
        for rel in real_sizes:
            if rel.startswith(("meta/", "data/")):
                tgt = dest / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / rel, tgt)
        return dest, sizes

    def fake_hub_download(repo_id, rel, dest):
        downloaded.append(rel)
        tgt = dest / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, tgt)
        return tgt

    monkeypatch.setattr(lb, "download_sample", fake_download_sample)
    monkeypatch.setattr(lb, "_hub_download", fake_hub_download)
    rep = lb.audit_one("org/ds", 250, 50 * 1000 * 1000)
    assert rep.video_audited == 4
    vq = {r.check_id: r for r in rep.results}["visual-quality"]
    assert not vq.skipped, f"visual-quality skipped: {vq.skip_reason}"
    assert downloaded == [vid_rel]  # exactly one shard fetched, not the spread


def test_audit_one_runs_visual_quality_on_downloaded_videos(tmp_path, monkeypatch):
    """Regression for the v2 lifecycle bug: the checks used to run after the
    tempdir was deleted, so every board visual-quality verdict was a SKIP
    even though the MP4s had been fetched."""
    av = pytest.importorskip("av")
    import json

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from kinema.fixtures.synth import make_dataset

    lb = _load_leaderboard()
    cam, fps, length, episodes, size = "observation.images.top", 10, 6, 6, 64
    root = make_dataset(tmp_path / "ds", episodes=episodes)
    vid = root / "videos" / cam / "chunk-000" / "file-000.mp4"
    vid.parent.mkdir(parents=True)
    with av.open(str(vid), "w") as out:
        st = out.add_stream("mpeg4", rate=fps)
        st.width = st.height = size
        st.pix_fmt = "yuv420p"
        st.bit_rate = 4_000_000
        img = np.full((size, size, 3), 140, dtype=np.uint8)
        img[16:48, 16:48] = 220
        for _ in range(episodes * length):
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    mf = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m = pq.read_table(mf).to_pydict()
    m[f"videos/{cam}/chunk_index"] = [0] * episodes
    m[f"videos/{cam}/file_index"] = [0] * episodes
    m[f"videos/{cam}/from_timestamp"] = [e * length / fps for e in m["episode_index"]]
    m[f"videos/{cam}/to_timestamp"] = [(e + 1) * length / fps for e in m["episode_index"]]
    pq.write_table(pa.table(m), mf)
    info = json.loads((root / "meta" / "info.json").read_text())
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    info["features"][cam] = {"dtype": "video", "shape": [size, size, 3]}
    (root / "meta" / "info.json").write_text(json.dumps(info))

    sizes = {str(p.relative_to(root)): p.stat().st_size
             for p in root.rglob("*") if p.is_file()}

    import shutil

    def fake_download_sample(repo_id, cap_bytes, dest):
        for rel in sizes:
            if rel.startswith(("meta/", "data/")):
                tgt = dest / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / rel, tgt)
        return dest, sizes

    def fake_hub_download(repo_id, rel, dest):
        tgt = dest / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, tgt)
        return tgt

    monkeypatch.setattr(lb, "download_sample", fake_download_sample)
    monkeypatch.setattr(lb, "_hub_download", fake_hub_download)
    rep = lb.audit_one("org/ds", 250, 50 * 1000 * 1000)
    assert rep.video_audited == 4
    vq = {r.check_id: r for r in rep.results}["visual-quality"]
    assert not vq.skipped, f"visual-quality skipped: {vq.skip_reason}"


def test_hub_listing_kwargs_match_installed_api(monkeypatch):
    """Listing calls must only use kwargs the installed huggingface_hub accepts.

    hub 2.x removed `direction`; the nightly then died inside
    find_lerobot_datasets before auditing anything and `|| true` kept the
    run green. Capture every kwarg both lanes pass and validate against the
    real signature of the installed HfApi.list_datasets.
    """
    import inspect

    from huggingface_hub import HfApi

    lb = _load_leaderboard()
    calls = []

    class _FakeApi:
        def list_datasets(self, **kwargs):
            calls.append(kwargs)
            return iter([])

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeApi())
    lb.find_lerobot_datasets(2, 2)

    assert len(calls) == 2  # top lane + newest lane
    params = inspect.signature(HfApi.list_datasets).parameters
    for kwargs in calls:
        unknown = set(kwargs) - set(params)
        assert not unknown, f"list_datasets kwargs not in installed hub API: {unknown}"


def test_audit_stale_reaudits_rows_without_video_audit():
    lb = _load_leaderboard()
    engine = "e" * 12
    base = {"engine": engine, "report": {"source": "org/ds", "episodes": 10, "truncated": False}}
    assert lb.audit_stale(base, engine, 250) is True  # no video_audited key
    for legacy in (True, 1, 2, 3):  # every superseded mark re-audits once
        row = {"engine": engine, "report": {**base["report"], "video_audited": legacy}}
        assert lb.audit_stale(row, engine, 250) is True
    settled = {"engine": engine, "report": {**base["report"], "video_audited": 4}}
    assert lb.audit_stale(settled, engine, 250) is False
    # depth-staleness still applies on top once video is settled
    deep = {"engine": engine, "report": {**base["report"], "video_audited": 4, "truncated": True}}
    assert lb.audit_stale(deep, engine, 250) is True


def test_report_video_audited_roundtrip():
    from kinema.report import DatasetReport, report_from_dict, report_to_dict

    rep = DatasetReport(source="org/ds", format_version="v3", episodes=1,
                        frames=10, fps=30.0, video_audited=4)
    assert report_from_dict(report_to_dict(rep)).video_audited == 4
    # rows stored before the field existed read as mark 0 (re-audit once)
    d = report_to_dict(rep)
    del d["video_audited"]
    assert report_from_dict(d).video_audited == 0
    # rows stored by the buggy v1 targeting (true) read as mark 1
    d["video_audited"] = True
    assert report_from_dict(d).video_audited == 1
