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


def test_merge_lanes_flagship_seats():
    lb = _load_leaderboard()
    top = [_d(i, 100 - i) for i in range(4)]
    # org/ds2 is already on the top board: no duplicate
    flagship = [{"id": "org/ds2", "downloads": 999, "likes": 0, "data_bytes": 1},
                {"id": "RoboCOIN/big", "downloads": 900, "likes": 0, "data_bytes": 1}]
    recent = [_d(200), _d(201)]
    out = lb._merge_lanes(top, recent, limit=3, new_lane=2, flagship=flagship)
    assert [d["id"] for d in out] == ["org/ds0", "org/ds1", "org/ds2",
                                      "RoboCOIN/big", "org/ds200", "org/ds201"]
    assert [d["lane"] for d in out] == ["top"] * 3 + ["flagship"] + ["new"] * 2


def test_merge_lanes_flagship_does_not_eat_the_new_lane():
    lb = _load_leaderboard()
    # the new lane keeps its full seat count however many flagship rows merge
    flagship = [{"id": f"RoboCOIN/t{i}", "downloads": 900, "likes": 0, "data_bytes": 1}
                for i in range(25)]
    recent = [_d(300 + i) for i in range(5)]
    out = lb._merge_lanes([], recent, limit=3, new_lane=5, flagship=flagship)
    assert [d["lane"] for d in out].count("flagship") == 25
    assert [d["lane"] for d in out].count("new") == 5


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
    assert lb._video_single_shard(targets, sizes, 99) is None  # all files > cap
    assert lb._video_single_shard([], sizes, 400) is None
    # sentinel expansion: only videos/ paths
    s3 = {"videos/a.mp4": 10, "videos/b.mp4": 20, "data/x.parquet": 5}
    assert lb._video_single_shard(["videos/**"], s3, 400) == "videos/b.mp4"


def test_video_single_shard_skips_unbudgetable_targets():
    lb = _load_leaderboard()
    # regression: v1 checked only the middle target and gave up when IT was
    # over the cap, even when other targets fit (berkeley_rpt: 98/523/524 MB
    # targets skipped entirely because the middle one was 523 MB)
    targets = [f"videos/cam/chunk-000/file-{i:03d}.mp4" for i in range(3)]
    sizes = {targets[0]: 98, targets[1]: 523, targets[2]: 524}
    assert lb._video_single_shard(targets, sizes, 400) == targets[0]
    # the middle of the BUDGETABLE list, not the first fitting file
    sizes = {targets[0]: 100, targets[1]: 523, targets[2]: 100}
    assert lb._video_single_shard(targets, sizes, 400) == targets[2]
    # unlisted targets (missing from the size map) are skipped over, not
    # treated as a reason to give up (robodojo: meta referenced 88 files
    # absent from the repo, and the middle pick was one of them)
    targets = [f"videos/cam/chunk-{i:03d}/file-000.mp4" for i in range(10)]
    sizes = {t: 100 for t in targets}
    del sizes[targets[5]]
    assert lb._video_single_shard(targets, sizes, 400) == targets[4]
    # nothing budgetable at all -> honest None
    assert lb._video_single_shard(targets, {}, 400) is None


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
    assert rep.video_audited == 5
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
    assert rep.video_audited == 5
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
    monkeypatch.setattr(lb, "probe_rlds_lane", lambda workers=8: [])  # no live GCS in a kwargs test
    lb.find_lerobot_datasets(2, 2)

    assert len(calls) == 6  # top lane + newest lane + 4 flagship org queries
    assert sum('author' in c for c in calls) == len(lb.FLAGSHIP_ORG_SEATS)
    params = inspect.signature(HfApi.list_datasets).parameters
    for kwargs in calls:
        unknown = set(kwargs) - set(params)
        assert not unknown, f"list_datasets kwargs not in installed hub API: {unknown}"


def test_audit_stale_reaudits_rows_without_video_audit():
    lb = _load_leaderboard()
    engine = "e" * 12
    base = {"engine": engine, "report": {"source": "org/ds", "episodes": 10, "truncated": False}}
    assert lb.audit_stale(base, engine, 250) is True  # no video_audited key
    for legacy in (True, 1, 2, 3, 4):  # every superseded mark re-audits once
        row = {"engine": engine, "report": {**base["report"], "video_audited": legacy}}
        assert lb.audit_stale(row, engine, 250) is True
    settled = {"engine": engine, "report": {**base["report"], "video_audited": 5}}
    assert lb.audit_stale(settled, engine, 250) is False
    # depth-staleness still applies on top once video is settled
    deep = {"engine": engine, "report": {**base["report"], "video_audited": 5, "truncated": True}}
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
def test_flagship_probe_skips_gated_repos(monkeypatch):
    """Gated flagship candidates must not take seats: info.json is public on
    gated repos, so without the gated check the seat shows 'gated' forever
    instead of going to the next auditable dataset."""
    import types

    lb = _load_leaderboard()
    monkeypatch.setattr(lb, "FLAGSHIP_ORG_SEATS", {"RoboCOIN": 2})
    monkeypatch.setattr(lb, "FLAGSHIP_SEEDS", ())
    candidates = [("RoboCOIN/gated_a", True), ("RoboCOIN/open_b", False),
                  ("RoboCOIN/gated_c", True), ("RoboCOIN/open_d", False)]

    class _FakeApi:
        def list_datasets(self, **kwargs):
            if "author" in kwargs:
                return iter(types.SimpleNamespace(id=rid, downloads=100 - i, likes=0)
                            for i, (rid, _) in enumerate(candidates))
            return iter([])

        def dataset_info(self, repo_id, **kwargs):
            gated = dict(candidates)[repo_id]
            return types.SimpleNamespace(
                gated=("auto" if gated else False),
                siblings=[types.SimpleNamespace(rfilename="meta/info.json", size=1)],
                downloads=10, likes=0)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeApi())
    monkeypatch.setattr(lb, "probe_rlds_lane", lambda workers=8: [])  # no live GCS in a flagship test
    out = lb.find_lerobot_datasets(2, 0)
    flag = [d["id"] for d in out if d["lane"] == "flagship"]
    assert flag == ["RoboCOIN/open_b", "RoboCOIN/open_d"]


def test_download_sample_skips_dual_layout_leftovers(monkeypatch, tmp_path):
    """v3 repos carrying stale episode_*.parquet beside the packed shards:
    the byte cap must buy the shards the metadata points to (seen on
    unitreerobotics/G1_Dex1_Stack_Block: 200 duplicate files sorted ahead
    of the packed shards and ate the download budget)."""
    import types

    import pyarrow as pa
    import pyarrow.parquet as pq

    lb = _load_leaderboard()
    sib = [
        types.SimpleNamespace(rfilename="meta/info.json", size=10),
        types.SimpleNamespace(rfilename="meta/episodes/chunk-000/file-000.parquet", size=10),
        types.SimpleNamespace(rfilename="data/chunk-000/episode_000000.parquet", size=100),
        types.SimpleNamespace(rfilename="data/chunk-000/episode_000001.parquet", size=100),
        types.SimpleNamespace(rfilename="data/chunk-000/file-000.parquet", size=100),
    ]
    tree = [types.SimpleNamespace(path=f.rfilename, size=f.size) for f in sib]
    monkeypatch.setattr("huggingface_hub.list_repo_tree",
                        lambda *a, **k: iter(tree))
    downloaded = []

    def fake_dl(rid, name, dest):
        downloaded.append(name)
        p = Path(dest) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if name == "meta/episodes/chunk-000/file-000.parquet":
            pq.write_table(pa.table({
                "episode_index": pa.array([0, 1], type=pa.int64()),
                "data/chunk_index": pa.array([0, 0], type=pa.int64()),
                "data/file_index": pa.array([0, 0], type=pa.int64()),
            }), p)
        else:
            p.write_text("x")

    monkeypatch.setattr(lb, "_hub_download", fake_dl)
    lb.download_sample("org/ds", 10**9, tmp_path)
    assert "data/chunk-000/file-000.parquet" in downloaded
    assert not any("episode_" in n for n in downloaded)


# --- nested datasets (repo//subpath) ---------------------------------------


def _fake_item(path, size=0):
    from huggingface_hub.hf_api import RepoFile

    return RepoFile(path=path, size=size, oid=None, lfs=None,
                    lastCommit=None, security=None)


def _fake_dir(path):
    from huggingface_hub.hf_api import RepoFolder

    return RepoFolder(path=path, oid=None, lastCommit=None)


def _fake_api(paths, gated=False):
    """paths: [(path, size)] of FILES; dirs are derived. Supports the
    non-recursive per-directory listing + get_paths_info shape."""
    import types

    files = [(p, s) for p, s in paths]

    def children(prefix):
        out = {}
        for p, s in files:
            if prefix and not p.startswith(prefix + "/"):
                continue
            rest = p[len(prefix) + 1:] if prefix else p
            head, _, tail = rest.partition("/")
            if tail:
                out.setdefault(prefix + "/" + head if prefix else head, None)
            else:
                out[p] = s
        return out

    class _Api:
        def dataset_info(self, repo_id, **kw):
            return types.SimpleNamespace(gated=gated, downloads=100, likes=5)

        def list_repo_tree(self, repo_id, repo_type=None, path_in_repo=None, recursive=False):
            if recursive:
                prefix = path_in_repo or ""
                return iter([_fake_item(p, s) for p, s in files
                             if not prefix or p.startswith(prefix + "/")])
            return iter([_fake_dir(p) if s is None else _fake_item(p, s)
                         for p, s in children(path_in_repo or "").items()])

        def get_paths_info(self, repo_id, paths, repo_type=None):
            have = {p for p, _ in files}
            return [_fake_item(p, 0) for p in paths if p in have]

    return _Api()


def _tree_paths(*task_specs):
    """Build a fake repo tree: (task, data_bytes) pairs at depth 1."""
    out = []
    for task, data_bytes in task_specs:
        out.append((f"{task}/meta/info.json", 10))
        out.append((f"{task}/data/chunk-000/file-000.parquet", data_bytes))
        out.append((f"{task}/videos/chunk-000/file-000.mp4", 999))
    return out


def test_probe_nested_finds_roots_and_ranks_by_data_bytes():
    lb = _load_leaderboard()
    api = _fake_api(
        [("meta/info.json", 10)]  # root-level dataset: the plain probe's row
        + _tree_paths(("task_b", 50), ("task_a", 100))
        + [("assets/meta/info.json", 10),          # never a dataset root
           (".hidden/meta/info.json", 10),         # dot dirs are not datasets
           ("a/very/deep/dir/meta/info.json", 10)]  # past the depth cap
    )
    out = lb._probe_nested(api, "org/repo", seats=1)
    assert [d["id"] for d in out] == ["org/repo//task_a"]
    assert out[0]["subpath"] == "task_a"
    assert out[0]["repo_scope"] is True
    assert out[0]["data_bytes"] == 100  # data/ only, not the videos
    assert out[0]["downloads"] == 100  # repo-level counts, marked repo_scope
    # both roots are found when the seats allow them, bigger first
    out2 = lb._probe_nested(api, "org/repo", seats=5)
    assert [d["id"] for d in out2] == ["org/repo//task_a", "org/repo//task_b"]


def test_probe_nested_descends_into_container_dirs():
    lb = _load_leaderboard()
    # Teleop-Sim's shape: a LeRobot/ container holding the task datasets
    api = _fake_api(_tree_paths(("LeRobot/task_x", 80), ("LeRobot/task_y", 60))
                    + [("HDF5/task_x/file-000.h5", 500)])
    out = lb._probe_nested(api, "org/repo", seats=5)
    assert [d["id"] for d in out] == ["org/repo//LeRobot/task_x", "org/repo//LeRobot/task_y"]


def test_probe_nested_skips_gated_and_caps_a_root_listing(monkeypatch):
    lb = _load_leaderboard()
    assert lb._probe_nested(_fake_api(_tree_paths(("task_a", 1)), gated=True),
                            "org/repo", 5) == []
    assert lb._probe_nested(_fake_api([]), "org/repo", 5) == []
    # an over-cap data/ listing sizes that root as 0 instead of costing
    # the repo its rows
    monkeypatch.setattr(lb, "_NESTED_ROOT_MAX_ITEMS", 0)
    out = lb._probe_nested(_fake_api(_tree_paths(("task_b", 50), ("task_a", 100))),
                           "org/repo", 5)
    assert [d["id"] for d in out] == ["org/repo//task_a", "org/repo//task_b"]
    assert all(d["data_bytes"] == 0 for d in out)


def test_index_marks_nested_rows_and_footnotes_repo_level_downloads():
    lb = _load_leaderboard()
    entries = [{"id": "org/repo//task_a", "slug": "org--repo----task_a", "downloads": 5,
                "lane": "flagship", "subpath": "task_a", "repo_scope": True,
                "report": {"verdict": "PASS", "score": 100, "score_model": "tiered-v2",
                           "episodes": 10, "truncated": False, "checks": []}}]
    html = lb.render_index(entries, "now")
    assert "(nested)</span>" in html
    assert "download counts are repo-level" in html
    entries[0].pop("subpath")
    assert "(nested)</span>" not in lb.render_index(entries, "now")


def test_download_sample_nested_filters_siblings_and_prefixes_fetches(monkeypatch, tmp_path):
    import types

    lb = _load_leaderboard()
    sib = [
        types.SimpleNamespace(rfilename="other/meta/info.json", size=10),
        types.SimpleNamespace(rfilename="task_a/meta/info.json", size=10),
        types.SimpleNamespace(rfilename="task_a/data/chunk-000/file-000.parquet", size=100),
        types.SimpleNamespace(rfilename="task_a/videos/chunk-000/file-000.mp4", size=100),
        types.SimpleNamespace(rfilename="unrelated/file.txt", size=1),
    ]
    tree = [types.SimpleNamespace(path=f.rfilename, size=f.size) for f in sib]
    monkeypatch.setattr("huggingface_hub.list_repo_tree",
                        lambda *a, **k: iter(tree))
    downloaded = []

    def fake_dl(rid, name, dest):
        downloaded.append((rid, name))
        p = Path(dest) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")

    monkeypatch.setattr(lb, "_hub_download", fake_dl)
    root, sizes = lb.download_sample("org/repo//task_a", 10**9, tmp_path)
    # the nested root is returned; names are root-relative in sizes but
    # repo-relative on the wire
    assert root == tmp_path / "task_a"
    assert sorted(sizes) == ["data/chunk-000/file-000.parquet", "meta/info.json",
                             "videos/chunk-000/file-000.mp4"]
    assert ("org/repo", "task_a/meta/info.json") in downloaded
    assert ("org/repo", "task_a/data/chunk-000/file-000.parquet") in downloaded
    assert not any(n.startswith("other/") or n.startswith("unrelated/")
                   for _, n in downloaded)


# --- multi-bucket rlds lane -------------------------------------------------

class _FakeRldsDs:
    def __init__(self):
        self.version = "1.0.0"
        self.data_bytes = 10
        self.episodes = 1


def _patch_rlds(monkeypatch, listings, probes, fail_listing=None):
    import kinema.rlds as rlds_mod

    class FakeClient:
        def __init__(self, bucket):
            self.bucket = bucket
            self.calls = 0

        def list(self, prefix="", delimiter=""):
            self.calls += 1
            if self.bucket == fail_listing:
                raise RuntimeError("gcs outage")
            return [prefix + n + "/" for n in listings[self.bucket]], []

    class FakeProbe:
        @classmethod
        def probe(cls, client, p):
            name = p.rstrip("/").split("/")[-1]
            if name not in probes.get(client.bucket, ()):
                raise ValueError("no real data")
            return _FakeRldsDs()

    monkeypatch.setattr(rlds_mod, "GcsClient", FakeClient)
    monkeypatch.setattr(rlds_mod, "RldsDataset", FakeProbe)


def test_probe_rlds_lane_first_bucket_with_data_wins(monkeypatch):
    lb = _load_leaderboard()
    gdm, gresearch = lb.RLDS_BUCKETS[0][0], lb.RLDS_BUCKETS[1][0]
    _patch_rlds(
        monkeypatch,
        listings={gdm: ["alpha", "beta", "gamma"], gresearch: ["alpha", "beta", "gamma", "delta"]},
        # beta and gamma are placeholder-only in gdm (probe fails) but real in
        # gresearch; delta is a non-Open-X set that boards too (approved scope)
        probes={gdm: ["alpha"], gresearch: ["alpha", "beta", "gamma", "delta"]},
    )
    rows = {r["id"]: r for r in lb.probe_rlds_lane()}
    assert set(rows) == {"rlds/alpha", "rlds/beta", "rlds/gamma", "rlds/delta"}
    assert rows["rlds/delta"]["rlds_bucket"] == gresearch
    assert rows["rlds/alpha"]["rlds_bucket"] == gdm  # mirrored: precedence wins
    assert rows["rlds/beta"]["rlds_bucket"] == gresearch  # placeholder falls through
    assert rows["rlds/gamma"]["rlds_bucket"] == gresearch
    assert all(r["rlds"] and r["downloads"] == 0 for r in rows.values())


def test_probe_rlds_lane_bucket_outage_zeroes_the_whole_lane(monkeypatch):
    # a listing-level failure must NOT fall through to the next bucket: that
    # would silently swap mirrored datasets' bytes on a transient outage.
    # Zero rows lets the resume prune guard keep the stored rlds board.
    lb = _load_leaderboard()
    gdm, gresearch = lb.RLDS_BUCKETS[0][0], lb.RLDS_BUCKETS[1][0]
    _patch_rlds(
        monkeypatch,
        listings={gdm: [], gresearch: ["gamma"]},
        probes={gresearch: ["gamma"]},
        fail_listing=gdm,
    )
    assert lb.probe_rlds_lane() == []


def test_audit_rlds_uses_the_rows_bucket(monkeypatch, tmp_path):
    lb = _load_leaderboard()
    gdm, gresearch = lb.RLDS_BUCKETS[0][0], lb.RLDS_BUCKETS[1][0]
    seen = []

    class FakeClient:
        def __init__(self, bucket):
            seen.append(bucket)
            self.bucket = bucket
            self.calls = 0

        def read_json(self, name):
            raise ValueError("stop after client construction")

    import kinema.rlds as rlds_mod
    monkeypatch.setattr(rlds_mod, "GcsClient", FakeClient)

    class FakeProbe:
        @classmethod
        def probe(cls, client, p):
            raise ValueError("stop after client construction")

    monkeypatch.setattr(rlds_mod, "RldsDataset", FakeProbe)
    try:
        lb.audit_rlds({"id": "rlds/gamma", "rlds": True,
                       "rlds_prefix": "robotics/gamma/", "rlds_bucket": gresearch})
    except ValueError:
        pass
    try:  # a stored row from before the field existed falls back to gdm
        lb.audit_rlds({"id": "rlds/alpha", "rlds": True, "rlds_prefix": "alpha/"})
    except ValueError:
        pass
    assert seen == [gresearch, gdm]


def test_probe_rlds_lane_budget_breaks_mid_pass(monkeypatch):
    # shard listings paginate, so a fat bucket burns calls per dataset; the
    # lane must stop at the budget and board what it has (2026-11-17: 600
    # starved gresearch at the d's - 43 datasets never boarded)
    lb = _load_leaderboard()
    gdm, gresearch = lb.RLDS_BUCKETS[0][0], lb.RLDS_BUCKETS[1][0]
    import kinema.rlds as rlds_mod

    class FatClient:
        def __init__(self, bucket):
            self.bucket = bucket
            self.calls = 0

        def list(self, prefix="", delimiter=""):
            self.calls += 1
            if self.bucket == gdm:
                return ([prefix + "alpha/"] if not delimiter else [prefix + "alpha/"]), []
            return [prefix + n + "/" for n in ["g%02d" % i for i in range(50)]], []

    class FatProbe:
        @classmethod
        def probe(cls, client, p):
            client.calls += 200  # paginated shard listing
            return _FakeRldsDs()

    monkeypatch.setattr(rlds_mod, "GcsClient", FatClient)
    monkeypatch.setattr(rlds_mod, "RldsDataset", FatProbe)
    rows = lb.probe_rlds_lane()
    gre = [r for r in rows if r["rlds_bucket"] == gresearch]
    assert 0 < len(gre) < 50  # stopped at the budget, kept what it probed
    assert len(gre) == 13  # 1 + 200*13 = 2601 > 2500; the tripping row boards
    assert [r["id"] for r in rows if r["rlds_bucket"] == gdm] == ["rlds/alpha"]


def test_ordered_pool_preserves_input_order():
    lb = _load_leaderboard()
    out = list(lb._ordered_pool(lambda i: i * 10, range(20), workers=8))
    assert out == [i * 10 for i in range(20)]


def test_ordered_pool_workers_1_is_strictly_serial_and_lazy():
    lb = _load_leaderboard()
    calls = []

    def fn(i):
        calls.append(i)
        return i

    stream = lb._ordered_pool(fn, range(10), workers=1)
    assert calls == []  # lazy: nothing runs before the first pull
    assert next(stream) == 0
    assert calls == [0]  # serial: one at a time, in order
    stream.close()


def test_ordered_pool_future_exception_yields_none_not_raise():
    lb = _load_leaderboard()

    def fn(i):
        if i == 2:
            raise RuntimeError("boom")
        return i

    out = list(lb._ordered_pool(fn, range(5), workers=4))
    assert out == [0, 1, None, 3, 4]


def test_top_lane_parallel_probe_keeps_order_and_early_stop(monkeypatch):
    """The top lane must board the first `limit` probed-ok candidates in
    listing order under parallel probing, exactly as the serial loop did."""
    import types

    lb = _load_leaderboard()
    monkeypatch.setattr(lb, "FLAGSHIP_ORG_SEATS", {})
    monkeypatch.setattr(lb, "FLAGSHIP_SEEDS", ())
    monkeypatch.setattr(lb, "NESTED_REPO_SEATS", {})
    monkeypatch.setattr(lb, "probe_rlds_lane", lambda workers=8: [])
    # odd candidates lack meta/info.json (probe returns None)
    listed = [types.SimpleNamespace(id=f"org/ds{i}", downloads=100 - i, likes=0)
              for i in range(12)]

    class _FakeApi:
        def list_datasets(self, **kwargs):
            return iter(listed)

        def dataset_info(self, repo_id, **kwargs):
            i = int(repo_id.rsplit("ds", 1)[1])
            sibs = [] if i % 2 else [types.SimpleNamespace(
                rfilename="meta/info.json", size=1)]
            return types.SimpleNamespace(gated=False, siblings=sibs,
                                         downloads=10, likes=0)

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeApi())
    out = lb.find_lerobot_datasets(3, 0, workers=8)
    assert [d["id"] for d in out] == ["org/ds0", "org/ds2", "org/ds4"]


def test_probe_rlds_lane_parallel_matches_serial(monkeypatch):
    """Parallel per-prefix probes board the same rows, buckets and order as
    the serial pass (consumption stays in sorted-prefix order)."""
    lb = _load_leaderboard()
    gdm, gresearch = lb.RLDS_BUCKETS[0][0], lb.RLDS_BUCKETS[1][0]
    listings = {gdm: ["alpha", "beta", "gamma"],
                gresearch: ["alpha", "beta", "gamma", "delta"]}
    probes = {gdm: ["alpha"], gresearch: ["alpha", "beta", "gamma", "delta"]}
    _patch_rlds(monkeypatch, listings=listings, probes=probes)
    serial = lb.probe_rlds_lane(workers=1)
    _patch_rlds(monkeypatch, listings=listings, probes=probes)
    parallel = lb.probe_rlds_lane(workers=8)
    assert parallel == serial
    assert [r["id"] for r in parallel] == [
        "rlds/alpha", "rlds/beta", "rlds/delta", "rlds/gamma"]
