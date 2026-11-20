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

    def fake_download_sample(repo_id, cap_bytes, dest, unservable=None, revision=None):
        for rel in real_sizes:
            if rel.startswith(("meta/", "data/")):
                tgt = dest / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / rel, tgt)
        return dest, sizes

    def fake_hub_download(repo_id, rel, dest, revision=None):
        downloaded.append(rel)
        tgt = dest / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, tgt)
        return tgt

    monkeypatch.setattr(lb, "download_sample", fake_download_sample)
    monkeypatch.setattr(lb, "resolve_revision", lambda repo: "c0ffee" * 6 + "c0ff")
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

    def fake_download_sample(repo_id, cap_bytes, dest, unservable=None, revision=None):
        for rel in sizes:
            if rel.startswith(("meta/", "data/")):
                tgt = dest / rel
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / rel, tgt)
        return dest, sizes

    def fake_hub_download(repo_id, rel, dest, revision=None):
        tgt = dest / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, tgt)
        return tgt

    monkeypatch.setattr(lb, "download_sample", fake_download_sample)
    monkeypatch.setattr(lb, "resolve_revision", lambda repo: "c0ffee" * 6 + "c0ff")
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

    def fake_dl(rid, name, dest, revision=None):
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

    def fake_dl(rid, name, dest, revision=None):
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
        self.shards = [("ds-train.tfrecord-00000-of-00001", 10)]


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


def test_probe_paginates_at_sibling_cap(monkeypatch):
    import types

    lb = _load_leaderboard()
    sib = [types.SimpleNamespace(rfilename=f"data/{i}.parquet", size=2)
           for i in range(100_000)]
    api = types.SimpleNamespace(dataset_info=lambda *a, **k:
                                types.SimpleNamespace(siblings=sib, gated=False))
    tree = [types.SimpleNamespace(path="meta/info.json", size=10),
            types.SimpleNamespace(path="data/full.parquet", size=999),
            types.SimpleNamespace(path="data")]
    monkeypatch.setattr("huggingface_hub.list_repo_tree", lambda *a, **k: iter(tree))
    assert lb._probe_dataset(api, "org/big", 1, 2)["data_bytes"] == 999


def test_cap_preserves_waiver_evidence_and_input():
    import copy

    from kinema.report import report_from_dict, rescore_report_dict

    lb = _load_leaderboard()
    # First 100 overlap perfectly, but full sets don't meet the waiver bar.
    # Capping these would spuriously waive stuck-actuator and change score.
    checks = []
    for cid, eps in (("idle-segments", list(range(500))),
                     ("stuck-actuator", list(range(100)) + list(range(1000, 1500))),
                     ("video-file-pointers", list(range(300)))):
        checks.append({"id": cid, "title": cid, "summary": "x", "verdict": "WARN",
                       "findings": [{"severity": "WARN", "message": "x",
                                     "episode_index": None, "details": {"episodes": eps}}]})
    rep = {"source": "org/a", "checks": checks}
    rescore_report_dict(rep)
    entries = [{"id": "org/a", "report": rep}]
    original = copy.deepcopy(entries)
    capped = lb._cap_detail_lists(entries)
    assert entries == original
    assert len(capped[0]["report"]["checks"][0]["findings"][0]["details"]["episodes"]) == 500
    detail = capped[0]["report"]["checks"][2]["findings"][0]["details"]
    assert len(detail["episodes"]) == 100 and detail["episodes_total"] == 300
    assert lb._cap_detail_lists(capped) == capped
    assert report_from_dict(capped[0]["report"]).score == rep["score"]
    rescore_report_dict(capped[0]["report"])
    assert capped[0]["report"]["score"] == rep["score"]


@pytest.mark.parametrize("severity", ["INFO", "WARN", "FAIL"])
def test_invalidate_capped_video_findings_all_severities(severity):
    import copy

    lb = _load_leaderboard()
    report = {"source": "org/capped", "checks": [
        {"id": "video-file-pointers", "title": "video", "verdict": severity,
         "findings": [{"severity": severity, "message": "missing", "details": {}}]}]}
    entries = [{"id": "org/capped", "report": report},
               {"id": "org/real", "report": copy.deepcopy(report)}]
    untouched = copy.deepcopy(entries[1])
    assert lb.invalidate_capped_video_results(entries, {"org/capped"}, "2026-11-18T20:00:00Z") == ["org/capped"]
    assert report["checks"][0]["verdict"] == "SKIP"
    assert entries[1] == untouched
    assert not lb.invalidate_capped_video_results(entries, {"org/capped"}, "later")
    assert lb._correction_summary(entries) == ("2026-11-18T20:00:00Z", 0, 1)


def test_index_labels_snapshot_and_targeted_corrections_separately():
    lb = _load_leaderboard()
    entries = [{"id": "org/a", "downloads": 0, "status": "gated", "video_pointer_correction": {
        "at": "2026-11-18T20:00:00Z", "status": "invalidated-pending-reaudit"}}]
    html = lb.render_index(entries, "2026-11-18 02:27 UTC")
    assert "Stored audit snapshot: 2026-11-18 02:27 UTC" in html
    assert "corrections through 2026-11-18T20:00:00Z: 0 verified, 1 pending" in html


def test_time_budget_includes_discovery_before_first_submission(tmp_path, monkeypatch):
    import time

    lb = _load_leaderboard()
    clock = [0.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])

    def discover(*args, **kwargs):
        clock[0] = 11.0
        return [_d(1)]

    monkeypatch.setattr(lb, "find_lerobot_datasets", discover)
    calls = []
    monkeypatch.setattr(lb, "audit_one", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(sys, "argv", ["leaderboard", "--out", str(tmp_path), "--time-budget", "10"])
    assert lb.main() == 0
    assert calls == []


def test_nested_sample_lists_only_target_prefix(monkeypatch, tmp_path):
    import types

    lb = _load_leaderboard()
    observed = []

    def tree(*a, **k):
        observed.append(k)
        return iter([types.SimpleNamespace(path="task/meta/info.json", size=1),
                     types.SimpleNamespace(path="task/data/file-0.parquet", size=1)])

    monkeypatch.setattr("huggingface_hub.list_repo_tree", tree)
    def download(rid, name, dest, revision=None):
        p = dest / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    monkeypatch.setattr(lb, "_hub_download", download)
    _, sizes = lb.download_sample("org/huge//task", 100, tmp_path)
    assert observed[0]["path_in_repo"] == "task"
    assert observed[0]["recursive"] is True
    assert "data/file-0.parquet" in sizes


def test_metadata_only_bucket_does_not_claim_data_bearing_mirror(monkeypatch):
    import kinema.rlds as rlds

    lb = _load_leaderboard()
    first, second = lb.RLDS_BUCKETS[0][0], lb.RLDS_BUCKETS[1][0]
    _patch_rlds(monkeypatch, {first: ["alpha"], second: ["alpha"]},
                {first: ["alpha"], second: ["alpha"]})
    def probe(cls, client, prefix):
        ds = _FakeRldsDs()
        if client.bucket == first:
            ds.shards = []
            ds.data_bytes = 0
        return ds
    monkeypatch.setattr(rlds.RldsDataset, "probe", classmethod(probe))
    rows = lb.probe_rlds_lane()
    assert len(rows) == 1 and rows[0]["rlds_bucket"] == second


def test_error_rows_retry_and_depth_attempts_settle():
    lb = _load_leaderboard()
    assert lb.audit_stale({'engine': 'e', 'status': 'error', 'reason': '429'}, 'e', 250)
    row = {'engine': 'e', 'audit_max_episodes': 250,
           'report': {'truncated': True, 'episodes': 15, 'video_audited': 5}}
    assert not lb.audit_stale(row, 'e', 250)
    assert lb.audit_stale(row, 'e', 500)
    assert lb.audit_stale({**row, 'last_audit_error': '429'}, 'e', 250)


def test_discovery_omission_preserves_every_lane_and_updates_observed_metadata():
    lb = _load_leaderboard()
    old = [{'id': str(i), 'lane': lane, 'report': {'score': 70}}
           for i, lane in enumerate(('top', 'new', 'flagship', 'rlds'))]
    result = lb.reconcile_discovery(old, [{'id': '0', 'lane': 'new', 'downloads': 12}])
    assert {e['id'] for e in result} == {'0', '1', '2', '3'}
    assert result[0]['lane'] == 'new' and result[0]['downloads'] == 12
    assert result[0]['discovery_current']
    assert all(not e['discovery_current'] for e in result[1:])


def test_failed_retry_does_not_destroy_previous_evidence():
    lb = _load_leaderboard()
    old = {'id': 'org/a', 'engine': 'old', 'report': {'score': 80}}
    failed = {'id': 'org/a', 'engine': 'new', 'status': 'error', 'reason': '429'}
    kept = lb.preserve_failed_retry(old, failed)
    assert kept['report'] == old['report'] and kept['engine'] == 'old'
    assert kept['last_audit_error'] == '429'


def test_stale_omitted_row_retries_without_destroying_good_report(tmp_path, monkeypatch):
    import json
    from kinema.report import DatasetReport, report_to_dict
    lb = _load_leaderboard()
    rep = DatasetReport('org/old', 'v3.0', 10, 600, 10.)
    stored = {'id': 'org/old', 'slug': 'org--old', 'lane': 'flagship', 'engine': 'old',
              'data_bytes': 100, 'downloads': 0, 'report': report_to_dict(rep)}
    (tmp_path / 'data.json').write_text(json.dumps({'datasets': [stored]}))
    monkeypatch.setattr(lb, 'find_lerobot_datasets', lambda *a, **kw: [])
    calls = []
    def retry(*args, **kwargs):
        calls.append(args[0])
        raise lb.DatasetError('HTTP 429 temporary')
    monkeypatch.setattr(lb, 'audit_one', retry)
    monkeypatch.setattr(sys, 'argv', ['leaderboard', '--out', str(tmp_path)])
    assert lb.main() == 0
    assert calls == ['org/old']
    row = json.loads((tmp_path / 'data.json').read_text())['datasets'][0]
    assert row['report'] == stored['report']
    assert row['engine'] == 'old' and '429' in row['last_audit_error']


def test_prune_only_after_repeated_fresh_misses_in_a_healthy_lane():
    lb = _load_leaderboard()
    rows = [{'id': f't{i}', 'lane': 'top', 'report': {'score': 70}} for i in range(10)]
    listing = [{'id': f't{i}', 'lane': 'top'} for i in range(1, 10)]  # t0 dropped off
    # a cached worklist proves nothing: no miss is counted
    kept = lb.reconcile_discovery(rows, listing, fresh=False)
    assert 't0' in {e['id'] for e in kept} and 'discovery_misses' not in kept[0]
    # fresh discoveries count; pruned on the third miss in a row
    state = rows
    for n in (1, 2):
        state = lb.reconcile_discovery(state, listing, fresh=True)
        assert next(e for e in state if e['id'] == 't0')['discovery_misses'] == n
    state = lb.reconcile_discovery(state, listing, fresh=True)
    assert 't0' not in {e['id'] for e in state}
    # reappearing resets the counter
    back = lb.reconcile_discovery(lb.reconcile_discovery(rows, listing, fresh=True),
                                  listing + [{'id': 't0', 'lane': 'top'}], fresh=True)
    assert next(e for e in back if e['id'] == 't0')['discovery_misses'] == 0


def test_unhealthy_lane_listing_counts_no_misses():
    lb = _load_leaderboard()
    rows = [{'id': f'r{i}', 'lane': 'rlds', 'report': {}} for i in range(10)]
    state = rows
    for _ in range(5):  # the rlds lane came back with 2 of 10: a listing failure
        state = lb.reconcile_discovery(state, [{'id': 'r0', 'lane': 'rlds'},
                                               {'id': 'r1', 'lane': 'rlds'}], fresh=True)
    assert len(state) == 10 and all(not e.get('discovery_misses') for e in state)


def test_worklist_roundtrip_and_age(tmp_path):
    import json
    import os
    import time
    lb = _load_leaderboard()
    p = tmp_path / 'worklist.json'
    lb.save_worklist(p, [{'id': 'a/b'}])
    ds, age = lb.load_worklist(p)
    assert ds == [{'id': 'a/b'}] and 0 <= age < 0.1
    # legacy bare-list format: age from mtime
    p.write_text(json.dumps([{'id': 'c/d'}]))
    old = time.time() - 50 * 3600
    os.utime(p, (old, old))
    ds, age = lb.load_worklist(p)
    assert ds == [{'id': 'c/d'}] and 49 < age < 51


def test_hard_deadline_abandons_stuck_audits_and_still_writes(tmp_path):
    import json
    import subprocess
    import time
    script = Path(__file__).resolve().parent.parent / "scripts" / "leaderboard.py"
    out = tmp_path / "docs"
    out.mkdir()
    wl = tmp_path / "worklist.json"
    wl.write_text(json.dumps({"generated": "2099-01-01T00:00:00+00:00",
                              "datasets": [{"id": f"o/d{i}", "downloads": 1, "likes": 0,
                                            "data_bytes": 0, "lane": "top"} for i in range(4)]}))
    driver = tmp_path / "drive.py"
    driver.write_text(f'''
import sys, time, importlib.util
spec = importlib.util.spec_from_file_location("lb", {str(script)!r})
lb = importlib.util.module_from_spec(spec); spec.loader.exec_module(lb)
lb.audit_one = lambda *a, **k: time.sleep(60)  # a download that never returns in time
sys.argv = ["lb", "--worklist", {str(wl)!r}, "--out", {str(out)!r}, "--workers", "2",
            "--time-budget", "1", "--hard-deadline", "3"]
sys.exit(lb.main())
''')
    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True, timeout=45)
    took = time.monotonic() - t0
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert took < 30, f"run outlived its 3 s hard deadline: {took:.0f}s"
    assert "hard deadline: abandoned" in proc.stdout
    assert (out / "data.json").exists()


def test_errored_probes_never_count_as_misses():
    lb = _load_leaderboard()
    rows = [{'id': f't{i}', 'lane': 'top', 'report': {}} for i in range(10)]
    rows.append({'id': 'nv/big//task-1', 'lane': 'flagship', 'report': {}})
    listing = [{'id': f't{i}', 'lane': 'top'} for i in range(2, 10)]
    state = rows
    for _ in range(4):  # t0 errored every week, t1 genuinely gone, nested base errored
        state = lb.reconcile_discovery(state, listing, fresh=True, uncertain={'t0', 'nv/big'})
    ids = {e['id'] for e in state}
    assert 't0' in ids and 'nv/big//task-1' in ids  # unknown, kept
    assert 't1' not in ids  # absent from 3+ healthy fresh discoveries: pruned
    t0 = next(e for e in state if e['id'] == 't0')
    assert not t0.get('discovery_misses') and t0['discovery_uncertain']


def test_probe_retries_transient_errors_and_records_persistent_ones(monkeypatch):
    lb = _load_leaderboard()
    monkeypatch.setattr(lb, '_PROBE_RETRY_SLEEPS', (0.0, 0.0))
    lb.PROBE_ERRORS.clear()

    class Info:
        gated = False
        siblings = [type('S', (), {'rfilename': 'meta/info.json', 'size': 1})(),
                    type('S', (), {'rfilename': 'data/a.parquet', 'size': 7})()]

    class FlakyApi:
        calls = 0

        def dataset_info(self, repo_id, files_metadata=True):
            FlakyApi.calls += 1
            if FlakyApi.calls < 3:
                raise RuntimeError('429 Client Error: Too Many Requests')
            return Info()
    entry = lb._probe_dataset(FlakyApi(), 'o/flaky', 5, 1)
    assert entry and entry['data_bytes'] == 7 and FlakyApi.calls == 3
    assert 'o/flaky' not in lb.PROBE_ERRORS

    class DeadApi:
        def dataset_info(self, repo_id, files_metadata=True):
            raise RuntimeError('429 Too Many Requests')
    assert lb._probe_dataset(DeadApi(), 'o/dead', 5, 1) is None
    assert 'o/dead' in lb.PROBE_ERRORS

    class NotLeRobot:
        def dataset_info(self, repo_id, files_metadata=True):
            raise RuntimeError('404 Repository Not Found')
    assert lb._probe_dataset(NotLeRobot(), 'o/gone', 5, 1) is None
    assert 'o/gone' not in lb.PROBE_ERRORS  # a real absence is not "unknown"


# --- range-read sampling: embedded-image shards (aloha_pen_uncap_diverse) ---

def _image_shard(path, episodes=3, length=300, rg_rows=100, img_bytes=20_000):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = episodes * length
    rng = np.random.default_rng(0)
    t = pa.table({
        "episode_index": np.repeat(np.arange(episodes), length).astype("int64"),
        "frame_index": np.tile(np.arange(length), episodes).astype("int64"),
        "action": [list(map(float, r)) for r in rng.normal(size=(n, 2))],
        # incompressible pixels: the column the read projects out
        "observation.images.top": [rng.bytes(img_bytes) for _ in range(n)],
    })
    pq.write_table(t, path, row_group_size=rg_rows)


def test_range_read_budgets_only_the_columns_it_fetches(tmp_path):
    import fsspec
    import pyarrow.parquet as pq
    lb = _load_leaderboard()
    src = tmp_path / "episode_000008.parquet"
    _image_shard(src)
    # 100 rows x 20 kB of pixels per group: counting image bytes, a 4 MB
    # budget stops after the first group and cuts episode 0 to 100 rows
    lb._range_read_parquet("o/r", "data/x.parquet", tmp_path / "out", max_bytes=4_000_000,
                           lengths={0: 300, 1: 300, 2: 300},
                           fs=fsspec.filesystem("file"), path=str(src))
    out = pq.read_table(tmp_path / "out" / "data" / "x.parquet").to_pydict()
    assert "observation.images.top" not in out
    assert out["episode_index"].count(0) == 300


def test_range_read_drops_truncated_episodes_and_refuses_a_fragment(tmp_path):
    import pyarrow as pa
    lb = _load_leaderboard()
    t = pa.table({"episode_index": [0] * 300 + [1] * 120})
    kept = lb._complete_episodes(t, {0: 300, 1: 300})
    assert set(kept.column("episode_index").to_pylist()) == {0}
    with pytest.raises(lb.DatasetError):
        lb._complete_episodes(pa.table({"episode_index": [1] * 120}), {1: 300})
    # unknown lengths (no catalogue entry) are kept, not guessed at
    assert lb._complete_episodes(t, None).num_rows == 420


def test_heavy_rows_never_exceed_their_slots():
    # GR00T-sim rows queued together exhausted the runner: at most
    # HEAVY_SLOTS heavy rows in flight, light rows keep the workers busy,
    # and every row is eventually submitted
    lb = _load_leaderboard()
    rows = ([{"id": f"h{i}", "data_bytes": 5 * 1024 ** 3} for i in range(8)]
            + [{"id": f"l{i}", "data_bytes": 10} for i in range(8)])
    it, deferred, in_flight, submitted = iter(rows), [], [], []
    while True:
        while len(in_flight) < 8:  # fill the pool
            d = lb.next_row(it, deferred, in_flight)
            if d is None:
                break
            in_flight.append(d)
            submitted.append(d["id"])
            assert sum(lb.is_heavy(x) for x in in_flight) <= lb.HEAVY_SLOTS
        if not in_flight:
            break
        in_flight.pop(0)  # oldest finishes
    assert sorted(submitted) == sorted(d["id"] for d in rows)
    assert submitted[:8] == ["h0", "h1", "l0", "l1", "l2", "l3", "l4", "l5"]


def test_unservable_hub_file_is_named(tmp_path, monkeypatch):
    # arekborucki/bridge_orig_lerobot: Xet CAS 404s on one file and the
    # classic endpoint fails too - the error must say WHICH file
    import huggingface_hub
    lb = _load_leaderboard()

    def xet_404(*a, **k):
        raise RuntimeError("Task error: File reconstruction error: CAS Client Error: 404 Not Found")

    def classic_fails(*a, **k):
        raise OSError("HTTP Error 404")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", xet_404)
    monkeypatch.setattr(lb, "_classic_resolve_download", classic_fails)
    with pytest.raises(lb.DatasetError, match="meta/episodes.jsonl"):
        lb._hub_download("o/r", "meta/episodes.jsonl", tmp_path)


def test_one_unservable_data_file_does_not_abort_the_sample(tmp_path, monkeypatch):
    import types

    import huggingface_hub
    lb = _load_leaderboard()
    files = ["meta/info.json"] + [f"data/chunk-000/episode_{i:06d}.parquet" for i in range(3)]
    monkeypatch.setattr(huggingface_hub, "list_repo_tree", lambda *a, **k: [
        types.SimpleNamespace(path=p, size=10) for p in files])
    got = []

    def fake_dl(repo, name, dest, revision=None):
        if name.endswith("episode_000000.parquet"):
            raise lb.DatasetError(f"the Hub cannot serve {name}: Xet storage has no data for it")
        got.append(name)
    monkeypatch.setattr(lb, "_hub_download", fake_dl)
    unservable: list = []
    lb.download_sample("o/r", 10_000, tmp_path, unservable)
    assert [n for n, _ in unservable] == ["data/chunk-000/episode_000000.parquet"]
    assert "data/chunk-000/episode_000002.parquet" in got
    # nothing servable at all is still a failed load
    monkeypatch.setattr(lb, "_hub_download", lambda r, n, d, revision=None: (_ for _ in ()).throw(
        lb.DatasetError(f"the Hub cannot serve {n}")) if n.startswith("data/") else None)
    with pytest.raises(lb.DatasetError, match="none of the sampled data files"):
        lb.download_sample("o/r", 10_000, tmp_path, [])


# --- robustness gate ----------------------------------------------------------

def _gate_report(lb, ctx, checks):
    from kinema.checks import CheckConfig
    from kinema.report import DatasetReport
    rep = DatasetReport(source="t", format_version="v3.0", episodes=ctx.total_episodes,
                        frames=ctx.row_count, fps=ctx.fps)
    rep.results = [c.run(ctx, CheckConfig()) for c in checks]
    return rep


class _AbsoluteThresholdCheck:
    """A deliberately representation-bound check: WARNs when any state value
    exceeds 0.5 in absolute units - exactly the bug class the probes catch."""
    id = "fake-absolute"

    def run(self, ctx, cfg):
        import numpy as np

        from kinema.checks.base import CheckResult, Finding, Severity
        res = CheckResult(self.id, "fake")
        if np.nanmax(np.abs(ctx.state)) > 0.5:
            res.findings.append(Finding(self.id, Severity.WARN, "big values", details={}))
        return res


def test_gate_demotes_a_verdict_that_moves_under_a_probe(tmp_path, monkeypatch):
    from kinema.dataset import load_dataset
    from kinema.fixtures.synth import make_dataset
    lb = _load_leaderboard()
    fake = _AbsoluteThresholdCheck()
    monkeypatch.setattr(lb, "ALL_CHECKS", [*lb.ALL_CHECKS, fake])
    ctx = load_dataset(str(make_dataset(tmp_path / "f", episodes=12)))
    rep = _gate_report(lb, ctx, [fake])
    assert rep.results[0].verdict.name == "WARN"
    gate = lb.TransitionGate(cap=10)
    lb.robustness_gate(ctx, rep, prior={}, gate=gate, repo_id="o/r")
    r = rep.results[0]
    assert r.verdict.name == "INFO" and r.details["robustness"]["unstable_under"] == "unit-milli"
    assert "not robust" in r.findings[0].message and gate.unstable[0]["check"] == "fake-absolute"


def test_gate_publishes_a_robust_new_verdict_and_ignores_old_ones(tmp_path):
    from kinema.checks import ALL_CHECKS
    from kinema.dataset import load_dataset
    from kinema.fixtures.synth import make_dataset
    lb = _load_leaderboard()
    ctx = load_dataset(str(make_dataset(tmp_path / "i", episodes=24, defects=["idle-segments"])))
    idle = next(c for c in ALL_CHECKS if c.id == "idle-segments")
    rep = _gate_report(lb, ctx, [idle])
    gate = lb.TransitionGate(cap=10)
    lb.robustness_gate(ctx, rep, prior={"idle-segments": "PASS"}, gate=gate)
    r = rep.results[0]
    assert r.verdict.name == "WARN" and set(r.details["robustness"]["probed"]) >= {"unit-milli", "order-reversed"}
    assert gate.admitted == {"idle-segments": 1}
    # already WARN on the board: not a transition, not touched
    rep2 = _gate_report(lb, ctx, [idle])
    lb.robustness_gate(ctx, rep2, prior={"idle-segments": "WARN"}, gate=gate)
    assert "robustness" not in (rep2.results[0].details or {}) and gate.admitted == {"idle-segments": 1}


def test_gate_cap_holds_the_overflow_as_info(tmp_path):
    from kinema.checks import ALL_CHECKS
    from kinema.dataset import load_dataset
    from kinema.fixtures.synth import make_dataset
    lb = _load_leaderboard()
    ctx = load_dataset(str(make_dataset(tmp_path / "c", episodes=24, defects=["idle-segments"])))
    idle = next(c for c in ALL_CHECKS if c.id == "idle-segments")
    gate = lb.TransitionGate(cap=1)
    reps = [_gate_report(lb, ctx, [idle]) for _ in range(2)]
    for i, rep in enumerate(reps):
        lb.robustness_gate(ctx, rep, prior=None, gate=gate, repo_id=f"o/r{i}", probe=False)
    assert reps[0].results[0].verdict.name == "WARN"
    held = reps[1].results[0]
    assert held.verdict.name == "INFO" and held.details["robustness"]["held"]
    assert gate.held == [{"id": "o/r1", "check": "idle-segments", "was": "WARN"}]


def test_half_sample_probe_only_on_sampled_loads(tmp_path):
    from kinema.dataset import load_dataset
    from kinema.fixtures.synth import make_dataset
    lb = _load_leaderboard()
    root = make_dataset(tmp_path / "h", episodes=24)
    full = load_dataset(str(root))
    sampled = load_dataset(str(root), max_episodes=12)
    assert "half-sample" not in [n for n, _ in lb._nightly_probes(full, set())]
    assert "half-sample" in [n for n, _ in lb._nightly_probes(sampled, set())]
