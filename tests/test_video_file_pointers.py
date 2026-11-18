"""video-file-pointers: metadata naming video files the repo does not hold.

The video sibling of episode-file-pointers: an export interrupted mid-upload
(or files pruned by hand) leaves meta/episodes pointing at MP4s that are not
in the repo. The episode rows all load, so every other check sees a healthy
dataset; the failure only surfaces when training decodes video. Verified live
against OpenMOSS-Team/robodojo_lerobot_v3.0 (100 of 593 referenced files
missing, 1769/3500 episodes affected) with the twin export
robodojo-lerobot-v3.0 complete.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.dataset import DatasetContext, load_dataset
from kinema.fixtures.synth import make_dataset

CHECK = {c.id: c for c in ALL_CHECKS}["video-file-pointers"]


def _repo_files(root) -> list[str]:
    """The file listing a Hub audit would pass in: everything under the root."""
    return sorted(p.relative_to(root).as_posix() for p in Path(root).rglob("*") if p.is_file())


def _run(root, repo_files="disk"):
    ctx = load_dataset(str(root))
    if repo_files == "disk":
        ctx.repo_files = _repo_files(root)
    return CHECK.run(ctx, CheckConfig())


def test_clean_video_dataset_passes(tmp_path):
    root = make_dataset(tmp_path / "clean", episodes=10, cameras=["cam_high", "cam_left"])
    res = _run(root)
    assert res.verdict == Severity.OK, res.summary
    # 10 episodes packed 5 per MP4 = 2 files per camera
    assert "all 4 metadata-referenced video files exist" in res.summary


def test_missing_file_at_half_the_dataset_fails(tmp_path):
    root = make_dataset(tmp_path / "broken", episodes=10,
                        cameras=["cam_high", "cam_left"], defects=["missing-video-file"])
    res = _run(root)
    assert res.verdict == Severity.FAIL, res.summary
    hit = res.findings[0]
    assert hit.details["camera"] == "cam_left"
    assert hit.details["missing_files"] == ["videos/cam_left/chunk-000/file-001.mp4"]
    assert hit.details["episodes"] == [5, 6, 7, 8, 9]
    assert "5/10 episodes affected" in hit.message
    assert res.details["missing"] == 1
    assert res.details["episodes_affected"] == [5, 6, 7, 8, 9]


def test_missing_file_at_a_quarter_warns(tmp_path):
    root = make_dataset(tmp_path / "warn", episodes=20,
                        cameras=["cam_high"], defects=["missing-video-file"])
    res = _run(root)
    assert res.verdict == Severity.WARN, res.summary
    assert "5/20 episodes affected" in res.findings[0].message


def test_missing_file_under_ten_percent_is_info(tmp_path):
    root = make_dataset(tmp_path / "info", episodes=60,
                        cameras=["cam_high"], defects=["missing-video-file"])
    res = _run(root)
    assert res.verdict == Severity.INFO, res.summary
    assert "5/60 episodes affected" in res.findings[0].message


def test_local_disk_glob_without_a_repo_listing(tmp_path):
    # a local audit has no repo listing: the files on disk are the evidence
    root = make_dataset(tmp_path / "local", episodes=10,
                        cameras=["cam_high"], defects=["missing-video-file"])
    res = _run(root, repo_files=None)
    assert res.verdict == Severity.FAIL, res.summary
    assert "against the files on disk" in res.summary


def test_local_copy_without_the_videos_tree_skips(tmp_path):
    # data/ + meta/ only (a common partial local copy): the pointers are not
    # evidence about videos the copy never fetched
    import shutil

    root = make_dataset(tmp_path / "novideos", episodes=10, cameras=["cam_high"])
    shutil.rmtree(root / "videos")
    res = _run(root, repo_files=None)
    assert res.skipped and "no episode videos fetched" in res.skip_reason


def test_partial_video_fetch_skips_without_a_listing(tmp_path):
    root = make_dataset(tmp_path / "partial", episodes=10,
                        cameras=["cam_high"], defects=["missing-video-file"])
    ctx = load_dataset(str(root))
    ctx.video_partial = True  # board-style targeted fetch: disk says nothing
    res = CHECK.run(ctx, CheckConfig())
    assert res.skipped and "partially fetched" in res.skip_reason


def _as_hub(monkeypatch, root, listing="tree"):
    """Resolve any source to this root as if it were a Hub fetch. listing:
    "tree" = the repo listing mirrors the files on disk; "fail" = the
    listing call raises (gated/offline)."""
    import types

    monkeypatch.setattr("kinema.dataset._resolve_source",
                        lambda source, video=False, max_episodes=None: (Path(root), "owner/name", None))
    if listing == "tree":
        files = sorted(p.relative_to(root).as_posix()
                       for p in Path(root).rglob("*") if p.is_file())
        tree = [types.SimpleNamespace(path=f, size=1) for f in files]
        monkeypatch.setattr("huggingface_hub.list_repo_tree", lambda *a, **k: iter(tree))
    else:
        def boom(*a, **k):
            raise OSError("offline")
        monkeypatch.setattr("huggingface_hub.list_repo_tree", boom)


def test_hub_load_without_video_verifies_against_the_repo_listing(monkeypatch, tmp_path):
    # regression pin for the no-video false FAIL: a Hub audit that fetched no
    # videos verifies pointers against the repo's own listing, not the empty
    # disk - and still FAILs a file the listing really lacks
    root = make_dataset(tmp_path / "hubish", episodes=10,
                        cameras=["cam_high"], defects=["missing-video-file"])
    _as_hub(monkeypatch, root)
    ctx = load_dataset("owner/name")
    assert ctx.video_partial is True
    assert ctx.repo_files is not None
    res = CHECK.run(ctx, CheckConfig())
    assert res.verdict == Severity.FAIL, res.summary
    assert "against the repo file listing" in res.summary


def test_hub_load_without_a_listing_skips(monkeypatch, tmp_path):
    # listing unavailable (gated/offline): absence must not read as defect
    root = make_dataset(tmp_path / "huboff", episodes=10,
                        cameras=["cam_high"], defects=["missing-video-file"])
    _as_hub(monkeypatch, root, listing="fail")
    ctx = load_dataset("owner/name")
    assert ctx.video_partial is True and ctx.repo_files is None
    res = CHECK.run(ctx, CheckConfig())
    assert res.skipped and "partially fetched" in res.skip_reason


def test_hub_sampled_load_marks_partial_even_with_video(monkeypatch, tmp_path):
    root = make_dataset(tmp_path / "hubsample", episodes=10,
                        cameras=["cam_high"], defects=["missing-video-file"])
    _as_hub(monkeypatch, root)
    ctx = load_dataset("owner/name", video=True, max_episodes=5)
    assert ctx.video_partial is True


def test_hub_full_video_load_stays_disk_based(monkeypatch, tmp_path):
    root = make_dataset(tmp_path / "hubfull", episodes=10, cameras=["cam_high"])
    _as_hub(monkeypatch, root)
    ctx = load_dataset("owner/name", video=True)
    assert ctx.video_partial is False
    res = CHECK.run(ctx, CheckConfig())
    assert res.verdict == Severity.OK, res.summary


def test_no_cameras_skips(tmp_path):
    res = _run(make_dataset(tmp_path / "novideo", episodes=8))
    assert res.skipped and "no video cameras" in res.skip_reason


def _v2_stub(root, repo_files) -> DatasetContext:
    n = np.zeros((4, 2))
    return DatasetContext(
        root=Path(root),
        repo_id=None,
        format_version="v2.1",
        info={
            "chunks_size": 1000,
            "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
            "features": {"observation.images.top": {"dtype": "video", "shape": [480, 640, 3]}},
        },
        stats={},
        fps=10.0,
        episodes=[],
        episode_index=np.zeros(4, dtype=np.int64),
        frame_index=np.zeros(4, dtype=np.int64),
        timestamp=np.zeros(4),
        state=n,
        action=n,
        task_index=None,
        repo_files=repo_files,
    )


def test_v21_template_paths_checked_against_listing(tmp_path):
    root = tmp_path / "v2"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({}))
    with open(root / "meta" / "episodes.jsonl", "w") as f:
        for ep in range(4):
            f.write(json.dumps({"episode_index": ep, "length": 60}) + "\n")
    files = [f"videos/observation.images.top/chunk-000/episode_{i:06d}.mp4" for i in (0, 1, 2)]
    res = CHECK.run(_v2_stub(root, files), CheckConfig())
    assert res.verdict == Severity.WARN, res.summary  # 1 of 4 episodes = 25%
    assert res.findings[0].details["missing_files"] == [
        "videos/observation.images.top/chunk-000/episode_000003.mp4"]
    assert res.findings[0].details["episodes"] == [3]
    res_ok = CHECK.run(_v2_stub(root, files + [
        "videos/observation.images.top/chunk-000/episode_000003.mp4"]), CheckConfig())
    assert res_ok.verdict == Severity.OK, res_ok.summary


def test_hub_root_listing_paths_are_not_trimmed(monkeypatch, tmp_path):
    root = make_dataset(tmp_path / "goodhub", episodes=10, cameras=["cam_high"])
    _as_hub(monkeypatch, root)
    import shutil

    shutil.rmtree(root / "videos")
    ctx = load_dataset("owner/name")
    assert "meta/info.json" in ctx.repo_files
    assert CHECK.run(ctx, CheckConfig()).verdict == Severity.OK


def test_hub_nested_listing_keeps_revision_and_relative_paths(monkeypatch, tmp_path):
    import types

    root = make_dataset(tmp_path / "nested", episodes=10, cameras=["cam_high"])
    monkeypatch.setattr("kinema.dataset._resolve_source", lambda *a, **k:
                        (root, "owner/name//task", "pinned-sha"))
    files = [types.SimpleNamespace(path="task/" + p, size=1) for p in _repo_files(root)]
    files.append(types.SimpleNamespace(path="other/else.mp4", size=1))
    calls = []

    def tree(*args, **kwargs):
        calls.append((args, kwargs))
        return iter(files)

    monkeypatch.setattr("huggingface_hub.list_repo_tree", tree)
    ctx = load_dataset("owner/name//task@pinned-sha")
    assert calls[0][1]["revision"] == "pinned-sha"
    assert "meta/info.json" in ctx.repo_files
    assert not any(p.startswith("other/") for p in ctx.repo_files)
    assert CHECK.run(ctx, CheckConfig()).verdict == Severity.OK
