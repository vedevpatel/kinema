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


def test_partial_video_fetch_skips_without_a_listing(tmp_path):
    root = make_dataset(tmp_path / "partial", episodes=10,
                        cameras=["cam_high"], defects=["missing-video-file"])
    ctx = load_dataset(str(root))
    ctx.video_partial = True  # board-style targeted fetch: disk says nothing
    res = CHECK.run(ctx, CheckConfig())
    assert res.skipped and "partially fetched" in res.skip_reason


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
