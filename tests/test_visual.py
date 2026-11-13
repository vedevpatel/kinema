"""visual-quality on a real v3 shared-file video: episodes are located by
their [from_timestamp, to_timestamp) segment, and dark-but-well-lit frames
are not called underexposed. Needs the `video` extra (PyAV)."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

av = pytest.importorskip("av")

from kinema.checks import ALL_CHECKS, CheckConfig, Severity  # noqa: E402
from kinema.dataset import load_dataset  # noqa: E402
from kinema.fixtures.synth import make_dataset  # noqa: E402

CHECK = {c.id: c for c in ALL_CHECKS}["visual-quality"]
FPS, LEN, EPISODES, SIZE = 10, 60, 24, 64
DARK, BLURRY = set(range(12, 18)), set(range(18, 24))


def _frame(ep: int) -> np.ndarray:
    # black background with a bright, sharp checkerboard patch: mean ~40,
    # highlights ~255 - a dark scene, not an underexposed one
    img = np.zeros((SIZE, SIZE), dtype=np.float64)
    yy, xx = np.mgrid[0:32, 0:32]
    img[16:48, 16:48] = np.where(((yy // 8) + (xx // 8)) % 2, 255.0, 60.0)
    if ep in DARK:
        img *= 0.2  # genuinely underexposed: nothing gets bright
    if ep in BLURRY:
        # out of focus: no edges anywhere, just a smooth falloff
        r2 = (np.mgrid[0:SIZE, 0:SIZE] - SIZE / 2) ** 2
        img = 200.0 * np.exp(-(r2[0] + r2[1]) / (2 * 14.0**2))
    return img.astype(np.uint8)


@pytest.fixture()
def v3_video_dataset(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=EPISODES)
    cam = "observation.images.top"
    vid = root / "videos" / cam / "chunk-000" / "file-000.mp4"
    vid.parent.mkdir(parents=True)
    with av.open(str(vid), "w") as out:
        st = out.add_stream("mpeg4", rate=FPS)
        st.width = st.height = SIZE
        st.pix_fmt = "yuv420p"
        st.bit_rate = 4_000_000
        for ep in range(EPISODES):
            g = _frame(ep)
            for _ in range(LEN):
                rgb = np.repeat(g[:, :, None], 3, axis=2)
                for pkt in st.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                    out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    mf = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m = pq.read_table(mf).to_pydict()
    m[f"videos/{cam}/chunk_index"] = [0] * EPISODES
    m[f"videos/{cam}/file_index"] = [0] * EPISODES
    m[f"videos/{cam}/from_timestamp"] = [e * LEN / FPS for e in m["episode_index"]]
    m[f"videos/{cam}/to_timestamp"] = [(e + 1) * LEN / FPS for e in m["episode_index"]]
    pq.write_table(pa.table(m), mf)
    info = json.loads((root / "meta" / "info.json").read_text())
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    info["features"][cam] = {"dtype": "video", "shape": [SIZE, SIZE, 3]}
    (root / "meta" / "info.json").write_text(json.dumps(info))
    return root


def test_v3_segments_attributed_and_graded(v3_video_dataset):
    res = CHECK.run(load_dataset(str(v3_video_dataset)), CheckConfig(video=True))
    assert not res.skipped, res.skip_reason
    by_kind = {f.details["kind"]: f for f in res.findings}
    assert set(by_kind["underexposed"].details["episodes"]) == DARK
    assert set(by_kind["blurry"].details["episodes"]) == BLURRY
    assert all(f.severity == Severity.WARN for f in by_kind.values())  # 6/24 = 25% of episodes


def test_sampled_load_only_reads_its_episodes(v3_video_dataset):
    res = CHECK.run(load_dataset(str(v3_video_dataset), max_episodes=4), CheckConfig(video=True))
    flagged = {e for f in res.findings for e in f.details.get("episodes", [])}
    loaded = set(load_dataset(str(v3_video_dataset), max_episodes=4).episode_ids())
    assert flagged <= loaded
