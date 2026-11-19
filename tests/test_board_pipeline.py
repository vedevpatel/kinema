"""The board's own download path must never manufacture a defect.

The leaderboard samples big datasets: a byte-capped subset of data files, or
- when every shard is over the cap - HTTP range reads of row groups. A bug
there reads as a dataset defect: the 2026-11-19 aloha_pen_uncap_diverse
FAILs ("metadata says 300 frames but the data holds 100 rows") were a range
read that budgeted image columns it never fetched and cut the episode short.

Invariant, per path: every episode the board's sample loads holds exactly the
rows its metadata catalogues - the sample may skip episodes, never cut one -
so dropped-frames reports no length mismatch the pipeline made. Network
suite (KINEMA_E2E=1); runs in the nightly end-to-end job, a minute or two.
The datasets are ones whose metadata and data were verified consistent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from kinema.checks import ALL_CHECKS, CheckConfig
from kinema.dataset import load_dataset
from netcheck import e2e_mark

pytestmark = e2e_mark()

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

# (repo, byte cap, the path the cap forces)
CASES = [
    # every shard far over 1 MB: range reads of row groups, image columns
    # projected out (the original bug)
    ("physical-intelligence/aloha_pen_uncap_diverse", 1 << 20, "range-read"),
    # v3 packed shards under a small cap: a subset of whole shards
    ("lerobot/pusht", 4 << 20, "shard-subset"),
    # v2.1 one file per episode: a capped spread of files
    ("lerobot/aloha_static_tape", 20 << 20, "file-spread"),
]


def _lb():
    spec = importlib.util.spec_from_file_location("leaderboard", SCRIPTS / "leaderboard.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["leaderboard"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("repo,cap,path", CASES, ids=[c[2] for c in CASES])
def test_board_sample_never_cuts_an_episode(tmp_path, repo, cap, path):
    lb = _lb()
    ranged = []
    real = lb._range_read_parquet
    lb._range_read_parquet = lambda *a, **k: (ranged.append(a[1]), real(*a, **k))[1]
    dest, _ = lb.download_sample(repo, cap, tmp_path)
    # the cap really forces the path this case is about
    assert bool(ranged) == (path == "range-read"), f"{path}: range reads {ranged}"
    ctx = load_dataset(str(dest), max_episodes=250)
    catalogued = {e.index: e.length for e in ctx.episodes}
    loaded = [e for e in ctx.episode_ids() if len(ctx.rows_for(e))]
    assert loaded, f"{path}: the sample loaded no episode at all"
    cut = {e: (len(ctx.rows_for(e)), catalogued[e]) for e in loaded
           if catalogued.get(e) and len(ctx.rows_for(e)) != catalogued[e]}
    assert not cut, f"{path}: episodes cut by the board's sampling (rows, catalogued): {dict(list(cut.items())[:5])}"
    df = {c.id: c for c in ALL_CHECKS}["dropped-frames"].run(ctx, CheckConfig())
    made = [f.message for f in df.findings if "metadata disagrees" in f.message]
    assert not made, f"{path}: {made[:2]}"
