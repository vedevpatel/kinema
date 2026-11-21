"""Structural regression pack: real metadata defects at pinned revisions,
checked offline on every push (tests/data/structural/, README.md there).

Each case is a trimmed extract of a real, redistributable dataset. Its
CASE.json holds the ground truth, computed by build_pack.py from the Hub
files at that revision WITHOUT kinema's check code. Here the detectors run
on the extract and must report exactly that truth - the same episodes,
shards, cameras and files, no more and no fewer:

- episode-file-pointers: every catalogued episode whose declared shard does
  not hold its rows, with where the rows actually are among the packed shards
- camera-coverage: every episode missing a camera's video segment
- video-file-pointers: every referenced video file absent from the listing
- dropped-frames: episodes catalogued more than once

The clean controls must raise none of these. The pack tests detector
correctness on real structure; it says nothing about training outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.checks.metadata import _POINTER_LIST_CAP as CAP
from kinema.dataset import load_dataset

PACK = Path(__file__).resolve().parent / "data" / "structural"
CASES = sorted(p.parent.name for p in PACK.glob("*/CASE.json"))
C = {c.id: c for c in ALL_CHECKS}
CFG = CheckConfig()
REDISTRIBUTABLE = {"mit", "apache-2.0", "cc-by-4.0", "cc0-1.0", "cc-by-sa-4.0", "odc-by", "bsd-3-clause"}
# documented policy (docs/checks.md, camera-coverage): WARN at >= 10% of episodes
COVERAGE_WARN_SHARE = 0.10


@pytest.fixture(scope="module")
def case():
    cache: dict = {}

    def get(name: str):
        if name not in cache:
            root = PACK / name
            ctx = load_dataset(str(root))
            # as on the board: the repo listing at the revision, videos not fetched
            ctx.repo_files = [ln.split("\t")[0] for ln in (root / "listing.txt").read_text().splitlines()]
            ctx.video_partial = True
            cache[name] = (ctx, json.loads((root / "CASE.json").read_text()))
        return cache[name]
    return get


def test_the_pack_is_there():
    assert len(CASES) >= 6, CASES


@pytest.mark.parametrize("name", CASES)
def test_provenance_and_license_are_recorded(name):
    doc = json.loads((PACK / name / "CASE.json").read_text())
    assert doc["source"] and len(doc["revision"]) == 40
    assert (doc["license"] or "").lower() in REDISTRIBUTABLE, doc["license"]
    assert name in (PACK / "README.md").read_text()


@pytest.mark.parametrize("name", CASES)
def test_episode_pointers_match_ground_truth(case, name):
    ctx, doc = case(name)
    truth = doc["truth"]
    skip = set(truth.get("conflicting_duplicates", []))
    res = C["episode-file-pointers"].run(ctx, CFG)
    want = {(p["episode"], p["declared"], p["actual"]) for p in truth["pointers"]}
    # every broken episode, exactly (details carry the full list) ...
    broken = set((res.details or {}).get("broken", [])) - skip
    assert broken == {e for e, _, _ in want}, (sorted(broken ^ {e for e, _, _ in want})[:8])
    # ... and each listed finding (capped) names the right declared and actual shard
    listed = {(f.episode_index, f.details["declared"], f.details["actual"]) for f in res.findings
              if f.severity >= Severity.WARN and f.episode_index not in skip}
    assert listed <= want, sorted(listed - want)[:5]
    assert len(listed) == min(len(want), CAP - len(skip & set((res.details or {}).get("broken", []))))
    if want:
        assert res.verdict == Severity.WARN
    else:
        assert res.verdict <= Severity.INFO, res.summary


@pytest.mark.parametrize("name", CASES)
def test_camera_segments_match_ground_truth(case, name):
    ctx, doc = case(name)
    res = C["camera-coverage"].run(ctx, CFG)
    got = {f.details["camera"]: sorted(f.details["episodes"]) for f in res.findings
           if "camera" in (f.details or {}) and f.details.get("episodes")}
    want = {cam: sorted(eps) for cam, eps in doc["truth"]["segments"].items()}
    assert got == want
    n = len(ctx.episode_ids())
    worst = max((len(e) / n for e in want.values()), default=0.0)
    expected = Severity.WARN if worst >= COVERAGE_WARN_SHARE else (Severity.INFO if want else Severity.OK)
    assert res.verdict == expected or (not want and res.verdict <= Severity.INFO), (res.verdict, res.summary)


@pytest.mark.parametrize("name", CASES)
def test_video_files_match_ground_truth(case, name):
    ctx, doc = case(name)
    res = C["video-file-pointers"].run(ctx, CFG)
    got = {f.details["camera"]: sorted(f.details.get("missing_files", [])) for f in res.findings
           if "camera" in (f.details or {})}
    want = {cam: v["missing_files"][:20] for cam, v in doc["truth"]["video_files"].items()}
    assert got == want
    if not want:
        assert res.verdict <= Severity.INFO, res.summary


@pytest.mark.parametrize("name", CASES)
def test_duplicate_catalogue_entries_are_reported(case, name):
    ctx, doc = case(name)
    dup = doc["truth"].get("duplicate_catalogue", [])
    res = C["dropped-frames"].run(ctx, CFG)
    reported = {int(k) for f in res.findings for k in ((f.details or {}).get("duplicates") or {})}
    assert reported == set(dup)
    if dup:
        assert res.verdict == Severity.FAIL


def test_clean_controls_raise_nothing_structural(case):
    for name in (n for n in CASES if n.endswith("_clean")):
        ctx, _ = case(name)
        for cid in ("episode-file-pointers", "camera-coverage", "video-file-pointers"):
            res = C[cid].run(ctx, CFG)
            assert not [f for f in res.findings if f.severity >= Severity.WARN], (name, cid, res.summary)


# --- wording: "missing data" only when every data shard was read --------------

@pytest.mark.parametrize("name,claim", [
    # rows live in file-002 / file-027, which the extract (like the board's
    # sample) never fetched: missing cannot be told from misplaced
    ("bitrobot_pointer_rows_unpacked", "fetched"),
    ("vlabench_pointer_rollover", "fetched"),
    # all three of the dataset's shards are read and the rows are in none
    ("mendelbeans_pointer_no_rows", "missing"),
])
def test_no_rows_wording_follows_what_was_read(case, name, claim):
    ctx, _ = case(name)
    summary = C["episode-file-pointers"].run(ctx, CFG).summary
    if claim == "missing":
        assert "missing data, not a pointer bug" in summary, summary
    else:
        assert "no rows in the fetched shards" in summary and "missing data" not in summary, summary


def test_no_rows_wording_on_a_local_dataset(tmp_path):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from kinema.fixtures.synth import make_dataset
    root = make_dataset(tmp_path / "d", episodes=12, shards=2)
    f = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f)
    pq.write_table(t.filter(pc.not_equal(t.column("episode_index"), 0)), f)  # episode 0's rows gone
    ctx = load_dataset(str(root))
    # a local audit holds every shard on disk: the rows are in none of them
    assert "missing data" in C["episode-file-pointers"].run(ctx, CFG).summary
    # the same files seen as a sample of a bigger repo: it cannot say so
    ctx.repo_files = [*(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()),
                      "data/chunk-000/file-007.parquet"]
    summary = C["episode-file-pointers"].run(ctx, CFG).summary
    assert "no rows in the fetched shards" in summary and "missing data" not in summary
