"""Health-score model tiered-v1.1: pins the tier table, the per-check weight
table with its basis labels, the per-check (never per-finding) rule, the
evidence-gated correlated-check waiver, and the recompute-from-stored path
the leaderboard migrates through."""

from __future__ import annotations

import copy
import importlib.util
import json
import random
from pathlib import Path

import pytest

from kinema.checks import ALL_CHECKS
from kinema.checks.base import CheckResult, Finding, Severity
from kinema.report import (
    SCORE_MODEL,
    WAIVER_EDGES,
    WARN_TIERS,
    DatasetReport,
    flagged_episodes,
    harm_tier,
    render_markdown,
    report_from_dict,
    report_to_dict,
    rescore_report_dict,
    score_from_verdicts,
    warn_weight,
)

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "leaderboard.py"

# legacy-v0, the hand-tuned table tiered-v1 replaced (default 8 for unlisted)
LEGACY_V0 = {
    "stats-drift": 10,
    "nan-and-normalization": 10,
    "deployment-readiness": 10,
    "timestamp-grid": 10,
    "dropped-frames": 8,
    "calibration-drift": 6,
    "idle-segments": 6,
    "stuck-actuator": 6,
    "action-saturation": 6,
    "visual-quality": 6,
    "duplicate-and-leakage": 4,
    "outlier-episodes": 3,
}

TIERED_V1 = {
    "idle-segments": (10, "measured"),
    "calibration-drift": (10, "measured"),
    "duplicate-and-leakage": (2, "measured"),
    "stats-drift": (10, "placeholder"),
    "nan-and-normalization": (10, "placeholder"),
    "deployment-readiness": (10, "placeholder"),
    "timestamp-grid": (10, "placeholder"),
    "dropped-frames": (8, "placeholder"),
    "stuck-actuator": (6, "placeholder"),
    "action-saturation": (6, "placeholder"),
    "visual-quality": (6, "placeholder"),
    "outlier-episodes": (3, "placeholder"),
    "task-strings": (8, "placeholder"),
    "reward-done": (8, "placeholder"),
}


def test_model_label():
    assert SCORE_MODEL == "tiered-v1.1"


def test_tier_table_is_pinned():
    assert WARN_TIERS == (
        ("A", "training killer", 100.0, 10),
        ("B", "skews learning", 10.0, 6),
        ("C", "review list", 0.0, 2),
    )


@pytest.mark.parametrize(
    "pct, tier",
    [(881.0, "A"), (347.0, "A"), (100.0, "A"), (99.9, "B"), (10.0, "B"),
     (9.99, "C"), (0.1, "C"), (0.0, "C"), (-3.0, "C")],
)
def test_band_edges(pct, tier):
    assert harm_tier(pct)[0] == tier
    assert harm_tier(pct)[2] == {"A": 10, "B": 6, "C": 2}[tier]


def test_weight_table_and_basis_are_pinned():
    for cid, expected in TIERED_V1.items():
        assert warn_weight(cid) == expected, cid
    assert warn_weight("some-future-check") == (8, "placeholder")


def test_only_the_three_measured_weights_moved():
    moved = {cid for cid, w in LEGACY_V0.items() if warn_weight(cid)[0] != w}
    assert moved == {"idle-segments", "calibration-drift", "duplicate-and-leakage"}
    measured = {c.id for c in ALL_CHECKS if warn_weight(c.id)[1] == "measured"}
    assert measured == moved


def test_fail_cost_and_floor():
    assert score_from_verdicts([("idle-segments", "FAIL")]) == 75
    assert score_from_verdicts([("duplicate-and-leakage", "FAIL")]) == 75
    assert score_from_verdicts([(f"c{i}", "FAIL") for i in range(5)]) == 0
    assert score_from_verdicts([("x", v) for v in ("PASS", "INFO", "SKIP")]) == 100


def _report(check_id: str, n_findings: int, sev: Severity = Severity.WARN) -> DatasetReport:
    r = CheckResult(check_id=check_id, title=check_id)
    for i in range(n_findings):
        r.findings.append(Finding(check_id=check_id, severity=sev, message=f"ep {i}",
                                  episode_index=i))
    rep = DatasetReport(source="t", format_version="v3.0", episodes=40, frames=400, fps=10.0)
    rep.results.append(r)
    return rep


def test_scoring_is_per_check_not_per_finding():
    assert _report("idle-segments", 1).score == _report("idle-segments", 40).score == 90
    assert _report("duplicate-and-leakage", 40).score == 98
    assert _report("timestamp-grid", 40, Severity.FAIL).score == 75


def test_report_json_labels_model_and_weights():
    d = report_to_dict(_report("duplicate-and-leakage", 3))
    assert d["score_model"] == "tiered-v1.1"
    assert d["score"] == 98
    (c,) = d["checks"]
    assert (c["weight"], c["weight_basis"], c["charge"]) == (2, "measured", 2)
    assert "waived_by" not in c and "waiver_evidence" not in c
    assert "tiered-v1.1" in render_markdown(_report("idle-segments", 1))


def _legacy_dict(verdicts: dict[str, str], score: int) -> dict:
    """A stored legacy-v0 report: no score_model, no per-check weights."""
    return {
        "source": "org/ds", "format_version": "v3.0", "episodes": 10, "frames": 100,
        "fps": 10.0, "truncated": False, "score": score, "verdict": "WARN",
        "checks": [
            {"id": cid, "title": cid, "verdict": v, "summary": "", "skipped": v == "SKIP",
             "skip_reason": "", "details": {},
             "findings": [] if v in ("PASS", "SKIP") else
             [{"severity": v, "message": "m", "episode_index": i, "details": {}}
              for i in range(7)]}
            for cid, v in verdicts.items()
        ],
    }


def test_rescore_uses_stored_verdicts_and_is_idempotent():
    d = _legacy_dict({"idle-segments": "WARN", "duplicate-and-leakage": "WARN",
                      "stats-drift": "PASS", "visual-quality": "SKIP"}, score=90)
    rescore_report_dict(d)
    assert d["score"] == 100 - 10 - 2
    assert d["score_model"] == "tiered-v1.1"
    assert [(c["weight"], c["weight_basis"]) for c in d["checks"]] == [
        (10, "measured"), (2, "measured"), (10, "placeholder"), (6, "placeholder"),
    ]
    again = rescore_report_dict(copy.deepcopy(d))
    assert again == d
    # the round-tripped report (what dataset pages render) agrees with it
    assert report_from_dict(d).score == d["score"]


def _load_leaderboard():
    spec = importlib.util.spec_from_file_location("leaderboard", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_leaderboard_rescore_site(tmp_path):
    lb = _load_leaderboard()
    out = tmp_path / "docs"
    (out / "dataset").mkdir(parents=True)
    idle = _legacy_dict({"idle-segments": "WARN"}, score=94)
    dup = _legacy_dict({"duplicate-and-leakage": "WARN"}, score=96)
    dup["source"] = "org/dup"
    entries = [
        {"id": "org/ds", "slug": "org--ds", "downloads": 10, "engine": "e0", "report": idle},
        {"id": "org/dup", "slug": "org--dup", "downloads": 5, "engine": "e0", "report": dup},
        {"id": "org/gated", "downloads": 1, "status": "gated", "reason": "GatedRepoError"},
    ]
    (out / "data.json").write_text(json.dumps({"generated": "2026-11-14 00:01 UTC",
                                               "datasets": entries}))
    stamp = "audited 2026-11-13 22:10 UTC by"
    (out / "dataset" / "org--ds.html").write_text(f"<h2>Health score: 94/100</h2><p>{stamp}</p>")

    assert lb.rescore_site(out) == 0

    board = json.loads((out / "data.json").read_text())
    assert board["generated"] == "2026-11-14 00:01 UTC"  # nothing was re-audited
    assert board["score_model"] == "tiered-v1.1"
    by_id = {e["id"]: e for e in board["datasets"]}
    assert by_id["org/ds"]["report"]["score"] == 90
    assert by_id["org/dup"]["report"]["score"] == 98
    assert all(e["report"]["score_model"] == "tiered-v1.1" for e in by_id.values() if "report" in e)
    assert "report" not in by_id["org/gated"]

    page = (out / "dataset" / "org--ds.html").read_text()
    assert "Health score: 90/100" in page and stamp in page and "tiered-v1.1" in page
    assert json.loads((out / "dataset" / "org--dup.json").read_text())["score"] == 98
    index = (out / "index.html").read_text()
    assert "tiered-v1.1" in index and "legacy-v0" not in index


def test_index_labels_unrescored_legacy_rows():
    lb = _load_leaderboard()
    legacy = _legacy_dict({"idle-segments": "WARN"}, score=94)
    html = lb.render_index([{"id": "org/ds", "slug": "org--ds", "downloads": 1,
                             "engine": "e0", "report": legacy}], "now")
    assert "(legacy-v0)" in html


# --- correlated-check waiver (tiered-v1.1) -----------------------------------

IDLE, STUCK = "idle-segments", "stuck-actuator"


def _result(check_id: str, sev: Severity, *, per_episode=(), grouped=()) -> CheckResult:
    """A CheckResult flagging `per_episode` via Finding.episode_index (the
    idle-segments shape) and `grouped` via one finding's details["episodes"]
    (the stuck-actuator shape)."""
    r = CheckResult(check_id=check_id, title=check_id)
    for ep in per_episode:
        r.findings.append(Finding(check_id, sev, f"ep {ep}", episode_index=ep))
    if grouped:
        r.findings.append(Finding(check_id, sev, "group", details={"episodes": list(grouped)}))
    return r


def _pair_report(idle: CheckResult, stuck: CheckResult, *extra: CheckResult) -> DatasetReport:
    rep = DatasetReport(source="t", format_version="v3.0", episodes=100, frames=1000, fps=10.0)
    rep.results += [idle, stuck, *extra]
    return rep


def test_edge_table_is_exactly_one_edge():
    assert WAIVER_EDGES == ((IDLE, STUCK),)


def test_scoring_code_stays_out_of_the_engine_hash():
    # the waiver is billing only: report.py must not feed the engine hash, or
    # a score-model change would mark every stored result stale
    from kinema import engine
    root = Path(engine.__file__).parent
    hashed = {p.relative_to(root).as_posix() for pat in engine._ENGINE_FILES
              for p in root.glob(pat)}
    assert "checks/idle.py" in hashed  # the glob still resolves
    assert "report.py" not in hashed


def test_flagged_episodes_reads_episode_index_and_details_episodes():
    r = _result(STUCK, Severity.WARN, per_episode=[3], grouped=[1, 2, 3])
    r.findings.append(Finding(STUCK, Severity.INFO, "no episodes", details={"joint": 2}))
    r.findings.append(Finding(STUCK, Severity.INFO, "junk", details={"episodes": ["x", True, 9]}))
    assert flagged_episodes(r) == {1, 2, 3, 9}
    assert flagged_episodes(CheckResult(STUCK, STUCK)) == set()


@pytest.mark.parametrize(
    "idle_eps, stuck_eps, overlap, waived",
    [
        (range(11), range(11), 1.0, True),            # berkeley_mvp shape
        ([8], [8, 23], 0.5, True),                    # svla_so100_sorting shape: inclusive
        (range(0, 75), range(26, 100), 0.49, False),  # 49 shared / 100 flagged
        ([5], [7], 0.0, False),
    ],
)
def test_waiver_gate(idle_eps, stuck_eps, overlap, waived):
    rep = _pair_report(_result(IDLE, Severity.WARN, per_episode=idle_eps),
                       _result(STUCK, Severity.WARN, grouped=stuck_eps))
    ch = rep.charges()[STUCK]
    assert rep.score == (90 if waived else 84)
    assert rep.charges()[IDLE] == {"charge": 10}
    if waived:
        shared = len(set(idle_eps) & set(stuck_eps))
        assert ch == {"charge": 0, "waived_by": IDLE,
                      "waiver_evidence": {"shared_episodes": shared, "overlap": overlap}}
    else:
        assert ch == {"charge": 6}


def test_downstream_fail_is_never_waived():
    rep = _pair_report(_result(IDLE, Severity.WARN, per_episode=[1, 2]),
                       _result(STUCK, Severity.FAIL, grouped=[1, 2]))
    assert rep.score == 100 - 10 - 25
    assert "waived_by" not in rep.charges()[STUCK]


def test_upstream_fail_gates_a_downstream_warn_waiver():
    rep = _pair_report(_result(IDLE, Severity.FAIL, per_episode=[1, 2]),
                       _result(STUCK, Severity.WARN, grouped=[1, 2]))
    assert rep.score == 100 - 25
    assert rep.charges()[STUCK]["waived_by"] == IDLE


@pytest.mark.parametrize("upstream", [Severity.INFO, Severity.OK])
def test_uncharged_upstream_never_waives(upstream):
    rep = _pair_report(_result(IDLE, upstream, per_episode=[1, 2]),
                       _result(STUCK, Severity.WARN, grouped=[1, 2]))
    assert rep.charges()[STUCK] == {"charge": 6}


def test_reverse_direction_is_not_an_edge():
    rep = _pair_report(_result(IDLE, Severity.WARN, per_episode=[1, 2]),
                       _result(STUCK, Severity.WARN, grouped=[1, 2]))
    assert rep.charges()[IDLE] == {"charge": 10}  # stuck-actuator never waives idle-segments


@pytest.mark.parametrize("empty_side", [IDLE, STUCK])
def test_missing_episode_evidence_never_waives(empty_side):
    idle = _result(IDLE, Severity.WARN, per_episode=[] if empty_side == IDLE else [1])
    stuck = _result(STUCK, Severity.WARN, grouped=[] if empty_side == STUCK else [1])
    # a WARN with no episode evidence at all
    (idle if empty_side == IDLE else stuck).findings.append(
        Finding(empty_side, Severity.WARN, "no episodes", details={"joint": 0}))
    rep = _pair_report(idle, stuck)
    assert rep.score == 84
    assert rep.charges()[STUCK] == {"charge": 6}


def test_verdict_only_scoring_is_tiered_v1():
    # no episode sets available: the waiver cannot fire
    assert score_from_verdicts([(IDLE, "WARN"), (STUCK, "WARN")]) == 84
    assert score_from_verdicts([(IDLE, "WARN"), (STUCK, "WARN")],
                               {IDLE: {1}, STUCK: {1}}) == 90


def test_check_order_invariance():
    results = [
        _result(IDLE, Severity.WARN, per_episode=[1, 2, 3]),
        _result(STUCK, Severity.WARN, grouped=[1, 2, 3, 4]),
        _result("timestamp-grid", Severity.FAIL, per_episode=[1]),
        _result("outlier-episodes", Severity.WARN, per_episode=[2, 3]),
        _result("duplicate-and-leakage", Severity.INFO, per_episode=[4]),
    ]
    base = _pair_report(*results)
    rng = random.Random(0)
    for _ in range(20):
        shuffled = results[:]
        rng.shuffle(shuffled)
        rep = _pair_report(*shuffled)
        assert rep.score == base.score == 100 - 10 - 25 - 3
        assert rep.charges() == base.charges()
        d = report_to_dict(rep)
        assert {c["id"]: c["charge"] for c in d["checks"]} == {
            c["id"]: c["charge"] for c in report_to_dict(base)["checks"]}


def _stored_pair(idle_eps, stuck_eps) -> dict:
    """A stored-shape report dict as data.json carries it."""
    d = _legacy_dict({IDLE: "WARN", STUCK: "WARN", "stats-drift": "PASS"}, score=84)
    by_id = {c["id"]: c for c in d["checks"]}
    by_id[IDLE]["findings"] = [{"severity": "WARN", "message": f"episode {e}",
                                "episode_index": e, "details": {"idle_fraction": 1.0}}
                               for e in idle_eps]
    by_id[STUCK]["findings"] = [{"severity": "WARN", "message": "dead",
                                 "episode_index": None, "details": {"episodes": list(stuck_eps)}}]
    return d


def test_rescore_sets_charge_fields_and_is_idempotent():
    d = _stored_pair([8], [8, 23])
    rescore_report_dict(d)
    assert d["score"] == 90 and d["score_model"] == "tiered-v1.1"
    by_id = {c["id"]: c for c in d["checks"]}
    assert by_id[STUCK]["weight"] == 6
    assert by_id[STUCK]["charge"] == 0
    assert by_id[STUCK]["waived_by"] == IDLE
    assert by_id[STUCK]["waiver_evidence"] == {"shared_episodes": 1, "overlap": 0.5}
    assert by_id[IDLE]["charge"] == 10 and "waived_by" not in by_id[IDLE]
    assert by_id["stats-drift"]["charge"] == 10 and "waived_by" not in by_id["stats-drift"]
    again = rescore_report_dict(copy.deepcopy(d))
    assert again == d
    # round trip (what dataset pages render) and the JSON writer agree with it
    rep = report_from_dict(d)
    assert rep.score == 90
    assert {c["id"]: c for c in report_to_dict(rep)["checks"]}[STUCK]["waived_by"] == IDLE


def test_rescore_drops_a_stale_waiver():
    d = _stored_pair([8], [8, 23])
    rescore_report_dict(d)
    stuck = next(c for c in d["checks"] if c["id"] == STUCK)
    stuck["findings"][0]["details"]["episodes"] = [30, 31, 32]  # no longer shared
    rescore_report_dict(d)
    assert d["score"] == 84
    assert stuck["charge"] == 6 and "waived_by" not in stuck and "waiver_evidence" not in stuck


def test_dataset_page_annotates_the_waived_row():
    lb = _load_leaderboard()
    waived = lb.render_dataset_page(report_from_dict(_stored_pair([8], [8, 23])), audited="x")
    assert "Health score: 90/100" in waived
    assert "charge waived - same episodes as idle-segments" in waived
    charged = lb.render_dataset_page(report_from_dict(_stored_pair([8], [30, 31])), audited="x")
    assert "Health score: 84/100" in charged and "charge waived" not in charged
