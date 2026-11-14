#!/usr/bin/env python3
"""Nightly leaderboard: audit the most-downloaded LeRobot datasets on the
Hugging Face Hub and render a static site (docs/ -> GitHub Pages).

Usage: python scripts/leaderboard.py [--limit 50] [--out docs] [--max-episodes 50]
       python scripts/leaderboard.py --rescore [--out docs]

--rescore recomputes every stored score from its stored per-check verdicts
under the current score model (kinema.report.SCORE_MODEL) and re-renders
the site - no downloads, no check re-runs. Every normal run rescores the
board it resumes from the same way, so a score-model change lands in one run.

Datasets whose parquet payload exceeds --size-cap-mb are listed as
"not audited (too large for the sampled audit)" instead of being
downloaded - honesty over coverage for v0.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kinema.checks import ALL_CHECKS, CheckConfig, Severity  # noqa: E402
from kinema.dataset import DatasetError, load_dataset  # noqa: E402
from kinema.engine import engine_version  # noqa: E402
from kinema.report import (  # noqa: E402
    SCORE_MODEL,
    DatasetReport,
    report_from_dict,
    report_to_dict,
    rescore_report_dict,
)

SIZE_CAP_MB_DEFAULT = 200

CSS = """
:root { color-scheme: light dark; }
body { font-family: ui-sans-serif, system-ui, sans-serif; max-width: 1080px;
       margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #8884; }
th { position: sticky; top: 0; background: Canvas; }
tr:hover td { background: #8881; }
.badge { display: inline-block; padding: .05rem .5rem; border-radius: 1rem;
         font-size: .8rem; font-weight: 600; }
.b-PASS { background: #2da44e22; color: #2da44e; }
.b-INFO { background: #0969da22; color: #0969da; }
.b-WARN { background: #bf870022; color: #bf8700; }
.b-FAIL { background: #cf222e22; color: #cf222e; }
.b-SKIP, .b-NA { background: #8882; color: #888; }
.score { font-weight: 700; }
.dim { color: #888; }
a { color: #0969da; }
h1 { margin-bottom: 0; }
"""

VERDICTS = ["PASS", "INFO", "WARN", "FAIL", "SKIP"]


def esc(s) -> str:
    return html.escape(str(s))


def page(title: str, body: str) -> str:
    return (
        "<!doctype html><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{esc(title)}</title><style>{CSS}</style>{body}"
    )


def badge(v: str) -> str:
    return f'<span class="badge b-{v}">{v}</span>'


def find_lerobot_datasets(limit: int) -> list[dict]:
    from huggingface_hub import HfApi

    api = HfApi()
    out = []
    for d in api.list_datasets(search="lerobot", sort="downloads", limit=limit * 3):
        try:
            info = api.dataset_info(d.id, files_metadata=True)
        except Exception:
            continue
        sib = [(s.rfilename, s.size or 0) for s in (info.siblings or [])]
        files = [n for n, _ in sib]
        if "meta/info.json" not in files:
            continue
        data_bytes = sum(sz for n, sz in sib if n.startswith("data/"))
        out.append(
            {
                "id": d.id,
                "downloads": d.downloads or 0,
                "likes": getattr(d, "likes", 0) or 0,
                "data_bytes": data_bytes,
            }
        )
        if len(out) >= limit:
            break
    return out


def _classic_resolve_download(repo_id: str, name: str, dest: Path) -> None:
    """Plain HTTP GET on the classic resolve endpoint - the fallback when the
    Xet CAS backend cannot reconstruct a file (upstream 404s)."""
    import os
    import shutil
    import urllib.request

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{name}"
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    out = dest / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req) as r, open(out, "wb") as w:
        shutil.copyfileobj(r, w)


def _hub_download(repo_id: str, name: str, dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(repo_id, name, repo_type="dataset", local_dir=str(dest))
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        if "reconstruction" in msg or "cas-server" in msg or "Xet" in msg:
            try:
                _classic_resolve_download(repo_id, name, dest)
            except Exception:
                raise e from None  # fallback failed too; surface the original
        else:
            raise


def _meta_episode_indices(root: Path) -> list[int] | None:
    """Episode indices from an already-downloaded meta/, either format."""
    import json as _json

    import pyarrow.parquet as _pq

    ep_dir = root / "meta" / "episodes"
    jsonl = root / "meta" / "episodes.jsonl"
    try:
        if ep_dir.is_dir():
            files = sorted(ep_dir.glob("**/*.parquet"))
            if files:
                idx: list[int] = []
                for f in files:
                    idx.extend(_pq.read_table(f, columns=["episode_index"]).column(0).to_pylist())
                return [int(i) for i in idx]
        if jsonl.exists():
            return [
                int(_json.loads(line)["episode_index"])
                for line in jsonl.read_text().splitlines()
                if line.strip()
            ]
    except Exception:  # noqa: BLE001 - targeting is best-effort
        return None
    return None


def _range_read_parquet(repo_id: str, name: str, dest: Path, episodes: list[int] | None = None,
                        min_rows: int = 20000, max_bytes: int = 150 * 1024 * 1024) -> None:
    """Materialize row groups of a remote parquet via HTTP range reads (footer
    first, then only the needed row groups) - no full download. Used when every
    data shard exceeds the sample cap. When episode indices are known, prefer
    row groups whose episode_index statistics overlap them, streaming batchwise
    so one giant row group cannot exhaust memory; falls back to leading row
    groups."""
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    # open via filesystem+path (not a stream handle) so pyarrow issues true
    # range requests instead of buffering the whole remote file
    pf = pq.ParquetFile(f"datasets/{repo_id}/{name}", filesystem=fs)
    # project out pixel columns: frames embedded as images dominate these
    # shards, and v0 runs no visual checks
    columns = [
        n for n in pf.schema_arrow.names
        if "image" not in n.lower() and "video" not in n.lower()
    ] or None
    table = None
    if episodes:
        want = sorted(set(episodes))
        groups, total, stats_ok = [], 0, True
        for i in range(pf.metadata.num_row_groups):
            rg = pf.metadata.row_group(i)
            bounds = None
            for j in range(rg.num_columns):
                col = rg.column(j)
                if col.path_in_schema == "episode_index":
                    if col.is_stats_set and col.statistics is not None:
                        bounds = (col.statistics.min, col.statistics.max)
                    break
            if bounds is None:
                stats_ok = False
                break
            lo, hi = bounds
            if any(lo <= e <= hi for e in want):
                groups.append(i)
                total += rg.total_byte_size
                if total >= max_bytes:
                    break
        if stats_ok and groups:
            # stream batchwise: a single row group can dwarf memory, so filter
            # to catalogued episodes per batch and stop once the sample
            # covers enough of them
            import pyarrow as pa
            import pyarrow.compute as pc

            value_set = pa.array(want)
            kept, seen = [], set()
            for i in groups:
                for batch in pf.iter_batches(batch_size=1024, row_groups=[i], columns=columns):
                    t = pa.Table.from_batches([batch])
                    t = t.filter(pc.is_in(t.column("episode_index"), value_set=value_set))
                    if t.num_rows:
                        kept.append(t)
                        seen.update(t.column("episode_index").to_pylist())
                    if len(seen) >= 50:
                        break
                if len(seen) >= 50:
                    break
            if kept:
                table = pa.concat_tables(kept)
    if table is None:
        groups, rows, total = [], 0, 0
        for i in range(pf.metadata.num_row_groups):
            rg = pf.metadata.row_group(i)
            groups.append(i)
            rows += rg.num_rows
            total += rg.total_byte_size
            if rows >= min_rows or total >= max_bytes:
                break
        table = pf.read_row_groups(groups, columns=columns)
    out = dest / name
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)

def download_sample(repo_id: str, cap_bytes: int, dest: Path) -> Path:
    """meta/** in full plus data files in name order until cap_bytes.

    v3 shards hold many episodes per file, so the first shard(s) are a
    real sample; v2.1 keeps one parquet per episode, so the first files
    are the first episodes. Either way the audit covers actual data.
    """
    from huggingface_hub import dataset_info

    sib = [
        (s.rfilename, s.size or 0)
        for s in (dataset_info(repo_id, files_metadata=True).siblings or [])
    ]
    picked = [n for n, _ in sib if n.startswith("meta/")]
    # meta always comes down first: every branch needs it, both to audit and
    # to target range reads at row groups overlapping catalogued episodes
    for name in picked:
        _hub_download(repo_id, name, dest)
    budget = cap_bytes
    data_files = sorted((n, sz) for n, sz in sib if n.startswith("data/"))
    # cap file count too: v2.1 keeps one small parquet per episode, and
    # hundreds of per-file requests are what trips the Hub's rate limits.
    # Candidates are spread evenly over the dataset (not the first files:
    # early episodes are often warm-up recordings).
    chosen = []
    n_files = len(data_files)
    spread = sorted({int(round(x)) for x in
                     (i * (n_files - 1) / 39 for i in range(40))} if n_files > 40 else range(n_files))
    for name, size in (data_files[i] for i in spread):
        if size > budget:
            continue
        chosen.append(name)
        budget -= size
    if not chosen:
        if not data_files:
            raise DatasetError(f"{repo_id}: no data files listed on the Hub")
        smallest, ssize = min(data_files, key=lambda x: x[1])
        try:
            # bound range-read memory: decompressed arrow tables run several
            # times the compressed size, and workers share one process
            _range_read_parquet(
                repo_id, smallest, dest,
                episodes=_meta_episode_indices(dest),
                max_bytes=min(cap_bytes, 64 * 1024 * 1024),
            )
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            if "reconstruction" in msg or "cas-server" in msg or "Xet" in msg:
                _classic_resolve_download(repo_id, smallest, dest)
                return dest
            raise DatasetError(
                f"smallest data shard is {ssize / 1e6:.0f} MB and range-read "
                f"sampling failed: {msg}"
            )
        return dest
    for name in chosen:
        _hub_download(repo_id, name, dest)
    return dest


def _clean_reason(msg: str) -> str:
    """Never publish local paths (or anything else environment-specific) to Pages."""
    import re

    return re.sub(r"/[\w./-]*kinema-leaderboard-[\w-]+/?\S*", "<sample>", msg)


# failure taxonomy: permanent states stay on the board as-is, transient
# errors are retried on the next run (see the resume filter in main)
def _classify_reason(reason: str) -> str:
    low = reason.lower()
    if "gatedrepoerror" in low or "gated repo" in low:
        return "gated"
    if "not a lerobot dataset" in low or "no episode metadata" in low:
        return "unsupported"
    return "error"


_STATUS_LABELS = {
    "gated": "gated",
    "unsupported": "unsupported format",
    "error": "audit failed - retry queued",
}


def audit_one(repo_id: str, max_episodes: int, cap_bytes: int) -> DatasetReport:
    import shutil
    import tempfile

    dest = Path(tempfile.mkdtemp(prefix="kinema-leaderboard-"))
    try:
        download_sample(repo_id, cap_bytes, dest)
        ctx = load_dataset(str(dest), max_episodes=max_episodes)
    finally:
        shutil.rmtree(dest, ignore_errors=True)
    rep = DatasetReport(
        source=repo_id,
        format_version=ctx.format_version,
        episodes=ctx.total_episodes,
        frames=ctx.row_count,
        fps=ctx.fps,
        truncated=ctx.truncated,
    )
    cfg = CheckConfig()
    for check in ALL_CHECKS:
        try:
            rep.results.append(check.run(ctx, cfg))
        except Exception as e:  # noqa: BLE001
            rep.results.append(check.skip(f"check crashed: {type(e).__name__}: {e}"))
    return rep


def render_dataset_page(rep: DatasetReport, audited: str | None = None) -> str:
    rows = "".join(
        f"<tr><td><code>{esc(r.check_id)}</code></td>"
        f"<td>{badge('SKIP' if r.skipped else r.verdict.name)}</td>"
        f"<td>{esc(r.skip_reason if r.skipped else r.summary)}</td></tr>"
        for r in rep.results
    )
    findings = []
    for r in rep.results:
        for f in r.findings:
            if f.severity >= Severity.WARN:
                findings.append(
                    f"<li>{badge(f.severity.name)} <code>{esc(r.check_id)}</code> {esc(f.message)}</li>"
                )
    body = (
        f"<p><a href='../index.html'>&larr; leaderboard</a></p>"
        f"<h1>{esc(rep.source)}</h1>"
        f"<p class=dim>format {esc(rep.format_version)} | {rep.episodes} episodes | "
        f"{rep.frames} frames | fps {esc(rep.fps or '?')}"
        f"{' | sampled subset' if rep.truncated else ''}</p>"
        f"<h2>Health score: {rep.score}/100</h2>"
        f"<table><tr><th>check</th><th>verdict</th><th>summary</th></tr>{rows}</table>"
        f"<h2>Warnings &amp; failures</h2><ul>{''.join(findings) or '<li>none</li>'}</ul>"
        f"<p class=dim>audited {esc(audited or f'{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}')} by "
        "<a href='https://github.com/vedevpatel/kinema'>kinema</a>"
        f" | score model <code>{esc(SCORE_MODEL)}</code></p>"
    )
    return page(f"{rep.source} - kinema audit", body)


def render_index(entries: list[dict], generated: str) -> str:
    engine = engine_version()
    rows = []
    n_stale = 0
    entries = sorted(entries, key=lambda e: -(e.get("downloads") or 0))  # current ranking
    for i, e in enumerate(entries, 1):
        stale = e.get("report") and e.get("engine") != engine
        n_stale += bool(stale)
        if e.get("report"):
            r = e["report"]
            link = f"<a href='dataset/{esc(e['slug'])}.html'>{esc(e['id'])}</a>"
            score = f"<span class=score>{r['score']}</span>"
            model = r.get("score_model", "legacy-v0")
            if model != SCORE_MODEL:
                score += f' <span class=dim>({esc(model)})</span>'

            verdict = badge(r["verdict"])
            checks = " ".join(
                badge(c["verdict"]) for c in r["checks"] if c["verdict"] in ("WARN", "FAIL")
            ) or '<span class=dim>all clear</span>'
            if stale:
                checks += ' <span class=dim title="audited by older check logic">(stale - re-audit queued)</span>'
        else:
            link = esc(e["id"])
            score = '<span class=dim>-</span>'
            verdict = badge("NA")
            label = _STATUS_LABELS.get(e.get("status") or "error", _STATUS_LABELS["error"])
            detail = esc(e.get("reason", ""))
            checks = f'<span class=dim title="{detail}">{label}</span>'
        rows.append(
            f"<tr><td>{i}</td><td>{link}</td><td>{e['downloads']:,}</td>"
            f"<td>{score}</td><td>{verdict}</td><td>{checks}</td></tr>"
        )
    body = (
        "<h1>LeRobot dataset health leaderboard</h1>"
        f"<p class=dim>The most-downloaded LeRobot datasets on Hugging Face, audited nightly by "
        "<a href='https://github.com/vedevpatel/kinema'>kinema</a>. "
        f"Generated {esc(generated)} by engine <code>{esc(engine)}</code>"
        + (f"; {n_stale} result(s) from older check logic are being re-audited" if n_stale else "")
        + f". Scores use model <code>{esc(SCORE_MODEL)}</code>: 100 minus 25 per failing "
        "check and a per-check weight for each warning check (10 / 6 / 2 by measured "
        "training harm where the harm study has measured it, hand-tuned otherwise; "
        "<a href='https://github.com/vedevpatel/kinema/blob/main/docs/checks.md#health-score'>"
        "details</a>). Scores are not comparable across score models. Video checks are not run "
        "here (MP4s are hundreds of MB per camera); run <code>kinema check --video</code> "
        "locally.</p>"
        "<p id=pager><button onclick=pg(-1)>&larr; Prev</button> "
        "<span id=pglabel></span> "
        "<button onclick=pg(1)>Next &rarr;</button></p>"
        "<table id=lb><thead><tr><th>#</th><th>dataset</th><th>downloads</th>"
        f"<th>score <span class=dim>({esc(SCORE_MODEL)})</span></th>"
        "<th>verdict</th><th>flags</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        "<script>"
        "const PER=50;let cur=0;"
        "const rows=[...document.querySelectorAll('#lb tbody tr')];"
        "function show(){rows.forEach((r,i)=>r.style.display="
        "(i>=cur*PER&&i<(cur+1)*PER)?'':'none');"
        "document.getElementById('pglabel').textContent="
        "(cur+1)+' / '+Math.max(1,Math.ceil(rows.length/PER));}"
        "function pg(d){cur=Math.min(Math.ceil(rows.length/PER)-1,Math.max(0,cur+d));show();}"
        "show();"
        "</script>"
    )
    return page("LeRobot dataset health leaderboard", body)


def _write(out: Path, entries: list, generated: str) -> None:
    """Atomic writes so a kill mid-run never leaves a corrupt or missing site."""
    data = json.dumps(
        {"generated": generated, "score_model": SCORE_MODEL, "datasets": entries},
        indent=2, default=str,
    )
    idx = render_index(entries, generated)
    for name, text in (("data.json", data), ("index.html", idx)):
        tmp = out / f".{name}.tmp"
        tmp.write_text(text)
        tmp.replace(out / name)


def rescore_entries(entries: list[dict]) -> int:
    """Rescore every audited entry in place from its stored per-check verdicts
    (never per finding, never a check re-run); returns how many scores moved."""
    moved = 0
    for e in entries:
        if e.get("report"):
            before = e["report"].get("score")
            rescore_report_dict(e["report"])
            moved += e["report"]["score"] != before
    return moved


_AUDITED_RE = re.compile(r"audited (\d{4}-\d\d-\d\d \d\d:\d\d UTC)")


def rewrite_dataset_pages(out: Path, entries: list[dict]) -> None:
    """Re-render each audited entry's dataset page + report JSON from its
    stored report, keeping the page's original audit stamp."""
    for e in entries:
        if not e.get("report"):
            continue
        page_path = out / "dataset" / f"{e['slug']}.html"
        m = _AUDITED_RE.search(page_path.read_text()) if page_path.exists() else None
        rep = report_from_dict(e["report"])
        # the page renders from the round-tripped report; it must agree with
        # the stored verdicts the board score was computed from
        if rep.score != e["report"]["score"]:
            raise RuntimeError(f"{e['id']}: round-tripped score {rep.score} != stored "
                               f"{e['report']['score']}")
        for name, text in (
            (f"{e['slug']}.json", json.dumps(e["report"], indent=2, default=str)),
            (f"{e['slug']}.html", render_dataset_page(rep, audited=m.group(1) if m else "unknown")),
        ):
            tmp = out / "dataset" / f".{name}.tmp"
            tmp.write_text(text)
            tmp.replace(out / "dataset" / name)


def rescore_site(out: Path) -> int:
    """--rescore: recompute data.json, index.html and every dataset page from
    the stored verdicts. Keeps the board's generated stamp and each page's
    audit stamp - nothing was re-audited."""
    data_json = out / "data.json"
    board = json.loads(data_json.read_text())
    entries = board.get("datasets", [])
    moved = rescore_entries(entries)
    rewrite_dataset_pages(out, entries)
    _write(out, entries, board.get("generated", "unknown"))
    n = sum(1 for e in entries if e.get("report"))
    print(f"rescored {n} datasets under {SCORE_MODEL}: {moved} score(s) changed")
    return 0


def main() -> int:
    import socket

    socket.setdefaulttimeout(150)  # no download op may hang the whole run
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", default="docs")
    ap.add_argument("--max-episodes", type=int, default=50)
    ap.add_argument("--size-cap-mb", type=int, default=SIZE_CAP_MB_DEFAULT)
    ap.add_argument("--time-budget", type=float, default=0, help="stop starting new audits after N seconds")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worklist", default="", help="cache file for the dataset list")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute stored scores under the current score model; no audits")
    args = ap.parse_args()

    out = Path(args.out)
    if args.rescore:
        return rescore_site(out)
    (out / "dataset").mkdir(parents=True, exist_ok=True)

    worklist_path = Path(args.worklist) if args.worklist else None
    if worklist_path and worklist_path.exists():
        datasets = json.loads(worklist_path.read_text())
        print(f"loaded {len(datasets)} datasets from worklist cache")
    else:
        print(f"listing top {args.limit} lerobot datasets on the Hub...")
        datasets = find_lerobot_datasets(args.limit)
        print(f"found {len(datasets)} LeRobot-format datasets")
        if worklist_path:
            worklist_path.write_text(json.dumps(datasets))

    # resume: skip entries already audited BY THIS ENGINE. Results from older
    # check logic stay on the board (marked stale) until they are replaced,
    # and are re-audited first; gated repos stay gated whatever the engine.
    engine = engine_version()
    entries = []
    data_json = out / "data.json"
    done_ids: set[str] = set()
    if data_json.exists():
        try:
            prev = json.loads(data_json.read_text()).get("datasets", [])
            entries = [
                e for e in prev
                if e.get("report") or (e.get("reason") and e.get("status") in ("gated", "unsupported"))
            ]
            # drop datasets that fell out of the current top list - unless the
            # listing came back suspiciously short (a Hub hiccup must not wipe
            # the board)
            current = {d["id"]: d for d in datasets}
            if len(current) >= 0.5 * len({e["id"] for e in entries}):
                dropped = [e["id"] for e in entries if e["id"] not in current]
                entries = [e for e in entries if e["id"] in current]
                for e in entries:  # fresh download counts for the ranking
                    e["downloads"] = current[e["id"]].get("downloads", e.get("downloads", 0))
                if dropped:
                    print(f"dropping {len(dropped)} dataset(s) no longer in the top list")
            else:
                print(f"listing returned {len(current)} datasets for a board of {len(entries)}; "
                      "not pruning")
            # stored results carry their verdicts: rescore them under the
            # current model so the whole board swaps models in this run
            n_moved = rescore_entries(entries)
            done_ids = {
                e["id"] for e in entries
                if e.get("status") == "gated" or e.get("engine") == engine
            }
            n_stale = sum(1 for e in entries if e["id"] not in done_ids)
            print(f"resuming: {len(done_ids)} current, {n_stale} stale (engine {engine})")
        except Exception:
            entries, done_ids, n_moved = [], set(), 0
        if n_moved:
            # outside the try: a page-render error must not wipe the board
            rewrite_dataset_pages(out, entries)
            print(f"rescored stored results under {SCORE_MODEL}: {n_moved} score(s) changed")
    stale_ids = {e["id"] for e in entries if e["id"] not in done_ids}

    import time
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    def process(d):
        slug = d["id"].replace("/", "--")
        entry = {**d, "slug": slug, "engine": engine}
        if d["data_bytes"] > args.size_cap_mb * 1024 * 1024:
            entry["sampled"] = True
        try:
            rep = audit_one(d["id"], args.max_episodes, args.size_cap_mb * 1024 * 1024)
            entry["report"] = report_to_dict(rep)
            (out / "dataset" / f"{slug}.html").write_text(render_dataset_page(rep))
            (out / "dataset" / f"{slug}.json").write_text(json.dumps(entry["report"], indent=2, default=str))
            print(f"    {d['id']}: score {rep.score}/100 ({rep.verdict.name})", flush=True)
        except DatasetError as e:
            entry["reason"] = _clean_reason(str(e))
            entry["status"] = _classify_reason(entry["reason"])
        except Exception as e:  # noqa: BLE001
            entry["reason"] = _clean_reason(f"audit error: {type(e).__name__}: {e}")
            entry["status"] = _classify_reason(entry["reason"])
        return entry

    started = time.monotonic()
    todo = [d for d in datasets if d["id"] not in done_ids]
    todo.sort(key=lambda d: d["id"] not in stale_ids)  # stale results first
    max_pending = max(1, args.workers) * 2

    def record(entry) -> None:
        for i, e in enumerate(entries):
            if e["id"] == entry["id"]:
                entries[i] = entry  # replace the stale result in place
                break
        else:
            entries.append(entry)
        generated = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}"
        _write(out, entries, generated)

    # submit incrementally: the queue stays shallow so --time-budget actually
    # bounds the run (at most max_pending audits in flight when it expires)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        pending = {}
        it = iter(todo)

        def submit_next() -> bool:
            try:
                d = next(it)
            except StopIteration:
                return False
            pending[pool.submit(process, d)] = d
            return True

        for _ in range(min(len(todo), max_pending)):
            submit_next()
        while pending:
            if args.time_budget and time.monotonic() - started > args.time_budget:
                print("time budget reached; letting in-flight audits finish", flush=True)
                break
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for fut in done:
                pending.pop(fut)
                record(fut.result())
            while len(pending) < max_pending and submit_next():
                pass
        for fut in list(pending):
            try:
                record(fut.result(timeout=600))
            except Exception as e:  # noqa: BLE001
                print(f"    dropped in-flight audit: {type(e).__name__}", flush=True)

    generated = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}"
    _write(out, entries, generated)
    n_ok = sum(1 for e in entries if e.get("report"))
    print(f"done: {n_ok}/{len(entries)} audited -> {out}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
