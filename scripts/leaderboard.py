#!/usr/bin/env python3
"""Nightly leaderboard: audit the most-downloaded LeRobot datasets on the
Hugging Face Hub and render a static site (docs/ -> GitHub Pages).

Usage: python scripts/leaderboard.py [--limit 50] [--out docs] [--max-episodes 50]

Datasets whose parquet payload exceeds --size-cap-mb are listed as
"not audited (too large for the sampled audit)" instead of being
downloaded - honesty over coverage for v0.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kinema.checks import ALL_CHECKS, CheckConfig, Severity  # noqa: E402
from kinema.dataset import DatasetError, load_dataset  # noqa: E402
from kinema.report import DatasetReport, report_to_dict  # noqa: E402

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


def _range_read_parquet(repo_id: str, name: str, dest: Path, min_rows: int = 20000,
                        max_bytes: int = 150 * 1024 * 1024) -> None:
    """Materialize the leading row groups of a remote parquet via HTTP range
    reads (footer first, then only the needed row groups) - no full download.
    Used when every data shard exceeds the sample cap."""
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    with fs.open(f"datasets/{repo_id}/{name}", "rb") as f:
        pf = pq.ParquetFile(f)
        groups, rows, total = [], 0, 0
        for i in range(pf.metadata.num_row_groups):
            rg = pf.metadata.row_group(i)
            groups.append(i)
            rows += rg.num_rows
            total += rg.total_byte_size
            if rows >= min_rows or total >= max_bytes:
                break
        table = pf.read_row_groups(groups)
    out = dest / name
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)


def download_sample(repo_id: str, cap_bytes: int, dest: Path) -> Path:
    """meta/** in full plus data files in name order until cap_bytes.

    v3 shards hold many episodes per file, so the first shard(s) are a
    real sample; v2.1 keeps one parquet per episode, so the first files
    are the first episodes. Either way the audit covers actual data.
    """
    from huggingface_hub import dataset_info, hf_hub_download

    sib = [
        (s.rfilename, s.size or 0)
        for s in (dataset_info(repo_id, files_metadata=True).siblings or [])
    ]
    picked = [n for n, _ in sib if n.startswith("meta/")]
    budget = cap_bytes
    data_files = sorted((n, sz) for n, sz in sib if n.startswith("data/"))
    # cap file count too: v2.1 keeps one small parquet per episode, and
    # hundreds of per-file requests are what trips the Hub's rate limits
    for name, size in data_files[:40]:
        if size > budget:
            continue
        picked.append(name)
        budget -= size
    if not any(n.startswith("data/") for n in picked):
        if not data_files:
            raise DatasetError(f"{repo_id}: no data files listed on the Hub")
        smallest, ssize = min(data_files, key=lambda x: x[1])
        try:
            _range_read_parquet(repo_id, smallest, dest, max_bytes=cap_bytes)
        except Exception as e:  # noqa: BLE001
            raise DatasetError(
                f"smallest data shard is {ssize / 1e6:.0f} MB and range-read "
                f"sampling failed: {type(e).__name__}: {e}"
            )
        return dest
    for name in picked:
        hf_hub_download(
            repo_id, name, repo_type="dataset", local_dir=str(dest)
        )
    return dest


def _clean_reason(msg: str) -> str:
    """Never publish local paths (or anything else environment-specific) to Pages."""
    import re

    return re.sub(r"/[\w./-]*kinema-leaderboard-[\w-]+/?\S*", "<sample>", msg)


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


def render_dataset_page(rep: DatasetReport) -> str:
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
        f"<p class=dim>audited {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC} by "
        "<a href='https://github.com/vedevpatel/kinema'>kinema</a></p>"
    )
    return page(f"{rep.source} - kinema audit", body)


def render_index(entries: list[dict], generated: str) -> str:
    rows = []
    for i, e in enumerate(entries, 1):
        if e.get("report"):
            r = e["report"]
            link = f"<a href='dataset/{esc(e['slug'])}.html'>{esc(e['id'])}</a>"
            score = f"<span class=score>{r['score']}</span>"
            verdict = badge(r["verdict"])
            checks = " ".join(
                badge(c["verdict"]) for c in r["checks"] if c["verdict"] in ("WARN", "FAIL")
            ) or '<span class=dim>all clear</span>'
        else:
            link = esc(e["id"])
            score = '<span class=dim>-</span>'
            verdict = badge("NA")
            checks = f"<span class=dim>{esc(e.get('reason', 'not audited'))}</span>"
        rows.append(
            f"<tr><td>{i}</td><td>{link}</td><td>{e['downloads']:,}</td>"
            f"<td>{score}</td><td>{verdict}</td><td>{checks}</td></tr>"
        )
    body = (
        "<h1>LeRobot dataset health leaderboard</h1>"
        f"<p class=dim>The most-downloaded LeRobot datasets on Hugging Face, audited nightly by "
        "<a href='https://github.com/vedevpatel/kinema'>kinema</a>. "
        f"Generated {esc(generated)}. Scores are heuristic: 100 minus penalties per "
        "warning/failing check. Video checks are off (v0).</p>"
        "<table><tr><th>#</th><th>dataset</th><th>downloads</th><th>score</th>"
        "<th>verdict</th><th>flags</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    return page("LeRobot dataset health leaderboard", body)


def _write(out: Path, entries: list, generated: str) -> None:
    """Atomic writes so a kill mid-run never leaves a corrupt or missing site."""
    data = json.dumps({"generated": generated, "datasets": entries}, indent=2, default=str)
    idx = render_index(entries, generated)
    for name, text in (("data.json", data), ("index.html", idx)):
        tmp = out / f".{name}.tmp"
        tmp.write_text(text)
        tmp.replace(out / name)


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
    args = ap.parse_args()

    out = Path(args.out)
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

    # resume: keep entries already audited in a previous run
    entries = []
    data_json = out / "data.json"
    done_ids: set[str] = set()
    if data_json.exists():
        try:
            prev = json.loads(data_json.read_text()).get("datasets", [])
            entries = [e for e in prev if e.get("report") or e.get("reason")]
            done_ids = {e["id"] for e in entries}
            print(f"resuming: {len(done_ids)} already done")
        except Exception:
            entries, done_ids = [], set()

    import time
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    def process(d):
        slug = d["id"].replace("/", "--")
        entry = {**d, "slug": slug}
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
        except Exception as e:  # noqa: BLE001
            entry["reason"] = _clean_reason(f"audit error: {type(e).__name__}: {e}")
        return entry

    started = time.monotonic()
    todo = [d for d in datasets if d["id"] not in done_ids]
    max_pending = max(1, args.workers) * 2

    def record(entry) -> None:
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
