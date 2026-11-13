"""Command-line interface.

  <name> check DATASET [--policy pi0] [--video] [--max-episodes N]
                      [--json [PATH]] [--markdown [PATH]] [--fail-on fail|warn]
  <name> report REPORT.json [--markdown [PATH]]
  <name> synth PATH [--episodes N] [--defects a,b,c]

DATASET is a local directory, a .zip, or a Hub repo id (owner/name or
hf://owner/name, optionally @revision to pin a branch, tag or commit).
"""

from __future__ import annotations

import argparse
import json
import sys

from kinema import __version__
from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.dataset import DatasetError, load_dataset
from kinema.report import (
    DatasetReport,
    render_console,
    render_json,
    render_markdown,
    report_from_dict,
)


def _run_checks(args) -> int:
    cfg = CheckConfig(video=args.video, policy=args.policy)
    try:
        ctx = load_dataset(args.source, max_episodes=args.max_episodes, video=args.video)
    except DatasetError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    report = DatasetReport(
        source=args.source,
        format_version=ctx.format_version,
        episodes=ctx.total_episodes,
        frames=ctx.row_count,
        fps=ctx.fps,
        truncated=ctx.truncated,
    )
    for check in ALL_CHECKS:
        try:
            report.results.append(check.run(ctx, cfg))
        except Exception as e:  # noqa: BLE001 - one broken check must not sink the audit
            report.results.append(
                check.skip(f"check crashed: {type(e).__name__}: {e}")
            )

    if args.json is not None:
        payload = render_json(report)
        if args.json == "-":
            print(payload)
        else:
            with open(args.json, "w") as f:
                f.write(payload + "\n")
    if args.markdown is not None:
        md = render_markdown(report)
        if args.markdown == "-":
            print(md)
        else:
            with open(args.markdown, "w") as f:
                f.write(md + "\n")
    if args.json != "-" and args.markdown != "-":
        render_console(report)

    gate = Severity.FAIL if args.fail_on == "fail" else Severity.WARN
    return 1 if report.verdict >= gate else 0


def _report(args) -> int:
    with open(args.report_json) as f:
        report = report_from_dict(json.load(f))
    if args.markdown is not None:
        md = render_markdown(report)
        if args.markdown == "-":
            print(md)
        else:
            with open(args.markdown, "w") as f:
                f.write(md + "\n")
    else:
        render_console(report)
    return 0


def _synth(args) -> int:
    from kinema.fixtures.synth import make_dataset

    defects = [d.strip() for d in args.defects.split(",") if d.strip()]
    make_dataset(args.path, episodes=args.episodes, defects=defects)
    print(f"synthetic dataset written to {args.path} (defects: {defects or 'none'})")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="kinema",
        description="Quality checks for LeRobot robot-learning datasets.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="run all checks on a dataset")
    p_check.add_argument("source", help="local dir, .zip, or Hub repo id (owner/name)")
    p_check.add_argument("--policy", default=None, help="target policy: act, diffusion, smolvla, pi0, pi0_fast")
    p_check.add_argument("--video", action="store_true", help="include video checks (needs the video extra)")
    p_check.add_argument("--max-episodes", type=int, default=None, help="load only the first N episodes")
    p_check.add_argument("--json", nargs="?", const="-", default=None, help="write JSON report (to PATH, or stdout)")
    p_check.add_argument("--markdown", nargs="?", const="-", default=None, help="write markdown report (to PATH, or stdout)")
    p_check.add_argument("--fail-on", choices=["fail", "warn"], default="fail", help="exit 1 on this severity or worse (default: fail)")
    p_check.set_defaults(func=_run_checks)

    p_report = sub.add_parser("report", help="re-render a saved JSON report")
    p_report.add_argument("report_json")
    p_report.add_argument("--markdown", nargs="?", const="-", default=None)
    p_report.set_defaults(func=_report)

    p_synth = sub.add_parser("synth", help="write a synthetic demo dataset (optionally with injected defects)")
    p_synth.add_argument("path")
    p_synth.add_argument("--episodes", type=int, default=24)
    p_synth.add_argument("--defects", default="", help="comma-separated defect names")
    p_synth.set_defaults(func=_synth)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
