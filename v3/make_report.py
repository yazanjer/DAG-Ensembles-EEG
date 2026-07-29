#!/usr/bin/env python3
"""
make_report.py -- build the report from predictions.jsonl.

Separate from run_study.py on purpose: the analysis can be re-run any number
of times on a finished (or partial) results directory without touching the
experiment. The experiment is run once; the reporting is idempotent.

Writes report.json and report.md next to the predictions, and prints the
pre-registered verdict per dataset.

Examples
--------
  python make_report.py --results /content/drive/MyDrive/EEG_DAGSA/results_v3
  python make_report.py --results ./results_v3 --quiet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import analysis as A


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Render tables and the verdict from a results directory.")
    p.add_argument("--results", required=True,
                   help="directory containing predictions.jsonl")
    p.add_argument("--quiet", action="store_true",
                   help="write the files but print only the verdict block")
    return p.parse_args(argv)


def print_verdicts(report: dict) -> None:
    print()
    print("=" * 70)
    print("  PRE-REGISTERED VERDICT")
    print("=" * 70)
    for ds, d in report["datasets"].items():
        v = d["verdict"]
        print(f"\n  {ds}")
        print(f"    verdict : {v['verdict']}")
        print(f"    reason  : {v['reason']}")
        print(f"    binding : ARTS vs {v['binding_comparison']}  "
              f"{v['binding_diff']:+.2f} pts  "
              f"CI [{v['binding_ci'][0]:+.2f}, {v['binding_ci'][1]:+.2f}]")
        print("    ablation (which component earns the gain):")
        for r in sorted(d["ablation"], key=lambda x: x["method"]):
            print(f"      {r['method']:24s} {r['acc_mean']:6.2f}")

    if "ds1_real_subjects_only" in report:
        v = report["ds1_real_subjects_only"]["verdict"]
        print("\n  ds1, four REAL subjects only (pre-specified secondary)")
        print(f"    verdict : {v['verdict']}")
    print("=" * 70)


def main(argv=None) -> int:
    a = parse_args(argv)
    out = Path(a.results).expanduser()
    preds = out / "predictions.jsonl"
    if not preds.exists():
        print(f"ERROR: no predictions.jsonl in {out}")
        print("       Run:  python run_study.py --drive-root <...>")
        return 2

    n = sum(1 for line in preds.open() if line.strip())
    print(f"[report] {n} units in {preds}")

    report = A.full_report(preds, out)
    if not a.quiet:
        print()
        print(A.render_markdown(report))
    print_verdicts(report)
    print(f"\nwrote {out/'report.md'}")
    print(f"wrote {out/'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
