#!/usr/bin/env python3
"""
run_all.py — one call that runs the whole v2 study unattended.

Designed to be started and forgotten:

* every campaign **resumes** — re-running after a Colab disconnect skips the
  (subject, seed) units already finished, so nothing is repeated and nothing is
  lost;
* results are written to disk after **every** unit, so a hard kill costs at
  most one unit;
* a campaign that fails is logged and the next one still runs;
* everything printed is also appended to a log file, so you can read what
  happened hours later even if the browser dropped the output;
* at the end it writes ONE summary file answering the two questions that
  matter: did anything beat the published method, and did the search actually
  select the strong baselines.

    import run_all
    run_all.main(project_root='/content/drive/MyDrive/EEG_DAGSA',
                 ds1_dir=..., ds2a_dir=...)
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import pandas as pd

import analyse_v2
import run_v2

# MNE prints a covariance/rank block per CSP fit -- hundreds of thousands of
# lines over a full campaign, which buries the progress output and bloats the
# log. Warnings and errors still come through.
try:
    import mne
    mne.set_log_level("WARNING")
except Exception:
    pass

# --------------------------------------------------------------------------- #
# What to run. Ordered by importance: if the session dies early, the campaign
# that answers the open question has already finished.
# --------------------------------------------------------------------------- #
STRONG_VARIANTS = ["V0_published", "V4u_enriched_unconstrained",
                   "V7u_strong_unconstrained", "V7l_strong_locked"]

CAMPAIGNS = [
    # name,        dataset, subjects,          seeds,          variants
    ("ds2a_strong_v2", "ds2a", list(range(1, 10)), list(range(42, 50)),
     STRONG_VARIANTS),
    ("ds1_strong_v2", "ds1", ["a", "b", "f", "g"], list(range(42, 50)),
     STRONG_VARIANTS),
]


class _Tee:
    """Mirror stdout to a log file so the run is readable after a disconnect."""

    def __init__(self, path):
        self.file = open(path, "a", buffering=1)
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def _diagnostics(exp_dir: Path) -> pd.DataFrame:
    """How often did the search actually pick each new member family?"""
    per = exp_dir / "v2_per_seed_results.csv"
    if not per.exists():
        return pd.DataFrame()
    d = pd.read_csv(per)
    marks = {"strong (EEGNet / exact B5)": '"RAW"',
             "tangent-space": '"RIEM"', "FBCSP": '"FB"'}
    rows = []
    for v, g in d.groupby("variant"):
        row = {"variant": v, "units": len(g),
               "accuracy": 100 * g.accuracy.mean()}
        for label, key in marks.items():
            row[label] = f"{sum(key in t for t in g.topology)}/{len(g)}"
        rows.append(row)
    return pd.DataFrame(rows)


def main(project_root=None, ds1_dir=None, ds2a_dir=None, campaigns=None,
         config_path=None, tiny=False, log_name="run_all.log"):
    campaigns = campaigns or CAMPAIGNS
    root = Path(project_root) if project_root else Path.cwd()
    results_root = root / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    tee = _Tee(results_root / log_name)
    sys.stdout = tee
    t0 = time.monotonic()
    print("=" * 72)
    print(f"run_all started {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"({len(campaigns)} campaigns)")
    print("=" * 72)

    done, failed = [], []
    for name, dataset, subjects, seeds, variant_names in campaigns:
        ddir = ds2a_dir if dataset == "ds2a" else ds1_dir
        n_units = len(subjects) * len(seeds)
        print(f"\n\n########## {name}: {dataset}, {n_units} units, "
              f"{len(variant_names)} variants ##########")
        try:
            run_v2.run_campaign(
                dataset=dataset, subjects=subjects, seeds=seeds,
                variant_names=variant_names, experiment=name,
                project_root=str(root), dataset_dir=ddir,
                config_path=config_path, tiny=tiny, resume=True, verbose=True)
            done.append(name)
        except Exception as e:
            print(f"!! campaign {name} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(name)

    # ---------------------------------------------------------------- report
    print("\n\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    lines = ["# v2 study — combined summary", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} after "
             f"{(time.monotonic() - t0) / 3600:.1f} h.", ""]
    for name, *_ in campaigns:
        exp_dir = results_root / name
        if not (exp_dir / "v2_per_seed_results.csv").exists():
            lines += [f"## {name}", "", "no results written", ""]
            continue
        try:
            analyse_v2.report(exp_dir)
        except Exception as e:                       # never lose the raw data
            print(f"  (report failed for {name}: {e})")
        diag = _diagnostics(exp_dir)
        lines += [f"## {name}", ""]
        report_md = exp_dir / "v2_report.md"
        if report_md.exists():
            lines += [report_md.read_text(), ""]
        if len(diag):
            lines += ["### Did the search select the new members?", "",
                      diag.to_markdown(index=False), "",
                      "If the *strong* column is near 0, the accuracies above "
                      "say nothing about whether embedding EEGNet and the B5 "
                      "baseline helps — the search never used them.", ""]
        print(f"\n--- {name} ---")
        print(diag.to_string(index=False) if len(diag) else "no rows")

    lines += ["## How to read this", "",
              "* Read the **W/T/L** column before the Δ column. A win or loss "
              "is counted only where McNemar reaches p < 0.05.",
              "* The noise floor measured on Dataset 1 is about **3 accuracy "
              "points** on a 32-unit mean: re-running the *identical* method "
              "with a different RNG trajectory moves the per-unit result by "
              "17 points (sd). Treat anything smaller than that as a tie, "
              "whichever way the mean points.", ""]
    out = results_root / "SUMMARY.md"
    out.write_text("\n".join(lines))

    print(f"\ncampaigns completed: {done or 'none'}")
    print(f"campaigns failed   : {failed or 'none'}")
    print(f"total wall clock   : {(time.monotonic() - t0) / 3600:.2f} h")
    print(f"summary written to : {out}")
    print(f"log written to     : {results_root / log_name}")
    print("\nALL DONE")
    sys.stdout = tee.stdout
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--ds1-dir", default=None)
    ap.add_argument("--ds2a-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    main(project_root=a.project_root, ds1_dir=a.ds1_dir, ds2a_dir=a.ds2a_dir,
         config_path=a.config, tiny=a.tiny)
