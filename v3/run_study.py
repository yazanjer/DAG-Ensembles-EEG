#!/usr/bin/env python3
"""
run_study.py -- command-line entry point for the confirmatory run.

This is the single command the notebook calls. It is deliberately thin: all
the science lives in protocol.py, and everything here is argument handling and
a readable progress banner. Keeping it that way means the notebook and a
terminal run execute byte-identical code.

Resumable. Every completed (dataset, subject, seed) unit is appended to
predictions.jsonl and fsync'd, and is skipped on a restart, so re-running
after a Colab disconnect costs at most the unit in progress.

Examples
--------
  python run_study.py --drive-root /content/drive/MyDrive/EEG_DAGSA
  python run_study.py --drive-root ~/EEG_DAGSA --datasets ds2a_binary --no-eegnet
  python run_study.py --drive-root ~/EEG_DAGSA --seeds 42,43 --no-ablation
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import protocol as PR

ALL_DATASETS = ("ds2a_binary", "ds1", "ds2a_4class")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the pre-registered confirmatory evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--drive-root", default="/content/drive/MyDrive/EEG_DAGSA",
                   help="folder holding dataset/ and dataset_2a/; results are "
                        "written to <drive-root>/results_v3 unless --out is given")
    p.add_argument("--ds1-dir", default=None,
                   help="override the Dataset 1 directory")
    p.add_argument("--ds2a-dir", default=None,
                   help="override the Dataset 2a directory")
    p.add_argument("--out", default=None,
                   help="override the results directory")
    p.add_argument("--datasets", default=",".join(ALL_DATASETS),
                   help="comma-separated subset of "
                        "ds1,ds2a_binary,ds2a_4class")
    p.add_argument("--seeds", default=None,
                   help="comma-separated seed override (default: the "
                        "pre-registered 42-51). Changing this departs from "
                        "the pre-registration -- use only for smoke tests.")
    p.add_argument("--no-eegnet", action="store_true",
                   help="skip the EEGNet baseline (it needs a GPU to be "
                        "practical). It is recorded as an explicit null per "
                        "unit either way, never silently omitted.")
    p.add_argument("--no-ablation", action="store_true",
                   help="skip the ablation ladder (A0-A5)")
    p.add_argument("--test-fraction", type=float, default=0.30)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    root = Path(a.drive_root).expanduser()

    ds1 = Path(a.ds1_dir).expanduser() if a.ds1_dir else root / "dataset"
    ds2a = Path(a.ds2a_dir).expanduser() if a.ds2a_dir else root / "dataset_2a"
    out = Path(a.out).expanduser() if a.out else root / "results_v3"

    datasets = tuple(d.strip() for d in a.datasets.split(",") if d.strip())
    bad = [d for d in datasets if d not in ALL_DATASETS]
    if bad:
        print(f"ERROR: unknown dataset(s) {bad}; choose from {ALL_DATASETS}")
        return 2

    seeds = (tuple(int(s) for s in a.seeds.split(","))
             if a.seeds else tuple(PR.PREREG["seeds"]))
    if a.seeds:
        print("!! WARNING: --seeds overrides the pre-registered seed list. "
              "Results from this run are NOT the confirmatory result.")

    # Fail early and loudly on a missing dataset rather than 20 minutes in.
    need_ds1 = "ds1" in datasets
    need_2a = any(d.startswith("ds2a") for d in datasets)
    for label, path, needed in (("Dataset 1", ds1, need_ds1),
                                ("Dataset 2a", ds2a, need_2a)):
        if needed and not path.is_dir():
            print(f"ERROR: {label} directory not found: {path}")
            print("       Expected layout under --drive-root:")
            print("         dataset/BCICIV_calib_ds1{a..g}.mat")
            print("         dataset_2a/A0{1..9}T.mat")
            return 2

    cfg = PR.RunCfg(
        ds1_dir=str(ds1), ds2a_dir=str(ds2a), out_dir=str(out),
        datasets=datasets, seeds=seeds,
        test_fraction=a.test_fraction,
        run_eegnet=not a.no_eegnet,
        run_ablation=not a.no_ablation,
    )

    n_units = sum(len(PR.D.subjects_for(d)) for d in datasets) * len(seeds)
    print("=" * 70)
    print("  Confirmatory run -- pre-registered protocol")
    print("=" * 70)
    print(f"  datasets      : {', '.join(datasets)}")
    print(f"  seeds         : {list(seeds)}")
    print(f"  units planned : {n_units}")
    print(f"  EEGNet        : {'yes' if cfg.run_eegnet else 'SKIPPED'}")
    print(f"  ablation      : {'yes' if cfg.run_ablation else 'skipped'}")
    print(f"  ds1  dir      : {ds1}")
    print(f"  ds2a dir      : {ds2a}")
    print(f"  results       : {out}")
    print("=" * 70, flush=True)

    t0 = time.time()
    PR.run(cfg)
    mins = (time.time() - t0) / 60
    print(f"\nfinished in {mins:.1f} min -> {out}")
    print("Next:  python make_report.py --results", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
