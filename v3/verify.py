#!/usr/bin/env python3
"""
verify.py -- integrity checks on a results directory.

Each check targets a specific defect the audit found in the previous codebase,
so this is not decoration: it is the guard rail that stops those defects
recurring silently. Exits non-zero if any check fails, so it can gate a
pipeline.

  duplicate units      -> audit B4 (--no-resume doubled every row, and the
                          win/tie/loss counts were row counts)
  unequal unit counts  -> audit B5 (a baseline that raised was silently
                          dropped, so methods were aggregated over different
                          numbers of units while the README claimed otherwise)
  length mismatches    -> every method must be scored on the same test trials
  ARTS vs A5_full      -> identical configurations must give identical
                          predictions; a disagreement means the ablation
                          plumbing has drifted from the method
  explicit nulls       -> a skipped method must be RECORDED, not omitted

Example
-------
  python verify.py --results ./results_v3
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

OK = "  ok  "
BAD = " FAIL "


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Integrity checks on a run.")
    p.add_argument("--results", required=True)
    a = p.parse_args(argv)

    preds = Path(a.results).expanduser() / "predictions.jsonl"
    if not preds.exists():
        print(f"ERROR: no predictions.jsonl in {a.results}")
        return 2

    recs = [json.loads(l) for l in preds.open() if l.strip()]
    keys = [(r["dataset"], str(r["subject"]), r["seed"]) for r in recs]
    failures = []

    print("=" * 70)
    print("  INTEGRITY CHECKS")
    print("=" * 70)

    by_ds = collections.Counter(r["dataset"] for r in recs)
    print(f"\n  units: {len(recs)} total  " +
          "  ".join(f"{k}={v}" for k, v in sorted(by_ds.items())))

    # --- duplicates (audit B4) ----------------------------------------- #
    dups = [k for k, c in collections.Counter(keys).items() if c > 1]
    tag = OK if not dups else BAD
    print(f"\n[{tag}] duplicate units: {len(dups)}")
    if dups:
        failures.append("duplicate units")
        for d in dups[:5]:
            print(f"          {d}")

    # --- units per method (audit B5) ----------------------------------- #
    scored = collections.Counter()
    skipped = collections.Counter()
    for r in recs:
        for m, pr in r["pred"].items():
            (scored if pr is not None else skipped)[m] += 1
    counts = set(scored.values())
    # A method that is null on EVERY unit (e.g. EEGNet with no GPU) is an
    # honest, recorded absence rather than an inconsistency.
    fully_absent = {m for m in skipped if scored.get(m, 0) == 0}
    present = {m: c for m, c in scored.items()}
    consistent = len(set(present.values())) <= 1
    tag = OK if consistent else BAD
    print(f"\n[{tag}] units per scored method (must all be equal):")
    for m, c in sorted(present.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"          {m:24s} {c}")
    if not consistent:
        failures.append("unequal unit counts per method")
    if skipped:
        print("\n  explicitly recorded as skipped (not silently dropped):")
        for m, c in sorted(skipped.items()):
            note = " [absent on every unit]" if m in fully_absent else ""
            print(f"          {m:24s} {c}{note}")

    # --- prediction lengths -------------------------------------------- #
    bad_len = [k for r, k in zip(recs, keys)
               if any(len(pr) != len(r["y_true"])
                      for pr in r["pred"].values() if pr is not None)]
    tag = OK if not bad_len else BAD
    print(f"\n[{tag}] prediction/label length mismatches: {len(bad_len)}")
    if bad_len:
        failures.append("prediction length mismatch")

    # --- ARTS vs A5_full ------------------------------------------------ #
    pairs = [r for r in recs
             if r["pred"].get("ARTS") is not None
             and r["pred"].get("A5_full") is not None]
    mism = [k for r, k in zip(recs, keys)
            if r["pred"].get("ARTS") is not None
            and r["pred"].get("A5_full") is not None
            and r["pred"]["ARTS"] != r["pred"]["A5_full"]]
    if pairs:
        tag = OK if not mism else BAD
        print(f"\n[{tag}] ARTS vs A5_full (identical configs): "
              f"{len(mism)} disagreement(s) in {len(pairs)} units")
        if mism:
            failures.append("ARTS != A5_full")
    else:
        print("\n[ skip ] ARTS vs A5_full: ablation not run")

    # --- test-set sizes -------------------------------------------------- #
    sizes = collections.Counter((r["dataset"], len(r["y_true"])) for r in recs)
    print("\n  test-set sizes:")
    for (ds, n), c in sorted(sizes.items()):
        print(f"          {ds:16s} {n} trials  ({c} units)")

    print("\n" + "=" * 70)
    if failures:
        print("  RESULT: FAILED -- " + "; ".join(failures))
        print("=" * 70)
        return 1
    print("  RESULT: all checks passed")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
