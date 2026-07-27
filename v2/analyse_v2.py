#!/usr/bin/env python3
"""
analyse_v2.py — turn a v2 campaign into the tables you would put in a paper.

    python analyse_v2.py results/ds1_v2

Writes, next to the results:
    v2_report.md    a readable summary (also printed)
    v2_table.tex    a LaTeX table of the incremental study

The report answers one question per row: does this single change improve on the
published method, and is the improvement resolvable by a paired test?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import variants as V

REF = "V0_published"


def load(exp_dir: Path):
    per = pd.read_csv(exp_dir / "v2_per_seed_results.csv")
    sig = (pd.read_csv(exp_dir / "v2_significance.csv")
           if (exp_dir / "v2_significance.csv").exists() else pd.DataFrame())
    return per, sig


def summarise(per: pd.DataFrame, sig: pd.DataFrame) -> pd.DataFrame:
    """Per-variant mean over seed-level means, plus the paired outcome."""
    seed_means = per.groupby(["variant", "seed"])["accuracy"].mean()
    rows = []
    ref = seed_means.loc[REF] if REF in seed_means.index.get_level_values(0) else None
    for vname in per["variant"].unique():
        v = seed_means.loc[vname]
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        half = 1.959963985 * sd / np.sqrt(len(v)) if len(v) > 1 else 0.0
        row = {"variant": vname, "what": V.describe(vname),
               "acc": 100 * float(v.mean()), "sd": 100 * sd,
               "ci_low": 100 * (v.mean() - half),
               "ci_high": 100 * (v.mean() + half), "n_seeds": len(v)}
        if ref is not None and vname != REF:
            common = ref.index.intersection(v.index)
            d = (v.loc[common] - ref.loc[common]).values
            row["delta"] = 100 * float(np.mean(d))
            # paired t-like summary over seeds (descriptive only)
            row["delta_sd"] = 100 * float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
            if len(sig):
                g = sig[sig.variant == vname]
                s = g[g.p_value < 0.05]
                row["wins"] = int((s.acc_variant > s.acc_reference).sum())
                row["losses"] = int((s.acc_variant < s.acc_reference).sum())
                row["ties"] = int(len(g) - len(s))
                row["n_pairs"] = int(len(g))
        else:
            row["delta"] = 0.0
        row["seconds"] = float(per[per.variant == vname]["seconds"].mean())
        rows.append(row)
    order = {n: i for i, n in enumerate(V.DEFAULT_ORDER)}
    return pd.DataFrame(rows).sort_values(
        "variant", key=lambda s: s.map(lambda x: order.get(x, 99)))


def report(exp_dir: Path) -> pd.DataFrame:
    per, sig = load(exp_dir)
    s = summarise(per, sig)

    lines = [f"# v2 incremental study — {exp_dir.name}", "",
             f"Units: {per.groupby(['subject','seed']).ngroups} "
             f"(subject, seed) pairs; {per['variant'].nunique()} variants; "
             f"identical splits throughout.", "",
             "| Variant | Change | Accuracy (%) | 95% CI | Δ vs V0 | W/T/L vs V0 | s/unit |",
             "|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        wtl = (f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}"
               if "wins" in r and not pd.isna(r.get("wins")) else "—")
        delta = "—" if r.variant == REF else f"{r.delta:+.1f}"
        lines.append(
            f"| `{r.variant}` | {r.what} | {r.acc:.1f} ± {r.sd:.1f} | "
            f"[{r.ci_low:.1f}, {r.ci_high:.1f}] | {delta} | {wtl} | "
            f"{r.seconds:.0f} |")

    best = s[s.variant != REF].sort_values("delta", ascending=False)
    lines += ["", "## Reading this table", "",
              "* `Δ vs V0` is the mean difference in accuracy over the seeds, "
              "each seed paired on identical splits.",
              "* `W/T/L` counts a win or a loss only when McNemar's exact test "
              "reaches p < 0.05 on that paired test set; everything else is a "
              "tie. A large tie count means the change is not resolvable at "
              "this sample size, whichever way the mean points.", ""]
    if len(best):
        top = best.iloc[0]
        lines.append(
            f"* Largest mean gain: `{top.variant}` ({top.delta:+.1f} points). "
            "Treat it as a hypothesis until the win/tie/loss column supports "
            "it — that is the mistake the previous version of this project "
            "made.")
    md = "\n".join(lines) + "\n"
    (exp_dir / "v2_report.md").write_text(md)
    print(md)

    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{Incremental study of the six proposed changes. Every "
           r"variant differs from the published method (V0) in exactly one "
           r"respect and is run on identical splits and seeds. A win or loss "
           r"is counted only when McNemar's exact test reaches $p<0.05$.}",
           r"\label{tab:v2}", r"\small",
           r"\begin{tabular}{llccc}", r"\toprule",
           r"Variant & Change & Accuracy (\%) & $\Delta$ & W/T/L \\", r"\midrule"]
    for _, r in s.iterrows():
        wtl = (f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}"
               if "wins" in r and not pd.isna(r.get("wins")) else "---")
        delta = "---" if r.variant == REF else f"{r.delta:+.1f}"
        name = r.variant.split("_")[0]
        lines_what = r.what.replace("item ", "").replace("&", r"\&")
        tex.append(f"{name} & {lines_what} & {r.acc:.1f} $\\pm$ {r.sd:.1f} & "
                   f"{delta} & {wtl} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (exp_dir / "v2_table.tex").write_text("\n".join(tex) + "\n")
    print(f"[analyse] wrote {exp_dir/'v2_report.md'} and {exp_dir/'v2_table.tex'}")
    return s


if __name__ == "__main__":
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/ds1_v2")
    report(d)
