"""
analysis.py -- turns predictions.jsonl into the pre-registered tables.

Statistical choices, and why they differ from the repository (audit A4, B3):

  * The unit of analysis is the SUBJECT, not the (subject, seed) pair. Seeds
    are re-splits of the same trials, so seed-level replicates are strongly
    dependent; a CI taken over them measures split-assignment noise rather
    than uncertainty about a new subject. Accuracy is averaged over seeds
    within a subject, and the interval is taken across subjects.
  * The critical value is Student's t on n-1 df, not 1.96. With n=7 or n=9
    subjects, z understates the interval by 20-30%.
  * McNemar p-values are Holm-corrected across the units of a comparison
    before being counted as wins or losses. The repository counted 160
    uncorrected tests at alpha=0.05.
  * Comparisons are reported as paired differences with a paired CI, because
    every method is evaluated on identical splits -- the pairing is the main
    source of power and discarding it would be wasteful.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, precision_recall_fscore_support)

PROPOSED = "ARTS"
BASELINES = ["riemannian_ts", "fbcsp", "eegnet", "single_best",
             "random_search", "dag_sa"]
ABLATION_ROWS = ["A0_single_band", "A1_filterbank_mean", "A2_filterbank_stack",
                 "A3_no_alignment", "A4_transfer_pooled", "A5_full"]
EQUIV_MARGIN = 2.0          # percentage points, pre-registered


# --------------------------------------------------------------------------- #
def load(path) -> List[dict]:
    recs = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            recs.append(json.loads(line))
    return recs


def exact_mcnemar(a_ok: np.ndarray, b_ok: np.ndarray) -> float:
    """Exact two-sided McNemar on the discordant pairs."""
    n01 = int(np.sum(~a_ok & b_ok))
    n10 = int(np.sum(a_ok & ~b_ok))
    n = n01 + n10
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * stats.binom.cdf(min(n01, n10), n, 0.5)))


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (m - i) * p[j])
        adj[j] = min(1.0, running)
    return adj


def t_ci(x: Sequence[float], alpha=0.05):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, np.nan, np.nan, 0.0
    sd = float(x.std(ddof=1))
    h = stats.t.ppf(1 - alpha / 2, n - 1) * sd / np.sqrt(n)
    return m, m - h, m + h, sd


# --------------------------------------------------------------------------- #
def per_unit_table(recs) -> List[dict]:
    rows = []
    for r in recs:
        yt = np.asarray(r["y_true"])
        for meth, pred in r["pred"].items():
            if pred is None:
                continue
            yp = np.asarray(pred)
            rows.append(dict(
                dataset=r["dataset"], subject=str(r["subject"]),
                seed=r["seed"], method=meth,
                accuracy=float((yp == yt).mean()),
                kappa=float(cohen_kappa_score(yt, yp)),
                balanced=float(balanced_accuracy_score(yt, yp)),
            ))
    return rows


def method_summary(rows, dataset, methods=None) -> List[dict]:
    """Per-method summary with the SUBJECT as the unit of analysis."""
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["dataset"] != dataset:
            continue
        by[r["method"]][r["subject"]].append(r)
    out = []
    for meth, subs in by.items():
        if methods and meth not in methods:
            continue
        acc_s = [np.mean([x["accuracy"] for x in v]) for v in subs.values()]
        kap_s = [np.mean([x["kappa"] for x in v]) for v in subs.values()]
        bal_s = [np.mean([x["balanced"] for x in v]) for v in subs.values()]
        m, lo, hi, sd = t_ci(acc_s)
        km, klo, khi, ksd = t_ci(kap_s)
        out.append(dict(
            dataset=dataset, method=meth,
            acc_mean=100 * m, acc_sd=100 * sd,
            acc_ci_low=100 * lo, acc_ci_high=100 * hi,
            kappa_mean=km, kappa_ci_low=klo, kappa_ci_high=khi,
            balanced_mean=100 * float(np.mean(bal_s)),
            n_subjects=len(subs),
            n_units=sum(len(v) for v in subs.values()),
        ))
    return sorted(out, key=lambda d: -d["acc_mean"])


def paired_comparison(rows, dataset, a=PROPOSED, b="riemannian_ts") -> dict:
    """Paired per-subject difference a - b, with a paired t interval."""
    acc = defaultdict(dict)
    for r in rows:
        if r["dataset"] != dataset:
            continue
        acc[r["method"]].setdefault(r["subject"], []).append(r["accuracy"])
    if a not in acc or b not in acc:
        return {}
    subs = sorted(set(acc[a]) & set(acc[b]))
    if not subs:
        return {}
    d = [np.mean(acc[a][s]) - np.mean(acc[b][s]) for s in subs]
    m, lo, hi, sd = t_ci(d)
    tstat, p = stats.ttest_rel(
        [np.mean(acc[a][s]) for s in subs],
        [np.mean(acc[b][s]) for s in subs]) if len(subs) > 1 else (np.nan, np.nan)
    return dict(dataset=dataset, proposed=a, baseline=b, n_subjects=len(subs),
                diff_mean=100 * m, diff_ci_low=100 * lo, diff_ci_high=100 * hi,
                diff_sd=100 * sd, t=float(tstat), p=float(p),
                per_subject={s: 100 * v for s, v in zip(subs, d)})


def mcnemar_wtl(recs, dataset, a=PROPOSED, b="riemannian_ts") -> dict:
    """Significance-gated win/tie/loss over (subject, seed) units."""
    units, praw, sign = [], [], []
    for r in recs:
        if r["dataset"] != dataset:
            continue
        pa, pb = r["pred"].get(a), r["pred"].get(b)
        if pa is None or pb is None:
            continue
        yt = np.asarray(r["y_true"])
        a_ok = np.asarray(pa) == yt
        b_ok = np.asarray(pb) == yt
        units.append((str(r["subject"]), r["seed"]))
        praw.append(exact_mcnemar(a_ok, b_ok))
        sign.append(int(np.sign(a_ok.mean() - b_ok.mean())))
    if not units:
        return {}
    padj = holm(praw)
    w = int(np.sum((padj < 0.05) & (np.array(sign) > 0)))
    l = int(np.sum((padj < 0.05) & (np.array(sign) < 0)))
    return dict(dataset=dataset, proposed=a, baseline=b, n_units=len(units),
                wins=w, losses=l, ties=len(units) - w - l,
                wins_uncorrected=int(np.sum((np.array(praw) < 0.05)
                                            & (np.array(sign) > 0))),
                losses_uncorrected=int(np.sum((np.array(praw) < 0.05)
                                              & (np.array(sign) < 0))))


def class_metrics(recs, dataset, method=PROPOSED) -> dict:
    yt, yp, names = [], [], None
    for r in recs:
        if r["dataset"] != dataset or r["pred"].get(method) is None:
            continue
        yt += r["y_true"]
        yp += r["pred"][method]
        names = r["class_names"]
    if not yt:
        return {}
    labs = sorted(set(yt) | set(yp))
    pr, rc, f1, sup = precision_recall_fscore_support(
        yt, yp, labels=labs, zero_division=0)
    nm = [names[i] if names and i < len(names) else str(i) for i in labs]
    return dict(dataset=dataset, method=method,
                classes={n: dict(precision=float(a), recall=float(b),
                                 f1=float(c), support=int(d))
                         for n, a, b, c, d in zip(nm, pr, rc, f1, sup)},
                confusion=confusion_matrix(yt, yp, labels=labs).tolist(),
                confusion_labels=nm)


def verdict(cmp_rows: List[dict]) -> dict:
    """Apply the pre-registered success criterion literally."""
    if not cmp_rows:
        return {"verdict": "no data"}
    worst = min(cmp_rows, key=lambda c: c["diff_ci_low"])
    all_positive = all(c["diff_ci_low"] > 0 for c in cmp_rows)
    all_big = all(c["diff_mean"] >= EQUIV_MARGIN for c in cmp_rows)
    if all_positive and all_big:
        v = "SUPERIOR"
        why = ("every baseline's 95% paired CI lower bound is above zero and "
               f"every point estimate is >= {EQUIV_MARGIN} points")
    elif all(abs(c["diff_ci_low"]) < EQUIV_MARGIN
             and abs(c["diff_ci_high"]) < EQUIV_MARGIN for c in cmp_rows):
        v = "TIE (equivalent)"
        why = (f"every CI lies entirely within +/-{EQUIV_MARGIN} points")
    else:
        v = "INCONCLUSIVE"
        why = ("at least one comparison has a CI that neither excludes zero "
               "nor lies inside the equivalence margin")
    return dict(verdict=v, reason=why,
                binding_comparison=worst["baseline"],
                binding_diff=worst["diff_mean"],
                binding_ci=[worst["diff_ci_low"], worst["diff_ci_high"]])


# --------------------------------------------------------------------------- #
def full_report(pred_path, out_dir=None) -> dict:
    recs = load(pred_path)
    rows = per_unit_table(recs)
    datasets = sorted({r["dataset"] for r in rows})
    report = {"datasets": {}}

    for ds in datasets:
        main = [PROPOSED] + BASELINES
        summ = method_summary(rows, ds, methods=set(main))
        abl = method_summary(rows, ds, methods=set(ABLATION_ROWS))
        cmps, wtl = [], []
        for b in BASELINES:
            c = paired_comparison(rows, ds, PROPOSED, b)
            if c:
                cmps.append(c)
                wtl.append(mcnemar_wtl(recs, ds, PROPOSED, b))
        report["datasets"][ds] = dict(
            summary=summ, ablation=abl, comparisons=cmps,
            win_tie_loss=[w for w in wtl if w],
            class_metrics=class_metrics(recs, ds, PROPOSED),
            per_subject=_per_subject(rows, ds, main),
            verdict=verdict(cmps),
        )

    # ds1 secondary analysis on the four real subjects
    if "ds1" in datasets:
        real = [r for r in rows if r["dataset"] != "ds1"
                or r["subject"] in ("a", "b", "f", "g")]
        c2 = [paired_comparison(real, "ds1", PROPOSED, b) for b in BASELINES]
        c2 = [c for c in c2 if c]
        report["ds1_real_subjects_only"] = dict(
            summary=method_summary(real, "ds1",
                                   methods=set([PROPOSED] + BASELINES)),
            comparisons=c2, verdict=verdict(c2))

    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "report.json").write_text(json.dumps(report, indent=2))
        (p / "report.md").write_text(render_markdown(report))
    return report


def _per_subject(rows, ds, methods):
    out = defaultdict(dict)
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["dataset"] == ds and r["method"] in methods:
            agg[r["subject"]][r["method"]].append(r["accuracy"])
    for s, mm in agg.items():
        for m, v in mm.items():
            out[s][m] = round(100 * float(np.mean(v)), 2)
    return dict(out)


def render_markdown(report: dict) -> str:
    L = ["# Confirmatory evaluation -- results", ""]
    for ds, d in report["datasets"].items():
        L += [f"## {ds}", "",
              "| method | acc mean | sd | 95% CI | kappa | n subj | n units |",
              "|---|---|---|---|---|---|---|"]
        for r in d["summary"]:
            L.append(f"| {r['method']} | {r['acc_mean']:.2f} | "
                     f"{r['acc_sd']:.2f} | [{r['acc_ci_low']:.2f}, "
                     f"{r['acc_ci_high']:.2f}] | {r['kappa_mean']:.3f} | "
                     f"{r['n_subjects']} | {r['n_units']} |")
        L += ["", "### ARTS vs each baseline (paired, per subject)", "",
              "| baseline | diff | 95% CI | p | wins | ties | losses |",
              "|---|---|---|---|---|---|---|"]
        wtl = {w["baseline"]: w for w in d["win_tie_loss"]}
        for c in d["comparisons"]:
            w = wtl.get(c["baseline"], {})
            L.append(f"| {c['baseline']} | {c['diff_mean']:+.2f} | "
                     f"[{c['diff_ci_low']:+.2f}, {c['diff_ci_high']:+.2f}] | "
                     f"{c['p']:.4f} | {w.get('wins','-')} | "
                     f"{w.get('ties','-')} | {w.get('losses','-')} |")
        v = d["verdict"]
        L += ["", f"**Verdict: {v['verdict']}** -- {v['reason']}. "
                  f"Binding comparison: {v['binding_comparison']} "
                  f"({v['binding_diff']:+.2f}, CI "
                  f"[{v['binding_ci'][0]:+.2f}, {v['binding_ci'][1]:+.2f}]).",
              "", "### Ablation", "",
              "| variant | acc mean | 95% CI |", "|---|---|---|"]
        for r in sorted(d["ablation"], key=lambda x: x["method"]):
            L.append(f"| {r['method']} | {r['acc_mean']:.2f} | "
                     f"[{r['acc_ci_low']:.2f}, {r['acc_ci_high']:.2f}] |")
        L += ["", "### Per subject (accuracy %)", ""]
        subs = sorted(d["per_subject"])
        meths = [r["method"] for r in d["summary"]]
        L += ["| subject | " + " | ".join(meths) + " |",
              "|" + "---|" * (len(meths) + 1)]
        for s in subs:
            L.append(f"| {s} | " + " | ".join(
                f"{d['per_subject'][s].get(m, float('nan')):.1f}"
                for m in meths) + " |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    r = full_report(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print(render_markdown(r))
