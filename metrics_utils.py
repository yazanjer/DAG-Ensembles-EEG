"""
metrics_utils.py
================
Statistics the reviewers asked for:

  * class-wise precision / recall / F1, balanced accuracy, Cohen's kappa (R2-10)
  * 95% confidence intervals — bootstrap (default) or normal approx (R2-4)
  * McNemar's exact test with discordant-pair counts (R1-1)
  * paired win / tie / loss tally with significance (R2-2)
  * helpers to aggregate across the 8 seeds as mean ± std ± 95% CI (R2-3)

No result is labelled "best" here; the driver only claims a win when McNemar is
significant (see build_winloss_table).
"""

from __future__ import annotations

from math import sqrt
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             balanced_accuracy_score, cohen_kappa_score)


# --------------------------------------------------------------------------- #
# Point metrics
# --------------------------------------------------------------------------- #
def classwise_metrics(y_true, y_pred, class_names=None) -> dict:
    """Return accuracy + per-class P/R/F1 + balanced acc + kappa."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if class_names is None:
        class_names = [str(l) for l in labels]
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class": {},
    }
    for i, lab in enumerate(labels):
        name = class_names[i] if i < len(class_names) else str(lab)
        out["per_class"][name] = {
            "precision": float(p[i]), "recall": float(r[i]),
            "f1": float(f1[i]), "support": int(sup[i]),
        }
    return out


def flatten_metrics(m: dict, prefix="") -> dict:
    """Flatten classwise_metrics output into a single row (for CSV)."""
    row = {f"{prefix}accuracy": m["accuracy"],
           f"{prefix}balanced_accuracy": m["balanced_accuracy"],
           f"{prefix}cohen_kappa": m["cohen_kappa"]}
    for cls, d in m["per_class"].items():
        for k, v in d.items():
            row[f"{prefix}{cls}_{k}"] = v
    return row


# --------------------------------------------------------------------------- #
# Confidence intervals
# --------------------------------------------------------------------------- #
def bootstrap_ci(y_true, y_pred, metric=accuracy_score, n_boot=2000,
                 alpha=0.05, seed=0):
    """Percentile bootstrap CI for any (y_true, y_pred) metric."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = metric(y_true[idx], y_pred[idx])
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return float(metric(y_true, y_pred)), lo, hi


def normal_ci(values: Sequence[float], alpha=0.05):
    """Normal-approx mean ± z*se across repeated measurements (e.g. seeds)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean())
    if n < 2:
        return mean, float("nan"), float("nan"), 0.0
    sd = float(v.std(ddof=1))
    z = 1.959963985 if abs(alpha - 0.05) < 1e-9 else _z_for(alpha)
    half = z * sd / sqrt(n)
    return mean, mean - half, mean + half, sd


def _z_for(alpha):
    from scipy.stats import norm
    return float(norm.ppf(1 - alpha / 2))


def aggregate_seeds(values: Sequence[float], alpha=0.05) -> dict:
    """mean ± std with 95% CI across seeds (R2-3/R2-4)."""
    mean, lo, hi, sd = normal_ci(values, alpha)
    return {"mean": mean, "std": sd, "ci_low": lo, "ci_high": hi,
            "n": len(values)}


# --------------------------------------------------------------------------- #
# McNemar's exact test (R1-1)
# --------------------------------------------------------------------------- #
def mcnemar_test(y_true, pred_a, pred_b):
    """
    Exact McNemar test comparing two classifiers on the same test set.
    Returns dict with discordant counts n01/n10 and the exact binomial p-value.
    n01 = A wrong & B right ; n10 = A right & B wrong.
    """
    y_true = np.asarray(y_true)
    a_correct = np.asarray(pred_a) == y_true
    b_correct = np.asarray(pred_b) == y_true
    n01 = int(np.sum(~a_correct & b_correct))
    n10 = int(np.sum(a_correct & ~b_correct))
    n = n01 + n10
    if n == 0:
        p = 1.0
    else:
        from scipy.stats import binom
        k = min(n01, n10)
        p = float(min(1.0, 2.0 * binom.cdf(k, n, 0.5)))
    return {"n01": n01, "n10": n10, "discordant": n, "p_value": p}


def build_significance_csv(records: list, path) -> pd.DataFrame:
    """
    records: list of dicts with keys subject, seed, method_a, method_b,
             acc_a, acc_b, and the mcnemar dict fields merged in.
    Writes significance_summary.csv and returns the DataFrame.
    """
    df = pd.DataFrame(records)
    if path is not None:
        df.to_csv(path, index=False)
    return df


def build_winloss_table(sig_df: pd.DataFrame, reference="DAG-SA",
                        alpha=0.05) -> pd.DataFrame:
    """
    Paired win/tie/loss of `reference` vs each other method (R2-2).
    A 'win' requires higher accuracy AND McNemar p < alpha; otherwise 'tie'.
    """
    rows = []
    for method, g in sig_df.groupby("method_b"):
        w = t = l = 0
        for _, r in g.iterrows():
            sig = r["p_value"] < alpha
            if r["acc_a"] > r["acc_b"] and sig:
                w += 1
            elif r["acc_a"] < r["acc_b"] and sig:
                l += 1
            else:
                t += 1
        rows.append({"reference": reference, "vs": method,
                     "wins": w, "ties": t, "losses": l, "n": len(g)})
    return pd.DataFrame(rows)
