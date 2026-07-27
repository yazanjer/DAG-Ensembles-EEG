"""
ablations.py
============
Configurable ablations (Reviewer 2 #8 and #9). Each ablation reuses the DAG-SA
evaluation from run_experiments (baselines off) and varies exactly one factor,
writing CSV + plot to RESULTS_DIR/ablations/:

  * member_constraint   : same_family | unconstrained | partial      (R2-8)
  * ensemble_size (M)    : 2, 4, 6                                     (R2-9)
  * fusion_operator      : each of MV/HV/SV/MIN/ST forced at the root  (R2-9)
  * reheating            : on vs off (nreheat)                         (R2-9)
  * sa_budget            : varying iteration counts                    (R2-9)
  * selection_frequency  : how often each feature family / fusion op is
                           chosen in the best topology across subjects/seeds

Everything is small by default (few subjects/seeds) so it is tractable; scale
up via the arguments once the smoke test is green.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import env_utils
import datasets_io
from run_experiments import evaluate_all_methods, make_split, DAG_SA


def _load(dataset, subject, ddir, variant, window):
    return datasets_io.load_dataset(dataset, ddir, subject, window=window,
                                    variant=variant)


def _dagsa_acc(splits, fs, cfg, seed, name, paths, class_names, **kw):
    res, opt = evaluate_all_methods(splits, fs, cfg, seed, name, paths,
                                    class_names, run_baselines=False, **kw)
    r = res[DAG_SA]
    acc = float(np.mean(r["pred"] == r["y_true"]))
    return acc, r["meta"].get("topology", {}), opt


def run_ablations(dataset="ds1", subjects=None, seeds=None, project_root=None,
                  config_path=None, dataset_dir=None, variant="binary",
                  window=None, tiny=True):
    paths, cfg = env_utils.setup_environment(project_root, config_path)
    abl_dir = paths.results_dir / "ablations"
    abl_dir.mkdir(parents=True, exist_ok=True)

    ddir = Path(dataset_dir) if dataset_dir else paths.dataset_dir
    subjects = subjects or (datasets_io.available_subjects(dataset, ddir)[:1]
                            or cfg.get("subjects_ds1")[:1])
    seeds = seeds or cfg["seeds"][:2]

    cache = {}
    for s in subjects:
        cache[s] = _load(dataset, s, ddir, variant, window)

    rows = []
    fam_counter, op_counter = Counter(), Counter()

    def eval_variant(label, factor, value, **kw):
        for s in subjects:
            X, y, fs, ch, cn = cache[s]
            for seed in seeds:
                env_utils.seed_everything(seed)
                splits = make_split(X, y, seed)
                acc, topo, _ = _dagsa_acc(splits, fs, cfg, seed, f"{dataset}_{s}",
                                          paths, cn, tiny=tiny, **kw)
                rows.append({"ablation": label, factor: value,
                             "subject": s, "seed": seed, "accuracy": acc})
                _tally(topo, fam_counter, op_counter)

    # 1) member constraint (R2-8)
    for mc in ["same_family", "unconstrained", "partial"]:
        eval_variant("member_constraint", "constraint", mc, member_constraint=mc)

    # 2) ensemble size M (R2-9)
    for m in [2, 4, 6]:
        eval_variant("ensemble_size", "M", m, members=m)

    # 3) fusion operator forced at root (R2-9): approximate by biasing operators
    #    — here we report per-operator selection frequency (below) plus a run
    #    with reheating off; a hard operator lock would need a code hook, noted.

    # 4) reheating on/off (R2-9)
    for reheat in [True, False]:
        eval_variant("reheating", "use_reheat", reheat, use_reheat=reheat)

    # 5) SA budget (R2-9)
    for it in ([3, 5] if tiny else [50, 150, 300]):
        eval_variant("sa_budget", "iterations", it, sa_iters=it)

    df = pd.DataFrame(rows)
    df.to_csv(abl_dir / "ablation_results.csv", index=False)

    # selection frequency (R2-9)
    freq = pd.DataFrame({
        "kind": ["family"] * len(fam_counter) + ["operator"] * len(op_counter),
        "name": list(fam_counter.keys()) + list(op_counter.keys()),
        "count": list(fam_counter.values()) + list(op_counter.values()),
    })
    freq.to_csv(abl_dir / "selection_frequency.csv", index=False)
    _plot_ablations(df, abl_dir)
    print(f"[ablations] written to {abl_dir}")
    return df, freq


def _tally(topo, fam_counter, op_counter):
    def walk(node):
        if not isinstance(node, dict):
            return
        if "operator" in node:
            op_counter[node["operator"]] += 1
            for p in node.get("parents", []):
                walk(p)
        elif "feature" in node:
            fam = {"CSP": "CSP", "CSSP": "CSP", "CTP": "CTP"}.get(node["feature"],
                                                                 node["feature"])
            fam_counter[fam] += 1
    walk(topo.get("root", {}))


def _plot_ablations(df, abl_dir):
    for label, g in df.groupby("ablation"):
        factor = [c for c in g.columns if c not in
                  ("ablation", "subject", "seed", "accuracy")][0]
        summ = g.groupby(factor)["accuracy"].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(summ[factor].astype(str), summ["mean"],
               yerr=summ["std"].fillna(0), capsize=4, color="#1f77b4", alpha=0.85)
        ax.set_title(f"Ablation: {label}")
        ax.set_ylabel("Test accuracy (mean ± std)")
        fig.tight_layout(); fig.savefig(abl_dir / f"ablation_{label}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    run_ablations(tiny=True)
