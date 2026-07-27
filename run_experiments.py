"""
run_experiments.py
==================
Unified multi-seed experiment driver (Reviewer 2 #3 is the crux: EVERY method
is evaluated on the SAME eight seeds and reported as mean ± std ± 95% CI).

Pipeline per (dataset, subject, seed):
  1. load raw epochs via datasets_io.load_dataset
  2. stratified Train/Val/Test split (seeded) OR nested repeated k-fold (R1-2)
  3. build CSP/CTP feature dicts (fit on TRAIN only — no leakage, R2-5)
  4. pre-train the classifier pool
  5. DAG-SA search on VAL, evaluate on TEST
  6. baselines B1/B2/B3 + EEGNet + Riemannian on the identical split
  7. class-wise metrics + kappa (R2-10); McNemar DAG-SA vs each baseline (R1-1)

Outputs (all under RESULTS_DIR/<experiment>/, never the CWD):
  * baseline_comparison_multiseed.csv   (mean ± std ± 95% CI per method) R2-3/2-4
  * significance_summary.csv            (McNemar p, discordant pairs)     R1-1
  * winloss_summary.csv                 (win/tie/loss w/ significance)    R2-2
  * exp2b_baseline_comparison.(pdf|png) (REAL results, not a flowchart)   R1-9
  * per_seed_results.csv, confusion_*.pdf, convergence_*.pdf
  * config_used.yaml, leakage_audit.txt, search_space.txt

Use `run_experiment(...)` programmatically (smoke_test does), or the CLI.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split,
                                     RepeatedStratifiedKFold)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import env_utils
import datasets_io
import dag_core
import baselines as B
import metrics_utils as M

DAG_SA = "DAG-SA"


# --------------------------------------------------------------------------- #
# One split → predictions for every method
# --------------------------------------------------------------------------- #
def evaluate_all_methods(splits, fs, cfg, seed, name, paths,
                         class_names, tiny=False, run_baselines=True,
                         member_constraint=None, members=None, use_reheat=True,
                         sa_iters=None, verbose=False):
    """
    splits = (Xtr_raw, ytr, Xval_raw, yval, Xte_raw, yte)
    Returns: dict method -> {"y_true", "pred", "meta"}, and the SA optimizer.
    """
    Xtr_raw, ytr, Xval_raw, yval, Xte_raw, yte = splits
    comp = cfg["component_options"][0]
    member_constraint = member_constraint or cfg.get("member_constraint", "same_family")
    members = members or cfg["ensemble"]["members"]
    sa_cfg = cfg["sa"]
    sa_iters = sa_iters if sa_iters is not None else (5 if tiny else sa_cfg["iterations"])

    feats = (dag_core.FeatureType.CSP, dag_core.FeatureType.CSSP,
             dag_core.FeatureType.CTP)
    proc = dag_core.DataProcessor(fs, comp, cfg=cfg, feature_types=feats)
    Xtr, Xval, Xte = proc.process_splits(Xtr_raw, ytr, Xval_raw, Xte_raw)

    pool = dag_core.ClassifierPool(comp, cfg=cfg, feature_types=feats, seed=seed)
    pool.generate_pool()
    pool.pre_train_all(Xtr, ytr)

    results = {}

    # ---- DAG-SA ----------------------------------------------------------- #
    opt = dag_core.SimulatedAnnealingOptimizer(
        pool, Xval, yval, name, proc, seed=seed, members=members,
        member_constraint=member_constraint, use_reheat=use_reheat,
        checkpoint_dir=paths.checkpoint_dir)
    best = opt.run(iterations=sa_iters, temp=sa_cfg["temp"],
                   cooling_rate=sa_cfg["cooling_rate"], nreheat=sa_cfg["nreheat"],
                   checkpoint_every=sa_cfg["checkpoint_every"],
                   checkpoint_minutes=sa_cfg.get("checkpoint_minutes", 10),
                   verbose=verbose)
    results[DAG_SA] = {"y_true": yte, "pred": best.root.predict(Xte),
                       "meta": {"val_acc": opt.best_acc,
                                "topology": best.to_spec()}}

    if run_baselines:
        # ---- B1 single best ------------------------------------------------ #
        if cfg["baselines"]["single_best"]["enabled"]:
            pred, meta = B.single_best(pool, Xval, yval, Xte)
            results["single_best"] = {"y_true": yte, "pred": pred, "meta": meta}
        # ---- B2 full-pool soft vote --------------------------------------- #
        if cfg["baselines"]["full_pool_soft_vote"]["enabled"]:
            pred, meta = B.full_pool_soft_vote(pool, Xte)
            results["full_pool_soft_vote"] = {"y_true": yte, "pred": pred, "meta": meta}
        # ---- B3 random search (budget matched to SA) ---------------------- #
        if cfg["baselines"]["random_search"]["enabled"]:
            rs_iters = sa_iters if tiny else cfg["random_search"]["iterations"]
            pred, meta = B.random_search(pool, Xval, yval, Xte, rs_iters, seed,
                                         members=members,
                                         member_constraint=member_constraint)
            results["random_search"] = {"y_true": yte, "pred": pred, "meta": meta}
        # ---- EEGNet (CNN) -------------------------------------------------- #
        if cfg["baselines"]["eegnet"]["enabled"] and B.eegnet_available():
            try:
                ec = cfg["baselines"]["eegnet"]
                net = B.EEGNetClassifier(
                    n_classes=len(set(ytr.tolist())), fs=fs,
                    epochs=3 if tiny else ec["epochs"], lr=ec["lr"],
                    batch_size=ec["batch_size"], seed=seed)
                net.fit(Xtr_raw, ytr)
                results["EEGNet"] = {"y_true": yte, "pred": net.predict(Xte_raw),
                                     "meta": {}}
            except Exception as e:
                print(f"    [EEGNet] skipped: {e}")
        elif cfg["baselines"]["eegnet"]["enabled"]:
            print("    [EEGNet] torch not installed — skipped.")
        # ---- Riemannian (non-CNN) ----------------------------------------- #
        if cfg["baselines"]["riemannian"]["enabled"] and B.riemannian_available():
            try:
                rm = B.build_riemannian(cfg["baselines"]["riemannian"]["estimator"], seed)
                rm.fit(Xtr_raw, ytr)
                results["Riemannian"] = {"y_true": yte, "pred": rm.predict(Xte_raw),
                                         "meta": {}}
            except Exception as e:
                print(f"    [Riemannian] skipped: {e}")
        elif cfg["baselines"]["riemannian"]["enabled"]:
            print("    [Riemannian] pyriemann not installed — skipped.")

    return results, opt


# --------------------------------------------------------------------------- #
# Split builders
# --------------------------------------------------------------------------- #
def make_split(X, y, seed):
    return dag_core.three_way_split(X, y, seed)


def make_cv_splits(X, y, cfg, seed):
    """Nested repeated k-fold: outer test fold, inner val carved from train."""
    rskf = RepeatedStratifiedKFold(n_splits=cfg["cv"]["n_splits"],
                                   n_repeats=cfg["cv"]["n_repeats"],
                                   random_state=seed)
    for tr_idx, te_idx in rskf.split(X, y):
        Xtr_all, ytr_all = X[tr_idx], y[tr_idx]
        Xtr, Xval, ytr, yval = train_test_split(
            Xtr_all, ytr_all, test_size=0.2, stratify=ytr_all, random_state=seed)
        yield (Xtr, ytr, Xval, yval, X[te_idx], y[te_idx])


# --------------------------------------------------------------------------- #
# Atomic, resumable on-disk stores (Section B.3 / D.3)
# --------------------------------------------------------------------------- #
def _torch_cuda():
    """True if a CUDA GPU is available (affects the runtime estimate wording)."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _atomic_write_csv(df, path):
    """Write a DataFrame to `path` atomically (temp file + os.replace).

    Recreates the parent directory first — guards against Google Drive's
    delete-then-recreate mount lag on Colab, where an mkdir'd results dir may
    not be materialised yet when the first write lands.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _append_rows(path, rows):
    """Append `rows` (list of dicts) to a CSV, atomically, preserving prior rows.

    Reading the existing file back before rewriting means a Colab disconnect
    after any completed (subject, seed) unit never loses work already on disk.
    """
    path = Path(path)
    new = pd.DataFrame(rows)
    if path.exists():
        try:
            old = pd.read_csv(path)
            new = pd.concat([old, new], ignore_index=True)
        except Exception:
            pass
    _atomic_write_csv(new, path)
    return new


def _completed_units(path):
    """Return the set of (subject, seed) pairs already present on disk."""
    path = Path(path)
    done = set()
    if path.exists():
        try:
            df = pd.read_csv(path)
            for _, r in df[["subject", "seed"]].drop_duplicates().iterrows():
                done.add((str(r["subject"]), int(r["seed"])))
        except Exception:
            pass
    return done


def _write_runtime_report(path, experiment, protocol, dataset, total_units,
                          done_count, unit_times, elapsed):
    """Write per-unit timings + measured/projected totals (Section C.4)."""
    avg = (sum(unit_times) / len(unit_times)) if unit_times else float("nan")
    remaining = max(0, total_units - done_count) * (avg if unit_times else 0.0)
    lines = [
        "Runtime report (Section C)",
        "==========================",
        f"experiment      : {experiment}",
        f"dataset         : {dataset}",
        f"protocol        : {protocol}",
        "unit definition : one (subject, seed) pair",
        f"total units     : {total_units}",
        f"units completed : {done_count}",
        f"mean unit time  : {avg:.2f} s" if unit_times else "mean unit time  : n/a",
        f"elapsed         : {elapsed:.1f} s ({elapsed/60:.2f} min)",
        f"est. remaining  : {remaining:.1f} s ({remaining/60:.2f} min)",
        f"projected total : {(elapsed + remaining):.1f} s "
        f"({(elapsed + remaining)/60:.2f} min)",
        "",
        "per-unit timings (seconds, in completion order):",
    ]
    lines += [f"  unit {i+1}: {t:.2f}" for i, t in enumerate(unit_times)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Main experiment
# --------------------------------------------------------------------------- #
def run_experiment(dataset="ds1", subjects=None, seeds=None, protocol=None,
                   experiment="exp2b", project_root=None, config_path=None,
                   dataset_dir=None, tiny=False, variant="binary",
                   window=None, verbose=False, resume=True):
    paths, cfg = env_utils.setup_environment(project_root, config_path)
    exp_dir = paths.results_dir / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(exp_dir / "config_used.yaml")

    protocol = protocol or cfg.get("eval_protocol", "split")
    seeds = seeds or cfg["seeds"]
    ddir = Path(dataset_dir) if dataset_dir else paths.dataset_dir
    if subjects is None:
        subjects = (datasets_io.available_subjects(dataset, ddir)
                    or cfg.get("subjects_ds1"))

    per_seed_path = exp_dir / "per_seed_results.csv"
    sig_path = exp_dir / "significance_summary.csv"
    # Resume: skip (subject, seed) units already written to disk (Section B.3).
    done_units = _completed_units(per_seed_path) if resume else set()
    if done_units:
        print(f"[resume] {len(done_units)} (subject, seed) unit(s) already on disk "
              f"in {per_seed_path.name}; they will be skipped.")

    convergence = {}
    # Runtime estimation bookkeeping (Section C).
    total_units = len(subjects) * len(seeds)
    unit_times, run_t0 = [], time.monotonic()
    runtime_path = exp_dir / "runtime_report.txt"
    done_count = len(done_units)

    # ---- ETA at start (Section C.2) -------------------------------------- #
    sa_iters_full = (5 if tiny else cfg["sa"]["iterations"])
    n_remaining = total_units - done_count
    print(f"[eta] {experiment}: {total_units} units "
          f"({len(subjects)} subjects x {len(seeds)} seeds), "
          f"{n_remaining} to run | protocol={protocol}, "
          f"SA budget={sa_iters_full} iters"
          f"{', ' + str(cfg['cv']['n_splits']*cfg['cv']['n_repeats']) + ' folds/unit' if protocol == 'cv' else ''}. "
          f"Per-unit time is calibrated from the first completed unit and the "
          f"total wall-clock estimate is refined live below.")

    for subject in subjects:
        X, y, fs, ch_names, class_names = datasets_io.load_dataset(
            dataset, ddir, subject, window=window, variant=variant)
        name = f"{dataset}_{subject}"
        print(f"\n=== {name} | {X.shape[0]} trials | fs={fs} | protocol={protocol} ===")

        # search-space size (once per subject) — R2-5a
        pool_size_est = (len(list(dag_core.FrequencyBand)) * 2 *
                         len(cfg["component_options"][0]) *
                         (len(cfg["svm_grid"]) + len(cfg["lda_grid"])))
        ss = dag_core.count_search_space(pool_size_est, cfg["ensemble"]["members"],
                                         len(cfg["ensemble"]["operators"]))
        (exp_dir / "search_space.txt").write_text(
            "Search-space size estimate (R2-5a)\n" +
            "\n".join(f"{k}: {v}" for k, v in ss.items()) + "\n")

        for seed in seeds:
            if (str(subject), int(seed)) in done_units:
                continue
            env_utils.seed_everything(seed)
            unit_rows, unit_sig = [], []
            unit_t0 = time.monotonic()

            if protocol == "cv":
                # R2-3 fairness fix: EVERY method is evaluated on EVERY fold
                # (previously baselines ran on fold 0 only, so DAG-SA was
                # averaged over folds while baselines were single-fold).
                for fi, splits in enumerate(make_cv_splits(X, y, cfg, seed)):
                    res, opt = evaluate_all_methods(
                        splits, fs, cfg, seed, name, paths, class_names,
                        tiny=tiny, run_baselines=True, verbose=verbose)
                    for method, r in res.items():
                        m = M.classwise_metrics(r["y_true"], r["pred"], class_names)
                        unit_rows.append({"dataset": dataset, "subject": subject,
                                          "seed": seed, "fold": fi,
                                          "method": method, **M.flatten_metrics(m)})
                    # McNemar DAG-SA vs each baseline, per fold (same test set)
                    da = res[DAG_SA]
                    for method, r in res.items():
                        if method == DAG_SA:
                            continue
                        mc = M.mcnemar_test(da["y_true"], da["pred"], r["pred"])
                        unit_sig.append({
                            "dataset": dataset, "subject": subject, "seed": seed,
                            "fold": fi, "method_a": DAG_SA, "method_b": method,
                            "acc_a": float(np.mean(da["pred"] == da["y_true"])),
                            "acc_b": float(np.mean(r["pred"] == r["y_true"])), **mc})
                    convergence.setdefault(name, opt)
                    if tiny:
                        break
            else:  # single split
                splits = make_split(X, y, seed)
                res, opt = evaluate_all_methods(
                    splits, fs, cfg, seed, name, paths, class_names,
                    tiny=tiny, verbose=verbose)
                convergence.setdefault(name, opt)
                for method, r in res.items():
                    m = M.classwise_metrics(r["y_true"], r["pred"], class_names)
                    unit_rows.append({"dataset": dataset, "subject": subject,
                                      "seed": seed, "fold": 0,
                                      "method": method, **M.flatten_metrics(m)})
                # McNemar DAG-SA vs each baseline (same test set / seed)
                da = res[DAG_SA]
                for method, r in res.items():
                    if method == DAG_SA:
                        continue
                    mc = M.mcnemar_test(da["y_true"], da["pred"], r["pred"])
                    unit_sig.append({
                        "dataset": dataset, "subject": subject, "seed": seed,
                        "fold": 0, "method_a": DAG_SA, "method_b": method,
                        "acc_a": float(np.mean(da["pred"] == da["y_true"])),
                        "acc_b": float(np.mean(r["pred"] == r["y_true"])), **mc})
                # confusion matrix for DAG-SA (first seed only)
                if seed == seeds[0]:
                    _save_confusion(da["y_true"], da["pred"], class_names,
                                    exp_dir / f"confusion_{name}_seed{seed}.pdf", name)

            # ---- persist this (subject, seed) unit immediately (Section B.3) -- #
            _append_rows(per_seed_path, unit_rows)
            if unit_sig:
                _append_rows(sig_path, unit_sig)
            done_units.add((str(subject), int(seed)))

            # ---- runtime accounting + live ETA (Section C.2/C.3) ------------- #
            dt = time.monotonic() - unit_t0
            unit_times.append(dt)
            done_count += 1
            avg = sum(unit_times) / len(unit_times)
            remaining = max(0, total_units - done_count) * avg
            if len(unit_times) == 1:
                # Calibration pass complete (Section C.1): first real unit timed.
                print(f"    [eta] calibrated from unit 1 ({dt:.1f}s): estimated "
                      f"total wall-clock for {total_units} units ~ "
                      f"{(dt*total_units)/60:.1f} min "
                      f"(assumptions: same per-unit cost, "
                      f"{'GPU' if _torch_cuda() else 'CPU'}, SA={sa_iters_full} iters).")
            print(f"    [runtime] unit {done_count}/{total_units} done in "
                  f"{dt:.1f}s | avg {avg:.1f}s | est. remaining "
                  f"{remaining/60:.1f} min | elapsed "
                  f"{(time.monotonic()-run_t0)/60:.1f} min")
            _write_runtime_report(runtime_path, experiment, protocol, dataset,
                                  total_units, done_count, unit_times,
                                  time.monotonic() - run_t0)

    # ---- aggregate from the on-disk store (so resumed runs see everything) --- #
    per_seed_df = pd.read_csv(per_seed_path)
    summary = _aggregate(per_seed_df, cfg)
    summary.to_csv(exp_dir / "baseline_comparison_multiseed.csv", index=False)

    if sig_path.exists():
        sig_df = pd.read_csv(sig_path)
        M.build_winloss_table(sig_df).to_csv(exp_dir / "winloss_summary.csv", index=False)

    _plot_exp2b(summary, exp_dir / "exp2b_baseline_comparison")
    for cname, opt in convergence.items():
        _plot_convergence(opt, exp_dir / f"convergence_{cname}.pdf", cname)

    _write_leakage_audit(exp_dir / "leakage_audit.txt")
    print(f"\n[done] Results written to {exp_dir}")
    return summary, exp_dir


# --------------------------------------------------------------------------- #
# Aggregation & plots
# --------------------------------------------------------------------------- #
def _aggregate(df, cfg):
    rows = []
    alpha = cfg["ci"]["alpha"]
    for method, g in df.groupby("method"):
        # average across subjects+folds first per seed, then across seeds
        per_seed = g.groupby("seed")["accuracy"].mean().values
        agg = M.aggregate_seeds(per_seed, alpha)
        rows.append({"method": method,
                     "acc_mean": agg["mean"], "acc_std": agg["std"],
                     "acc_ci_low": agg["ci_low"], "acc_ci_high": agg["ci_high"],
                     "kappa_mean": g["cohen_kappa"].mean(),
                     "bal_acc_mean": g["balanced_accuracy"].mean(),
                     "n_seeds": agg["n"]})
    out = pd.DataFrame(rows).sort_values("acc_mean", ascending=False)
    return out


def _save_confusion(y_true, y_pred, class_names, path, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    ax.set_title(f"{title} — DAG-SA test confusion", fontsize=13)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _plot_exp2b(summary, path_stem):
    """REAL Exp 2-B results: accuracy per method with 95% CI (R1-9)."""
    s = summary.sort_values("acc_mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    yerr = [s["acc_mean"] - s["acc_ci_low"], s["acc_ci_high"] - s["acc_mean"]]
    colors = ["#d62728" if m == DAG_SA else "#1f77b4" for m in s["method"]]
    ax.barh(s["method"], s["acc_mean"], xerr=yerr, color=colors,
            capsize=4, alpha=0.85)
    ax.set_xlabel("Test accuracy (mean ± 95% CI across seeds)")
    ax.set_title("Experiment 2-B: DAG-SA vs baselines (identical splits/seeds)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        fig.savefig(str(path_stem) + ext, dpi=200)
    plt.close(fig)


def _plot_convergence(opt, path, title):
    if not opt.history["accuracy"]:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.plot(opt.history["accuracy"], label="val acc", alpha=0.6)
    a1.plot(opt.history["best_accuracy"], label="best", color="red", lw=2)
    a1.set_title(f"{title}: optimisation"); a1.legend()
    a2.plot(opt.history["temperature"], color="orange")
    a2.set_title(f"{title}: cooling")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _write_leakage_audit(path):
    path.write_text(
        "Leakage audit (Reviewer 2 #5)\n"
        "=============================\n"
        "1. CSP / CSSP spatial filters: fit on TRAIN only, then applied to\n"
        "   val and test (DataProcessor.process_splits). No test/val data\n"
        "   enters CSP estimation.\n"
        "2. Platt calibration: SVC(probability=True) performs internal CV on\n"
        "   the TRAIN fold only. It never sees val or test.\n"
        "3. Topology selection (simulated annealing) uses the VAL split; the\n"
        "   TEST split is untouched until the final single evaluation.\n"
        "4. Stacking meta-learner is fit on the same data used for topology\n"
        "   scoring (val); base learners it stacks were fit on train. This is\n"
        "   documented rather than hidden; for the strict protocol use\n"
        "   eval_protocol: cv (nested repeated k-fold), where topology\n"
        "   selection and final test are on disjoint outer folds.\n"
        "5. Standardisation for EEGNet/Riemannian is fit on TRAIN only.\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli():
    p = argparse.ArgumentParser(description="DAG-SA unified multi-seed driver")
    p.add_argument("--dataset", default="ds1", choices=["ds1", "ds2a"])
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--eval-protocol", dest="protocol", default=None,
                   choices=["split", "cv"])
    p.add_argument("--experiment", default="exp2b")
    p.add_argument("--project-root", default=None)
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--variant", default="binary", choices=["binary", "4class"])
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="recompute all units even if some are already on disk")
    a = p.parse_args()
    run_experiment(dataset=a.dataset, subjects=a.subjects, seeds=a.seeds,
                   protocol=a.protocol, experiment=a.experiment,
                   project_root=a.project_root, dataset_dir=a.dataset_dir,
                   variant=a.variant, tiny=a.tiny, verbose=a.verbose,
                   resume=a.resume)


if __name__ == "__main__":
    _cli()
