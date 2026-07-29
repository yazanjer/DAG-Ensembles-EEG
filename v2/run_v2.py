#!/usr/bin/env python3
"""
run_v2.py — incremental evaluation of the six improvements.

Every variant is run on **identical splits and seeds**, so each comparison is
paired trial-by-trial and McNemar's exact test applies directly. V0 reproduces
the published pipeline, which is what makes the comparison meaningful.

Usage
-----
    python run_v2.py --dataset ds1 --subjects a b f g --seeds 42 43 44 45 46 47 48 49
    python run_v2.py --dataset ds1 --variants V0_published V1_oof_objective
    python run_v2.py --dataset ds2a --subjects 1 2 3 4 5 6 7 8 9 --tiny   # quick check

Outputs (under <results>/<experiment>/)
---------------------------------------
    v2_per_seed_results.csv   one row per (subject, seed, variant): accuracy,
                              balanced accuracy, kappa, class-wise P/R/F1,
                              the selected topology and the search objective
    v2_summary.csv            mean +/- sd and 95% interval per variant
    v2_significance.csv       McNemar of every variant against V0, per unit
    v2_winloss.csv            significance-gated win/tie/loss against V0
    runtime_report.txt        wall-clock accounting
    config_used.yaml          the resolved configuration

Nothing here re-runs the published experiments: the V0 rows regenerate them
under the same seeds so that the paired tests are exact.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import dag_core
import datasets_io
import env_utils
import metrics_utils as M
import pool_ext
import selection
import variants as V

DAG_SA = "DAG-SA"


# --------------------------------------------------------------------------- #
# One (subject, seed) unit
# --------------------------------------------------------------------------- #
def run_unit(Xtr_raw, ytr, Xval_raw, yval, Xte_raw, yte, fs, cfg, seed,
             name, paths, variant_names, class_names=None, tiny=False,
             verbose=False):
    """Run every requested variant on one split. Returns a list of row dicts."""
    comp = cfg["component_options"][0]
    members = cfg["ensemble"]["members"]
    default_constraint = cfg.get("member_constraint", "same_family")
    sa_cfg = cfg["sa"]
    sa_iters = 5 if tiny else sa_cfg["iterations"]
    feats = (dag_core.FeatureType.CSP, dag_core.FeatureType.CSSP,
             dag_core.FeatureType.CTP)

    rows, preds = [], {}
    pool_cache, oof_cache = {}, {}

    for vname in variant_names:
        # AUDIT FIX E1 (2026-07-29) -- INVALIDATES THE WHOLE v2 CAMPAIGN.
        # seed_everything() was called once per (subject, seed) unit, OUTSIDE
        # this loop, and SimulatedAnnealingOptimizer never re-seeds (grep:
        # `random.seed` appears nowhere in dag_core.py -- the optimiser only
        # CONSUMES the global stream). So V0 burned ~1200+ random draws, V1
        # started wherever V0 left off, V2 wherever V1 left off, and so on:
        # every variant ran a DIFFERENT annealing trajectory. Only
        # V0_published began from a fresh seed state.
        #
        # The campaign's own control measurement says trajectory-only
        # variation moves a unit by 17 accuracy points (sd), which puts the
        # standard error of a 32-unit mean difference at ~3 points. That SE
        # is NOT a property of the method -- it is this bug. Under it,
        # "0 wins / 32 ties / 0 losses" is the mathematically expected
        # outcome whether or not a variant works, so none of the six ideas
        # was actually tested.
        #
        # Re-seeding here makes the variants paired on the SEARCH as well as
        # on the split, which is what the design claimed.
        env_utils.seed_everything(seed)
        v = V.VARIANTS[vname]
        member_constraint = v.get("member_constraint") or default_constraint
        sig = V.pool_signature(v)
        t0 = time.monotonic()

        # ---- feature pipeline + pool (shared between variants) ------------ #
        if sig not in pool_cache:
            vcfg = copy.deepcopy(cfg.data if hasattr(cfg, "data") else cfg)
            vcfg["preprocessing"]["csp_reg"] = v["csp_reg"]
            vcfg["pool"] = {"include_riemannian": v["include_riemannian"],
                            "include_riemannian_exact":
                                v["include_riemannian_exact"],
                            "include_eegnet": v["include_eegnet"],
                            "include_fbcsp": v["include_fbcsp"],
                            "fbcsp_select_k": v["fbcsp_select_k"],
                            "fs": fs,
                            "n_classes": len(set(np.asarray(ytr).tolist())),
                            "riemannian_estimator":
                                cfg.get("pool", {}).get("riemannian_estimator",
                                                        "oas")}
            proc = dag_core.DataProcessor(fs, comp, cfg=vcfg,
                                          feature_types=feats)
            Xtr, Xval, Xte = proc.process_splits(Xtr_raw, ytr, Xval_raw, Xte_raw)
            pool = dag_core.ClassifierPool(comp, cfg=vcfg, feature_types=feats,
                                           seed=seed)
            pool.generate_pool()
            Xtr, Xval, Xte, pool = pool_ext.build_views_and_pool(
                proc, pool, vcfg, comp, Xtr_raw, ytr, Xval_raw, Xte_raw,
                Xtr, Xval, Xte, seed=seed)
            n_ok = pool.pre_train_all(Xtr, ytr)
            if verbose:
                print(f"    pool[{sig}] {n_ok}/{len(pool.pool)} members fitted")
            pool_cache[sig] = (vcfg, proc, pool, Xtr, Xval, Xte)
        vcfg, proc, pool, Xtr, Xval, Xte = pool_cache[sig]

        # ---- selection objective ------------------------------------------ #
        scorer, keep_hist = None, 0
        if v["objective"] == "oof":
            if sig not in oof_cache:
                if verbose:
                    folds = 2 if tiny else v["oof_folds"]
                    print(f"    computing out-of-fold matrix ({folds} folds)")
                oof_cache[sig] = selection.precompute_oof(
                    Xtr_raw, ytr, fs, vcfg, feats, comp, seed,
                    n_folds=2 if tiny else v["oof_folds"], verbose=verbose)
            probs, y_ref = oof_cache[sig]
        elif v["objective"] == "val":
            probs, y_ref = _val_probs(pool, Xval), np.asarray(yval)
        else:
            probs = None

        if probs is not None:
            scorer = selection.OOFScorer(
                probs, y_ref,
                complexity_penalty=v["complexity_penalty"],
                diversity_weight=v["diversity_weight"],
                members_ref=members, stacking_cv=v["stacking_cv"], seed=seed)
        if v["one_se_rule"] or v["top_k_average"] > 1:
            keep_hist = max(20, int(v["top_k_average"]) * 4)

        # ---- annealed search ---------------------------------------------- #
        opt = dag_core.SimulatedAnnealingOptimizer(
            pool, Xval, yval, name, proc, seed=seed, members=members,
            member_constraint=member_constraint, use_reheat=True,
            checkpoint_dir=None, scorer=scorer, keep_history=keep_hist,
            init_family=v.get("init_family"))
        best = opt.run(iterations=sa_iters, temp=sa_cfg["temp"],
                       cooling_rate=sa_cfg["cooling_rate"],
                       nreheat=sa_cfg["nreheat"], checkpoint_every=10 ** 9,
                       checkpoint_minutes=None, verbose=False)

        # ---- finalise ------------------------------------------------------ #
        info = {}
        if scorer is not None:
            model, info = selection.finalise(opt, scorer, v)
        else:
            model = best

        pred = np.asarray(model.predict(Xte)) if hasattr(model, "predict") \
            else np.asarray(model.root.predict(Xte))
        preds[vname] = pred

        m = M.classwise_metrics(yte, pred, class_names)
        row = {"subject": name.split("_")[-1], "seed": seed, "variant": vname,
               "objective": v["objective"], "constraint": member_constraint,
               "search_score": float(opt.best_acc),
               "n_pool": len(pool.pool), "seconds": round(time.monotonic() - t0, 1),
               **M.flatten_metrics(m)}
        spec = model.to_spec() if hasattr(model, "to_spec") else {}
        row["topology"] = json.dumps(spec)[:900]
        row.update({f"sel_{k}": val for k, val in info.items()})
        rows.append(row)
        if verbose:
            print(f"    {vname:26s} acc={row['accuracy']:.3f} "
                  f"({row['seconds']}s)")
        del opt

    return rows, preds


def _val_probs(pool, Xval):
    """Member probabilities on the validation split (V0's own signal)."""
    out = {}
    for node in pool.pool:
        try:
            out[selection._member_key(node)] = node.predict_proba(Xval)
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------- #
# Campaign
# --------------------------------------------------------------------------- #
def run_campaign(dataset="ds1", subjects=None, seeds=None, variant_names=None,
                 experiment=None, project_root=None, config_path=None,
                 dataset_dir=None, tiny=False, variant="binary", window=None,
                 verbose=True, resume=True):
    paths, cfg = env_utils.setup_environment(project_root, config_path)
    variant_names = variant_names or V.DEFAULT_ORDER
    experiment = experiment or f"{dataset}_v2"
    exp_dir = paths.results_dir / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(exp_dir / "config_used.yaml")

    seeds = seeds or cfg["seeds"]
    ddir = Path(dataset_dir) if dataset_dir else paths.dataset_dir
    if subjects is None:
        subjects = (datasets_io.available_subjects(dataset, ddir)
                    or cfg.get("subjects_ds1"))

    # ---- resume: keep whatever a previous (interrupted) run completed ----
    rows, sig_rows, completed_units = [], [], set()
    prev = exp_dir / "v2_per_seed_results.csv"
    if resume and prev.exists():
        old = pd.read_csv(prev)
        keep = old[old.variant.isin(variant_names)] if "variant" in old else old
        rows = keep.to_dict("records")
        complete = (keep.groupby(["subject", "seed"]).variant.nunique()
                    == len(variant_names))
        completed_units = {(str(s), int(sd))
                           for (s, sd), ok in complete.items() if ok}
        sig_prev = exp_dir / "v2_significance.csv"
        if sig_prev.exists():
            sig_rows = pd.read_csv(sig_prev).to_dict("records")
        print(f"[v2] resuming: {len(completed_units)} (subject, seed) units already "
              f"complete, {len(rows)} rows kept")

    t_start = time.monotonic()
    unit_times, n_units = [], len(subjects) * len(seeds)
    print(f"[v2] {experiment}: {len(variant_names)} variants x {n_units} "
          f"(subject, seed) units")
    for vn in variant_names:
        print(f"      {vn:26s} {V.describe(vn)}")

    for subject in subjects:
        X, y, fs, ch_names, class_names = datasets_io.load_dataset(
            dataset, ddir, subject, window=window, variant=variant)
        name = f"{dataset}_{subject}"
        print(f"\n=== {name} | {X.shape[0]} trials | fs={fs} ===")

        for seed in seeds:
            if (str(subject), int(seed)) in completed_units:
                print(f"  seed {seed}: already complete, skipped")
                continue
            env_utils.seed_everything(seed)
            t0 = time.monotonic()
            splits = dag_core.three_way_split(X, y, seed)
            try:
                unit_rows, preds = run_unit(
                    *splits, fs, cfg, seed, name, paths, variant_names,
                    class_names=class_names, tiny=tiny, verbose=verbose)
            except Exception as e:                     # keep the campaign alive
                print(f"  !! {name} seed {seed} failed: "
                      f"{type(e).__name__}: {e}")
                import traceback; traceback.print_exc()
                _write(exp_dir, rows, sig_rows)
                continue
            rows += unit_rows

            # paired McNemar of every variant against V0 on the same trials
            yte = splits[5]
            if "V0_published" in preds:
                base = preds["V0_published"]
                for vn, pr in preds.items():
                    if vn == "V0_published":
                        continue
                    mc = M.mcnemar_test(yte, pr, base)
                    sig_rows.append({
                        "dataset": dataset, "subject": subject, "seed": seed,
                        "variant": vn, "reference": "V0_published",
                        "acc_variant": float(np.mean(pr == yte)),
                        "acc_reference": float(np.mean(base == yte)), **mc})

            unit_times.append(time.monotonic() - t0)
            done = len(unit_times)
            eta = (n_units - done) * float(np.mean(unit_times)) / 60.0
            print(f"  seed {seed}: {unit_times[-1]:.0f}s "
                  f"({done}/{n_units} units, ~{eta:.0f} min left)")
            _write(exp_dir, rows, sig_rows)

    _write(exp_dir, rows, sig_rows)
    _runtime(exp_dir, experiment, dataset, unit_times, t_start, n_units,
             variant_names)
    print(f"\n[v2] written to {exp_dir}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _write(exp_dir, rows, sig_rows):
    df = pd.DataFrame(rows)
    df.to_csv(exp_dir / "v2_per_seed_results.csv", index=False)
    if not len(df):
        return
    # summary: mean over seeds of the per-seed subject means (as in the paper)
    per_seed = df.groupby(["variant", "seed"])["accuracy"].mean().reset_index()
    out = []
    for vn, g in per_seed.groupby("variant"):
        v = g["accuracy"].values
        sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        half = 1.959963985 * sd / np.sqrt(len(v)) if len(v) > 1 else 0.0
        out.append({"variant": vn, "acc_mean": float(v.mean()), "acc_std": sd,
                    "ci_low": float(v.mean() - half),
                    "ci_high": float(v.mean() + half), "n_seeds": len(v),
                    "kappa_mean": float(
                        df[df.variant == vn]["cohen_kappa"].mean()),
                    "seconds_mean": float(
                        df[df.variant == vn]["seconds"].mean())})
    s = pd.DataFrame(out).sort_values("acc_mean", ascending=False)
    s.to_csv(exp_dir / "v2_summary.csv", index=False)

    if sig_rows:
        sg = pd.DataFrame(sig_rows)
        sg.to_csv(exp_dir / "v2_significance.csv", index=False)
        wl = []
        for vn, g in sg.groupby("variant"):
            sigd = g[g.p_value < 0.05]
            wl.append({"variant": vn, "reference": "V0_published",
                       "wins": int((sigd.acc_variant > sigd.acc_reference).sum()),
                       "losses": int((sigd.acc_variant < sigd.acc_reference).sum()),
                       "ties": int(len(g) - len(sigd)), "n": int(len(g)),
                       "median_p": float(g.p_value.median())})
        pd.DataFrame(wl).to_csv(exp_dir / "v2_winloss.csv", index=False)


def _runtime(exp_dir, experiment, dataset, unit_times, t_start, n_units,
             variant_names):
    el = time.monotonic() - t_start
    txt = ["Runtime report (v2 incremental study)",
           "=" * 38,
           f"experiment      : {experiment}",
           f"dataset         : {dataset}",
           f"variants        : {', '.join(variant_names)}",
           f"unit definition : one (subject, seed) pair, all variants",
           f"total units     : {n_units}",
           f"units completed : {len(unit_times)}",
           f"mean unit time  : {np.mean(unit_times):.2f} s" if unit_times else "",
           f"elapsed         : {el:.1f} s ({el/60:.2f} min)",
           "",
           "per-unit timings (seconds, in completion order):"]
    txt += [f"  unit {i+1}: {t:.2f}" for i, t in enumerate(unit_times)]
    (exp_dir / "runtime_report.txt").write_text("\n".join(txt) + "\n")


# --------------------------------------------------------------------------- #
def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ds1", choices=["ds1", "ds2a"])
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--variants", nargs="*", default=None,
                    choices=list(V.VARIANTS) + [None])
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--project-root", default=None)
    ap.add_argument("--variant", default="binary", choices=["binary", "4class"])
    ap.add_argument("--tiny", action="store_true",
                    help="5 SA iterations and 2 inner folds - wiring check only")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing partial run and start over")
    a = ap.parse_args()
    run_campaign(dataset=a.dataset, subjects=a.subjects, seeds=a.seeds,
                 variant_names=a.variants, experiment=a.experiment,
                 project_root=a.project_root, config_path=a.config,
                 dataset_dir=a.dataset_dir, tiny=a.tiny, variant=a.variant,
                 verbose=not a.quiet, resume=not a.no_resume)


if __name__ == "__main__":
    _cli()
