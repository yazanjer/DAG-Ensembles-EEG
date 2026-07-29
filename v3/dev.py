"""
dev.py -- DEVELOPMENT harness.

Rule of this file: **no test fold is ever scored here.**

Every number produced by dev.py is a cross-validated estimate computed inside
the training portion of a development split, using development seeds (900+)
that are disjoint from the final evaluation seeds (42-51). Design decisions
(band set, regularisation strengths, fusion form) are made from these numbers
only. The confirmatory protocol in `protocol.py` is then run once.

This is the mechanism that makes "do not tune on test results and re-report"
enforceable rather than aspirational.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

import data as D
import pipeline as P

DEV_SEEDS = [900, 901, 902]
TEST_SIZE = 0.30


def dev_split(y, seed):
    """Same split function the final protocol uses; test indices are RETURNED
    but dev.py never scores them."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=seed,
                              stratify=y)
    return np.sort(tr), np.sort(te)


def inner_cv_score(tb, train_idx, source_bands, cfg, seed, folds=5,
                   src_models=None):
    """
    Accuracy of the full method estimated by k-fold CV *inside* train_idx.
    Each fold re-runs fit() end to end, so the alignment reference, all base
    learners and the meta-learner are refit on the fold's training part.

    `src_models` may be passed in because source models depend only on the
    OTHER subjects' data -- never on the target's split -- so refitting them
    per fold would be identical work for an identical result.
    """
    y = tb.y[train_idx]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    correct = 0
    for a, b in skf.split(np.zeros(len(train_idx)), y):
        m = P.ARTS(cfg, seed=seed)
        if src_models is not None:
            m._src_models = src_models
        else:
            m.fit_sources(source_bands)
        m.fit(tb, train_idx[a], source_bands)
        pred = m.predict(tb, train_idx[b])
        correct += (pred == tb.y[train_idx[b]]).sum()
    return correct / len(train_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds1-dir", required=True)
    ap.add_argument("--out", default="dev_results.json")
    ap.add_argument("--subjects", default="a,b,f,g")
    ap.add_argument("--seeds", default="900")
    ap.add_argument("--configs", default="full")
    args = ap.parse_args()

    subs = args.subjects.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    print("[dev] loading + precomputing band covariances ...", flush=True)
    bands_by_sub = {}
    for s in D.DS1_ALL:
        d = D.load_ds1(args.ds1_dir, s)
        bands_by_sub[s] = P.SubjectBands.build(d["X"], d["y"], d["fs"], s)
        print(f"   {s}: {d['X'].shape}", flush=True)

    CONFIGS = {
        "full":        P.ARTSConfig(),
        "no_transfer": P.ARTSConfig(use_transfer=False),
        "no_align":    P.ARTSConfig(use_alignment=False),
        "mean_fusion": P.ARTSConfig(fusion="mean"),
        "best_view":   P.ARTSConfig(fusion="best"),
        "single_band": P.ARTSConfig(bands=(P.RIEMANNIAN_BASELINE_BAND,),
                                    use_transfer=False, fusion="mean"),
    }
    want = args.configs.split(",")
    # Resumable: each invocation appends whatever it manages to finish, so the
    # sweep can be driven by repeated short calls.
    out = []
    if Path(args.out).exists():
        out = json.loads(Path(args.out).read_text())
    done = {(r["config"], r["seed"], r["subject"]) for r in out}
    src_cache = {}
    for name in want:
        cfg = CONFIGS[name]
        for seed in seeds:
            for s in subs:
                if (name, seed, s) in done:
                    continue
                tb = bands_by_sub[s]
                srcs = [bands_by_sub[o] for o in D.DS1_ALL if o != s]
                key = (s, tuple(cfg.bands), cfg.use_alignment,
                       cfg.use_transfer, cfg.C_src, seed)
                if cfg.use_transfer and key not in src_cache:
                    t0 = time.time()
                    src_cache[key] = P.ARTS(cfg, seed=seed).fit_sources(
                        srcs)._src_models
                    print(f"   [src] {s} fitted in "
                          f"{time.time()-t0:.0f}s", flush=True)
                sm = src_cache.get(key) if cfg.use_transfer else {}
                tr, _te = dev_split(tb.y, seed)
                t0 = time.time()
                acc = inner_cv_score(tb, tr, srcs, cfg, seed, src_models=sm)
                rec = dict(config=name, seed=seed, subject=s,
                           inner_cv_acc=float(acc), secs=round(time.time() - t0, 1))
                out.append(rec)
                print(json.dumps(rec), flush=True)
                Path(args.out).write_text(json.dumps(out, indent=1))

    # summary
    import collections
    agg = collections.defaultdict(list)
    for r in out:
        agg[r["config"]].append(r["inner_cv_acc"])
    print("\n[dev] inner-CV means (NO test data involved)")
    for k, v in agg.items():
        print(f"   {k:14s} {np.mean(v):.4f}  (n={len(v)})")


if __name__ == "__main__":
    main()
