"""
dev2a.py -- development harness on Dataset 2a (inner-CV inside train only).

Dataset 2a is where cross-subject transfer can pay: all nine subjects perform
the SAME four cued tasks, so a source model trained on other subjects predicts
a label that means the same thing for the target. That is not true of
Dataset 1, where the two classes differ by subject (left/foot for a and f,
left/right for the rest) -- pooling those subjects asks a source model to map
"foot" and "right hand" onto the same output.
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


def inner_cv(tb, train_idx, srcs, cfg, seed, folds=5, src_models=None,
             src_per=None):
    y = tb.y[train_idx]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    ok = 0
    for a, b in skf.split(np.zeros(len(train_idx)), y):
        m = P.ARTS(cfg, seed=seed)
        if src_models is not None:
            m._src_models = src_models
            if src_per is not None:
                m._src_per = src_per
        else:
            m.fit_sources(srcs)
        m.fit(tb, train_idx[a], srcs)
        ok += (m.predict(tb, train_idx[b]) == tb.y[train_idx[b]]).sum()
    return ok / len(train_idx)


CONFIGS = {
    "single_band":  P.ARTSConfig(bands=(P.RIEMANNIAN_BASELINE_BAND,),
                                 use_transfer=False, fusion="mean"),
    "fb_only":      P.ARTSConfig(use_transfer=False),
    "fb_mean":      P.ARTSConfig(use_transfer=False, fusion="mean"),
    "transfer":     P.ARTSConfig(),
    "transfer_noal": P.ARTSConfig(use_alignment=False),
    "sb_transfer":  P.ARTSConfig(bands=(P.RIEMANNIAN_BASELINE_BAND,)),
    "meta_c01":     P.ARTSConfig(C_meta=0.1),
    "meta_c001":    P.ARTSConfig(C_meta=0.01),
    "csp_only":     P.ARTSConfig(use_transfer=False, use_csp=True,
                                 bands=P.DEFAULT_BANDS),
    "fb_csp":       P.ARTSConfig(use_transfer=False, use_csp=True),
    "fb_csp_tr":    P.ARTSConfig(use_csp=True),
    "fb_csp_tr_c1": P.ARTSConfig(use_csp=True, C_meta=0.1),
    "srcper_wb":    P.ARTSConfig(src_mode="per_subject",
                                 src_bands=(P.RIEMANNIAN_BASELINE_BAND,)),
    "srcper_all":   P.ARTSConfig(src_mode="per_subject"),
    "srcper_wb_c1": P.ARTSConfig(src_mode="per_subject", C_meta=0.1,
                                 src_bands=(P.RIEMANNIAN_BASELINE_BAND,)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--variant", default="binary")
    ap.add_argument("--subjects", default="1,2,3,4,5,6,7,8,9")
    ap.add_argument("--seeds", default="900")
    ap.add_argument("--configs", default="single_band,fb_only,transfer")
    ap.add_argument("--out", default="/tmp/dev2a.json")
    a = ap.parse_args()

    subs = [int(x) for x in a.subjects.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]

    out = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else []
    done = {(r["config"], r["seed"], r["subject"], r["variant"]) for r in out}

    tbs = {}
    for s in D.DS2A_ALL:
        u = D.load_ds2a(a.dir, s, variant=a.variant)
        tbs[s] = P.SubjectBands.build(u["X"], u["y"], u["fs"], s)
    print(f"[dev2a] built {a.variant}", flush=True)

    src_cache = {}
    for name in a.configs.split(","):
        cfg = CONFIGS[name]
        for seed in seeds:
            for s in subs:
                if (name, seed, s, a.variant) in done:
                    continue
                tb = tbs[s]
                srcs = [tbs[o] for o in D.DS2A_ALL if o != s]
                key = (s, tuple(cfg.bands), cfg.use_alignment, cfg.C_src,
                       cfg.src_mode, tuple(cfg.src_bands or ()))
                sm = {}
                sp = None
                if cfg.use_transfer:
                    if key not in src_cache:
                        m0 = P.ARTS(cfg, seed=seed).fit_sources(srcs)
                        if cfg.src_mode == "per_subject":
                            m0.fit_sources_per_subject(srcs, cfg.src_bands)
                        src_cache[key] = (m0._src_models,
                                          getattr(m0, "_src_per", None))
                    sm, sp = src_cache[key]
                idx = np.arange(len(tb.y))
                tr, _ = train_test_split(idx, test_size=TEST_SIZE,
                                         random_state=seed, stratify=tb.y)
                tr = np.sort(tr)
                t0 = time.time()
                acc = inner_cv(tb, tr, srcs, cfg, seed, src_models=sm,
                               src_per=sp)
                rec = dict(config=name, seed=seed, subject=s,
                           variant=a.variant, inner_cv_acc=float(acc),
                           secs=round(time.time() - t0, 1))
                out.append(rec)
                print(json.dumps(rec), flush=True)
                Path(a.out).write_text(json.dumps(out, indent=1))

    import collections
    agg = collections.defaultdict(list)
    for r in out:
        agg[r["config"]].append(r["inner_cv_acc"])
    print("\n[dev2a] inner-CV means (train-only, no test data)")
    for k, v in sorted(agg.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"   {k:14s} {np.mean(v):.4f}  (n={len(v)})")


if __name__ == "__main__":
    main()
