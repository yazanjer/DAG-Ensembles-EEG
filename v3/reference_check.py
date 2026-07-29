"""
reference_check.py -- independent sanity check on the accuracy LEVEL.

Purpose: the manuscript reports 60-70% accuracy on BCI IV Dataset 1, which is
far below what the published literature obtains on this dataset with textbook
methods. The audit found that every non-CSP baseline was handed unfiltered
broadband epochs and that the CSP path used a causal `lfilter`. This script
asks whether the accuracy LEVEL itself is an artefact, using two completely
independent, textbook decoders written from scratch here:

    * CSP (6 components) + shrinkage LDA on log-variance, 8-30 Hz
    * Riemannian tangent space + L2 logistic regression, 8-30 Hz

If these land near the literature (~80-90% on ds1, ~75-85% on 2a binary) while
the repository lands at 66%, the repository's numbers are a pipeline artefact
rather than a property of the data.

This script scores held-out folds, so it is NOT part of the development
protocol for tuning the proposed method -- it is a diagnostic on the EXISTING
pipeline, run at seeds 0-4 which are disjoint from the final evaluation seeds.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import data as D
import pipeline as P


def csp_filters(Xb, y, n_comp=6):
    """Classic two-class CSP by generalised eigendecomposition of class covs."""
    cls = np.unique(y)
    C = []
    for c in cls:
        Xi = Xb[y == c]
        Ci = P.oas_cov(Xi).mean(axis=0)
        C.append(Ci / np.trace(Ci))
    w, V = eigh(C[0], C[0] + C[1])
    order = np.argsort(np.abs(w - 0.5))[::-1]
    V = V[:, order]
    k = n_comp // 2
    return np.concatenate([V[:, :k], V[:, -k:]], axis=1).T   # (n_comp, ch)


def csp_logvar(W, Xb):
    Z = np.einsum("kc,nct->nkt", W, Xb)
    v = Z.var(axis=2)
    return np.log(v / v.sum(axis=1, keepdims=True) + 1e-12)


def run(unit, folds=5, seed=0):
    X, y, fs = unit["X"], unit["y"], unit["fs"]
    Xb = P.bandpass(X, 8.0, 30.0, fs)
    covs = P.oas_cov(Xb)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    acc_csp, acc_ts = [], []
    for tr, te in skf.split(X, y):
        # --- CSP + shrinkage LDA (one-vs-rest for >2 classes) ---
        cls = np.unique(y)
        if len(cls) == 2:
            W = csp_filters(Xb[tr], y[tr])
        else:
            W = np.concatenate([csp_filters(Xb[tr], (y[tr] == c).astype(int))
                                for c in cls], axis=0)
        Ftr, Fte = csp_logvar(W, Xb[tr]), csp_logvar(W, Xb[te])
        m = LDA(solver="lsqr", shrinkage="auto").fit(Ftr, y[tr])
        acc_csp.append((m.predict(Fte) == y[te]).mean())

        # --- Riemannian tangent space + L2 LR ---
        ref = P.euclidean_reference(covs[tr])
        Z = P.tangent(P.align(covs, ref))
        lr = LogisticRegression(C=0.1, max_iter=500,
                                class_weight="balanced").fit(Z[tr], y[tr])
        acc_ts.append((lr.predict(Z[te]) == y[te]).mean())
    return float(np.mean(acc_csp)), float(np.mean(acc_ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds1-dir")
    ap.add_argument("--ds2a-dir")
    ap.add_argument("--which", default="ds1")
    ap.add_argument("--subjects", default="")
    args = ap.parse_args()

    rows = []
    if args.which == "ds1":
        subs = args.subjects.split(",") if args.subjects else D.DS1_ALL
        for s in subs:
            u = D.load_ds1(args.ds1_dir, s)
            c, t = run(u)
            tag = "artificial" if s in D.DS1_ARTIFICIAL else "real"
            rows.append((f"ds1-{s} ({tag})", c, t, "/".join(u["class_names"])))
            print(f"{rows[-1][0]:22s} CSP+LDA {c:.3f}   TS+LR {t:.3f}   "
                  f"[{rows[-1][3]}]", flush=True)
    else:
        subs = ([int(x) for x in args.subjects.split(",")]
                if args.subjects else D.DS2A_ALL)
        variant = "4class" if args.which.endswith("4class") else "binary"
        for s in subs:
            u = D.load_ds2a(args.ds2a_dir, s, variant=variant)
            c, t = run(u)
            rows.append((f"2a-{s} ({variant})", c, t, ""))
            print(f"{rows[-1][0]:22s} CSP+LDA {c:.3f}   TS+LR {t:.3f}",
                  flush=True)

    a = np.array([[r[1], r[2]] for r in rows])
    print(f"\nMEAN                   CSP+LDA {a[:,0].mean():.3f}   "
          f"TS+LR {a[:,1].mean():.3f}   (n={len(rows)})")


if __name__ == "__main__":
    main()
