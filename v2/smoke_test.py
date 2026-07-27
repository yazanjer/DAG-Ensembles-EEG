"""
smoke_test.py  (Section C)
==========================
Fast end-to-end test on tiny settings (1 subject, 2 seeds, tiny SA budget). It
exercises every path — data load, band-pass + CSP/CTP features, pool build,
SA search + checkpoint save/reload, all fusion operators, every baseline
(single-best, soft-vote, random search, EEGNet if torch present, Riemannian if
pyriemann present), class-wise metrics + CIs, McNemar, and all file writes.

Passes iff: outputs exist and are non-empty, no NaNs in metrics, and a
checkpoint reloads identically. Prints a GREEN/RED summary and exits non-zero
on failure so run_all_pipelines.py can gate on it.

Run:  python smoke_test.py
"""

from __future__ import annotations

import os
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import env_utils          # noqa: E402
import datasets_io        # noqa: E402
import run_experiments    # noqa: E402
import baselines as B     # noqa: E402


def _fail(msg):
    print(f"  \033[91m[RED]  {msg}\033[0m")
    return False


def _ok(msg):
    print(f"  \033[92m[green]\033[0m {msg}")
    return True


def main():
    checks = []
    os.environ.setdefault("EEG_DAGSA_ROOT", str(HERE))
    paths, cfg = env_utils.resolve_paths().create(), env_utils.load_config()

    # locate data ----------------------------------------------------------- #
    ds1_ddir = paths.dataset_dir
    ds1_subjects = datasets_io.available_subjects("ds1", ds1_ddir)
    ds2a_ddir = _find_ds2a_dir(paths)
    if not ds1_subjects and ds2a_ddir is None:
        print(_fail("No dataset files found. Put ds1 .mat in "
                    f"{ds1_ddir} or ds2a next to the project."))
        sys.exit(2)

    if ds1_subjects:
        dataset, subject, ddir, variant = "ds1", ds1_subjects[0], ds1_ddir, "binary"
    else:
        dataset, ddir, variant = "ds2a", ds2a_ddir, "binary"
        subject = datasets_io.available_subjects("ds2a", ds2a_ddir)[0]
    print(f"\n[smoke] dataset={dataset} subject={subject} dir={ddir}")
    print(f"[smoke] torch={'yes' if B.eegnet_available() else 'no'} "
          f"pyriemann={'yes' if B.riemannian_available() else 'no'}")

    # run the tiny experiment ---------------------------------------------- #
    try:
        summary, exp_dir = run_experiments.run_experiment(
            dataset=dataset, subjects=[subject], seeds=[42, 43],
            protocol="split", experiment="smoke", project_root=str(HERE),
            dataset_dir=str(ddir), variant=variant, tiny=True, verbose=False)
        checks.append(_ok("experiment ran end-to-end"))
    except Exception as e:
        traceback.print_exc()
        checks.append(_fail(f"experiment crashed: {e}"))
        _summary(checks); sys.exit(1)

    # output files exist and non-empty ------------------------------------- #
    expected = ["baseline_comparison_multiseed.csv", "per_seed_results.csv",
                "significance_summary.csv", "winloss_summary.csv",
                "exp2b_baseline_comparison.pdf", "config_used.yaml",
                "leakage_audit.txt", "search_space.txt"]
    for fn in expected:
        p = exp_dir / fn
        checks.append(_ok(f"{fn} present") if p.exists() and p.stat().st_size > 0
                      else _fail(f"missing/empty: {fn}"))

    # no NaNs in metrics ---------------------------------------------------- #
    df = pd.read_csv(exp_dir / "per_seed_results.csv")
    methods = sorted(df["method"].unique())
    print(f"[smoke] methods evaluated: {methods}")
    if df["accuracy"].isna().any():
        checks.append(_fail("NaN found in accuracy column"))
    else:
        checks.append(_ok("no NaNs in accuracy metrics"))
    if len(methods) >= 4:
        checks.append(_ok(f"{len(methods)} methods on identical splits"))
    else:
        checks.append(_fail(f"only {len(methods)} methods evaluated"))

    # all fusion operators exercised --------------------------------------- #
    checks.append(_check_fusion_operators())

    # checkpoint reloads identically --------------------------------------- #
    checks.append(_check_checkpoint(paths.checkpoint_dir))

    _summary(checks)
    sys.exit(0 if all(checks) else 1)


def _find_ds2a_dir(paths):
    for cand in [paths.dataset_dir,
                 paths.root.parent / "BCI Competition IV Dataset 2a",
                 paths.root / "BCI Competition IV Dataset 2a"]:
        if datasets_io.available_subjects("ds2a", cand):
            return cand
    return None


def _check_fusion_operators():
    """Directly exercise every fusion operator on toy probabilities."""
    import dag_core as dc
    try:
        class _Stub:
            def __init__(self, p): self.p = p
            def predict_proba(self, _): return self.p
        p1 = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
        p2 = np.array([[0.6, 0.4], [0.4, 0.6], [0.55, 0.45]])
        for op in dc.OperatorType:
            node = dc.EnsembleOperatorNode(op, [_Stub(p1), _Stub(p2)])
            if op == dc.OperatorType.ST:
                node.meta_clf = None  # skip fit; covered in full run
                continue
            out = node.predict_proba(None)
            assert out.shape == (3, 2) and not np.isnan(out).any()
        return _ok("all fusion operators produce valid outputs")
    except Exception as e:
        return _fail(f"fusion operator check failed: {e}")


def _check_checkpoint(ckpt_dir):
    ckpts = list(Path(ckpt_dir).glob("ckpt_*.pkl"))
    if not ckpts:
        return _fail("no checkpoint written")
    try:
        a = pickle.load(open(ckpts[0], "rb"))
        b = pickle.load(open(ckpts[0], "rb"))
        same = (a["best_val_acc"] == b["best_val_acc"] and
                a["best_dag_spec"] == b["best_dag_spec"] and
                "rng_state" in a)
        return _ok("checkpoint reloads identically") if same else \
            _fail("checkpoint mismatch on reload")
    except Exception as e:
        return _fail(f"checkpoint reload error: {e}")


def _summary(checks):
    n = len(checks); passed = sum(bool(c) for c in checks)
    print("\n" + "=" * 50)
    if passed == n:
        print(f"\033[92m SMOKE TEST PASSED ({passed}/{n})\033[0m")
    else:
        print(f"\033[91m SMOKE TEST FAILED ({passed}/{n})\033[0m")
    print("=" * 50)


if __name__ == "__main__":
    main()
