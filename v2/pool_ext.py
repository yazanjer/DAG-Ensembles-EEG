"""
pool_ext.py — v2 pool enrichment (improvement items 4 and 6).

Why this exists
---------------
In the published version the searchable pool contains only CSP / CSSP / channel
log-variance views. The revision's own results show that a Riemannian
tangent-space classifier beats DAG-SA on Dataset 1 and that EEGNet beats it on
Dataset 2a -- but neither is *in* the pool, so the search cannot reach that
level of accuracy by construction. This module adds two view families the
search can now select:

  * ``RIEM`` -- per-band covariance -> tangent-space projection. A same-family
    committee of RIEM members is, in the limit, the B5 baseline; the search can
    therefore match it or combine it with other views.
  * ``FB``   -- filter-bank CSP: CSP log-variance features concatenated across
    all frequency bands, optionally reduced by mutual-information feature
    selection (the classic FBCSP/MIBIF recipe Reviewer 2 mentioned).

Item 6 (shrinkage covariance for CSP) needs no code: ``preprocessing.csp_reg``
already reaches ``mne.decoding.CSP``; the v2 config sets it to ``oas``.

Everything here is fitted on the training split only, exactly like the original
views, so the leakage audit of the paper still holds.
"""
from __future__ import annotations

import numpy as np

from dag_core import (AlgorithmType, BaseClassifierNode, FeatureType,
                      FrequencyBand)

# --------------------------------------------------------------------------- #
# Optional dependencies
# --------------------------------------------------------------------------- #
def riemann_available() -> bool:
    try:
        import pyriemann  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# 1. Riemannian tangent-space views
# --------------------------------------------------------------------------- #
def add_riemannian_views(Xtr, Xval, Xte, bands_raw, y_tr, n_comp_key,
                         estimator="oas"):
    """Add one tangent-space view per frequency band.

    `bands_raw` maps FrequencyBand -> (train, val, test) band-pass filtered
    epochs of shape (trials, channels, samples). Covariance estimation and the
    tangent-space reference point are fitted on the TRAIN epochs only and then
    applied unchanged to val and test.
    """
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    for band, (tr, va, te) in bands_raw.items():
        key = (band, FeatureType.RIEM, n_comp_key)
        cov = Covariances(estimator=estimator)
        c_tr = cov.fit_transform(tr.astype(np.float64))
        c_va = cov.transform(va.astype(np.float64))
        c_te = cov.transform(te.astype(np.float64))
        ts = TangentSpace()
        ts.fit(c_tr, y_tr)                      # TRAIN only
        Xtr[key] = ts.transform(c_tr)
        Xval[key] = ts.transform(c_va)
        Xte[key] = ts.transform(c_te)
    return Xtr, Xval, Xte


# --------------------------------------------------------------------------- #
# 2. Filter-bank CSP views
# --------------------------------------------------------------------------- #
def add_filterbank_views(Xtr, Xval, Xte, y_tr, component_options,
                         nominal_band=FrequencyBand.FULL_MU,
                         select_k=None, seed=42):
    """Concatenate the per-band CSP log-variance features into one FB view.

    One view per component count. If `select_k` is given, mutual-information
    feature selection (fitted on TRAIN) keeps the k most informative features,
    which is the standard FBCSP/MIBIF reduction.
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_classif

    for n_comp in component_options:
        parts_tr, parts_va, parts_te = [], [], []
        for band in FrequencyBand:
            key = (band, FeatureType.CSP, n_comp)
            if key not in Xtr:
                continue
            parts_tr.append(Xtr[key])
            parts_va.append(Xval[key])
            parts_te.append(Xte[key])
        if not parts_tr:
            continue
        f_tr = np.hstack(parts_tr)
        f_va = np.hstack(parts_va)
        f_te = np.hstack(parts_te)
        if select_k:
            k = int(min(select_k, f_tr.shape[1]))
            sel = SelectKBest(
                score_func=lambda X, y: mutual_info_classif(
                    X, y, random_state=seed), k=k)
            sel.fit(f_tr, y_tr)                 # TRAIN only
            f_tr, f_va, f_te = (sel.transform(f_tr), sel.transform(f_va),
                                sel.transform(f_te))
        key = (nominal_band, FeatureType.FB, n_comp)
        Xtr[key], Xval[key], Xte[key] = f_tr, f_va, f_te
    return Xtr, Xval, Xte



# --------------------------------------------------------------------------- #
# 2b. The strong baselines themselves, as selectable pool members
# --------------------------------------------------------------------------- #
# The first version of this file added *cousins* of the strong baselines
# (band-passed tangent-space views with SVM heads). That is not the same thing
# as making the baselines reachable: the paper's B5 runs on UNFILTERED epochs
# with a logistic-regression head, and B4 (EEGNet) was not representable at all.
# The nodes below are the baselines themselves, fitted once per unit exactly
# like any other pool member, so the search can at worst tie them and at best
# combine them with the CSP views.

def add_raw_view(Xtr, Xval, Xte, Xtr_raw, Xval_raw, Xte_raw, key):
    """Expose the unprocessed epochs as a 'view' so whole models can be members."""
    Xtr[key] = np.asarray(Xtr_raw, dtype=np.float32)
    Xval[key] = np.asarray(Xval_raw, dtype=np.float32)
    Xte[key] = np.asarray(Xte_raw, dtype=np.float32)
    return Xtr, Xval, Xte


class RiemannianExactNode(BaseClassifierNode):
    """Exactly the paper's B5: Covariances(OAS) -> TangentSpace -> LogisticRegression
    on unfiltered epochs. A same-family committee of these reproduces B5."""

    def __init__(self, band, n_comp_key, estimator="oas", C=1.0, seed=42):
        self.band = band
        self.feat_type = FeatureType.RAW
        self.n_comp = n_comp_key
        self.algo_type = AlgorithmType.LDA        # label only; see spec()
        self.params = {"model": "riemannian_exact", "estimator": estimator, "C": C}
        self.seed = seed
        self.id = f"{band.name}_RIEMEXACT_{estimator}_C{C}"
        self.model = self._build_pipeline()

    def _build_pipeline(self):
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("cov", Covariances(estimator=self.params["estimator"])),
            ("ts", TangentSpace()),
            ("lr", LogisticRegression(max_iter=1000, C=self.params["C"],
                                      random_state=self.seed)),
        ])

    def fit(self, X_dict, y):
        self.model.fit(np.asarray(self._data(X_dict), dtype=np.float64), y)

    def predict_proba(self, X_dict):
        return self.model.predict_proba(
            np.asarray(self._data(X_dict), dtype=np.float64))

    def predict(self, X_dict):
        return self.model.predict(
            np.asarray(self._data(X_dict), dtype=np.float64))


class EEGNetNode(BaseClassifierNode):
    """The paper's B4 (EEGNet) as a pool member.

    Trained once per unit like every other member. `predict_proba` is the
    softmax of the network logits -- EEGNetClassifier itself only exposes a
    hard `predict`, so the probability head is added here.
    """

    def __init__(self, band, n_comp_key, fs, n_classes=2, seed=42, **kw):
        self.band = band
        self.feat_type = FeatureType.RAW
        self.n_comp = n_comp_key
        self.algo_type = AlgorithmType.SVM        # label only; see spec()
        self.params = {"model": "eegnet", **kw}
        self.seed = seed
        self.fs = fs
        self.n_classes = n_classes
        self.id = (f"{band.name}_EEGNET_F1x{kw.get('F1', 8)}"
                   f"_e{kw.get('epochs', 100)}")
        self.model = None

    def fit(self, X_dict, y):
        import baselines as B
        kw = {k: v for k, v in self.params.items() if k != "model"}
        self.model = B.EEGNetClassifier(n_classes=self.n_classes, fs=self.fs,
                                        seed=self.seed, **kw)
        self.model.fit(self._data(X_dict), y)

    def predict_proba(self, X_dict):
        torch = self.model.torch
        X = np.asarray(self._data(X_dict), dtype=np.float32)
        Xn = (X - self.model.mu) / self.model.sd
        self.model.net.eval()
        with torch.no_grad():
            Xt = torch.tensor(Xn[:, None, :, :], device=self.model.device)
            return torch.softmax(self.model.net(Xt), 1).cpu().numpy()

    def predict(self, X_dict):
        return np.argmax(self.predict_proba(X_dict), axis=1)

# --------------------------------------------------------------------------- #
# 3. Pool members for the new views
# --------------------------------------------------------------------------- #
def extend_pool(pool, cfg, component_options, seed=42,
                nominal_band=FrequencyBand.FULL_MU):
    """Append RIEM and FB classifier nodes to an already-generated pool.

    The grids are deliberately small: the point is to make the strong
    baselines *reachable* by the search, not to enlarge the search space for
    its own sake (which the paper identifies as the core problem).
    """
    v2 = cfg.get("pool", {}) if hasattr(cfg, "get") else {}
    n_comp_key = component_options[0]

    # The same-family constraint draws the family from `pool.feature_types`
    # (SimulatedAnnealingOptimizer._pick_family), so the new views must be
    # registered there or the search can never select them. Omitting this was
    # a real bug: in the first v2 campaign no RIEM or FB member was ever
    # chosen, which made V4 and VALL silently equivalent to their base
    # variants plus unreachable pool entries.
    def _register(ft):
        if ft not in pool.feature_types:
            pool.feature_types = tuple(pool.feature_types) + (ft,)

    if v2.get("include_riemannian", False):
        _register(FeatureType.RIEM)
        grid = v2.get("riemannian_grid") or [
            {"kernel": "linear", "C": 0.1},
            {"kernel": "linear", "C": 1.0},
            {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        ]
        for band in FrequencyBand:
            for params in grid:
                pool.pool.append(BaseClassifierNode(
                    band, FeatureType.RIEM, n_comp_key, AlgorithmType.SVM,
                    dict(params), seed))

    if v2.get("include_riemannian_exact", False):
        _register(FeatureType.RAW)
        for C in (v2.get("riemannian_exact_C") or [0.1, 1.0, 10.0]):
            pool.pool.append(RiemannianExactNode(
                FrequencyBand.FULL_MU, n_comp_key,
                estimator=v2.get("riemannian_estimator", "oas"), C=C,
                seed=seed))

    if v2.get("include_eegnet", False):
        import baselines as B
        if not B.eegnet_available():
            print("    [v2] torch not installed - EEGNet members skipped.")
        else:
            fs = v2.get("fs")
            assert fs, "pool.fs must be set for EEGNet members"
            for kw in (v2.get("eegnet_grid") or [
                    {"F1": 8, "D": 2, "F2": 16, "epochs": 100},
                    {"F1": 4, "D": 2, "F2": 8, "epochs": 100}]):
                _register(FeatureType.RAW)
                pool.pool.append(EEGNetNode(
                    FrequencyBand.FULL_MU, n_comp_key, fs=fs,
                    n_classes=v2.get("n_classes", 2), seed=seed, **kw))

    if v2.get("include_fbcsp", False):
        _register(FeatureType.FB)
        grid = v2.get("fbcsp_grid") or [
            {"kernel": "linear", "C": 1.0},
            {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
        ]
        lda_grid = v2.get("fbcsp_lda_grid") or [{"solver": "lsqr",
                                                 "shrinkage": "auto"}]
        for n_comp in component_options:
            for params in grid:
                pool.pool.append(BaseClassifierNode(
                    nominal_band, FeatureType.FB, n_comp, AlgorithmType.SVM,
                    dict(params), seed))
            for params in lda_grid:
                pool.pool.append(BaseClassifierNode(
                    nominal_band, FeatureType.FB, n_comp, AlgorithmType.LDA,
                    dict(params), seed))
    return pool


# --------------------------------------------------------------------------- #
# 4. One entry point used by the driver
# --------------------------------------------------------------------------- #
def build_views_and_pool(proc, pool, cfg, component_options,
                         Xtr_raw, y_tr, Xval_raw, Xte_raw,
                         Xtr, Xval, Xte, seed=42):
    """Add the v2 views to the feature dictionaries and the matching members
    to the pool. A no-op when the v2 pool switches are off, so V0 is
    bit-identical to the published pipeline."""
    v2 = cfg.get("pool", {})
    if not any(v2.get(k) for k in ("include_riemannian", "include_fbcsp",
                                   "include_riemannian_exact",
                                   "include_eegnet")):
        return Xtr, Xval, Xte, pool

    if v2.get("include_riemannian"):
        if not riemann_available():
            print("    [v2] pyriemann not installed - RIEM views skipped.")
        else:
            bands_raw = {}
            for band in FrequencyBand:
                low, high = band.value
                bands_raw[band] = (proc._bandpass(Xtr_raw, low, high),
                                   proc._bandpass(Xval_raw, low, high),
                                   proc._bandpass(Xte_raw, low, high))
            Xtr, Xval, Xte = add_riemannian_views(
                Xtr, Xval, Xte, bands_raw, y_tr, component_options[0],
                estimator=v2.get("riemannian_estimator", "oas"))
            del bands_raw

    if v2.get("include_riemannian_exact") or v2.get("include_eegnet"):
        add_raw_view(Xtr, Xval, Xte, Xtr_raw, Xval_raw, Xte_raw,
                     (FrequencyBand.FULL_MU, FeatureType.RAW,
                      component_options[0]))

    if v2.get("include_fbcsp"):
        Xtr, Xval, Xte = add_filterbank_views(
            Xtr, Xval, Xte, y_tr, component_options,
            select_k=v2.get("fbcsp_select_k"), seed=seed)

    pool = extend_pool(pool, cfg, component_options, seed=seed)
    return Xtr, Xval, Xte, pool
