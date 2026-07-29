"""
pipeline.py -- the v3 proposed method and its building blocks.

METHOD: ARTS  (Aligned Riemannian Transfer Stacking)

Motivation, stated so the design is auditable against the diagnosis:

  The published DAG-SA ranks ~4e11 ensemble topologies by accuracy on ~30
  held-out validation trials. That objective has a standard error of about
  9 accuracy points and a resolution of 1/30 = 3.3 points, so the argmax over
  a combinatorial space is dominated by selection noise. Re-running the same
  method with a different RNG trajectory moves a unit by 17 points (sd).

  ARTS removes the three things that produce that noise, one at a time:

    1. NO DISCRETE SEARCH. Ensemble construction becomes a convex weighting
       problem with a closed, regularised solution instead of an argmax over
       a combinatorial space. Given the split, ARTS is deterministic: the only
       stochasticity left is the inner-fold assignment.
    2. A SELECTION SIGNAL THAT IS THE WHOLE TRAINING SET. Fusion weights are
       fit on out-of-fold predictions across all n_train trials (140-230),
       not on a 30-trial holdout. Resolution improves from 1/30 to 1/n_train
       and there is no held-out split to overfit.
    3. A POOL OF GENUINELY DIFFERENT LEARNERS. The published pool is ~540
       near-duplicate CSP/CSSP log-variance views over 4 bands; a committee of
       four of them is close to a committee of one. ARTS uses one strong
       learner per frequency band (Riemannian tangent space), plus, for each
       band, a second learner trained on every OTHER subject's data after
       Euclidean alignment. Members differ by band and by data source, which
       is where ensemble gain actually comes from.

  Component 3's transfer half is also the only component with a documented
  effect size large enough to clear the ~4-point resolution bar of a
  subject-level comparison (He & Wu 2020, Euclidean alignment for transfer).

Everything below is fit on training data only. Per-trial band-pass filtering
and per-trial covariance estimation involve no cross-trial statistics, so
covariances may be precomputed for all trials before splitting without
leakage; every quantity that IS fitted (the alignment reference, the tangent
map, every classifier, the meta-learner) sees training indices only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# --------------------------------------------------------------------------- #
# Filter bank
# --------------------------------------------------------------------------- #
# Six overlapping bands spanning the mu/beta range. Fixed before any test data
# was examined; chosen to match the standard FBCSP 4 Hz grid plus a wideband
# 8-30 Hz member so that the single-band Riemannian baseline is nested inside
# the bank (which is what makes the ablation interpretable).
DEFAULT_BANDS: Tuple[Tuple[float, float], ...] = (
    (4.0, 8.0), (8.0, 12.0), (12.0, 16.0),
    (16.0, 22.0), (22.0, 30.0), (8.0, 30.0),
)
RIEMANNIAN_BASELINE_BAND = (8.0, 30.0)


def bandpass(X: np.ndarray, low: float, high: float, fs: int,
             order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth band-pass over the last axis of (trials, ch, time).

    filtfilt, not lfilter: the repository used a causal `lfilter`, which
    imposes a frequency-dependent phase delay that shifts the discriminative
    ERD/ERS window differently in each band. That is harmless when one band is
    used and harmful when bands are combined.
    """
    nyq = 0.5 * fs
    hi = min(high / nyq, 0.99)
    lo = max(low / nyq, 1e-4)
    b, a = butter(order, [lo, hi], btype="band")
    pad = min(3 * max(len(a), len(b)), X.shape[-1] - 1)
    return filtfilt(b, a, X, axis=-1, padlen=pad)


# --------------------------------------------------------------------------- #
# Covariance, alignment, tangent space
# --------------------------------------------------------------------------- #
def oas_cov(X: np.ndarray) -> np.ndarray:
    """
    Oracle-approximating-shrinkage covariance per trial.

    X: (trials, channels, samples) -> (trials, channels, channels).
    OAS is the right estimator here: n_samples per trial (350-1000) is not
    large relative to n_channels (22-59), and an unshrunk sample covariance
    is frequently near-singular, which the tangent-space log map cannot
    tolerate.
    """
    n_t, n_c, n_s = X.shape
    x = X - X.mean(axis=2, keepdims=True)
    s = np.einsum("nct,ndt->ncd", x, x) / n_s
    mu = np.trace(s, axis1=1, axis2=2) / n_c                    # (n_t,)
    alpha = (s * s).mean(axis=(1, 2))
    num = alpha + mu * mu
    den = (n_s + 1.0) * (alpha - (mu * mu) / n_c)
    rho = np.where(den <= 0, 1.0, np.minimum(1.0, num / np.where(den == 0, 1, den)))
    I = np.eye(n_c)
    return (1.0 - rho)[:, None, None] * s + (rho * mu)[:, None, None] * I


def _inv_sqrtm(C: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w, V = eigh(C)
    w = np.maximum(w, eps)
    return (V * (w ** -0.5)) @ V.T


def _logm_spd(C: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w, V = eigh(C)
    w = np.maximum(w, eps)
    return (V * np.log(w)) @ V.T


def euclidean_reference(covs: np.ndarray) -> np.ndarray:
    """
    Euclidean-alignment reference: the arithmetic mean covariance
    (He & Wu 2020). Must be estimated from TRAINING trials only.
    """
    return covs.mean(axis=0)


def align(covs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Whiten by the reference so the aligned mean is the identity."""
    W = _inv_sqrtm(ref)
    return W @ covs @ W          # batched matmul; einsum here is ~30x slower


_TRI_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def tangent(covs: np.ndarray) -> np.ndarray:
    """
    Tangent-space map at the identity: vec(logm(C)) over the upper triangle
    with off-diagonal entries scaled by sqrt(2) so the Euclidean norm of the
    vector equals the Frobenius norm of the matrix logarithm.

    Valid because `align` has already moved the mean to I. Using the identity
    reference (rather than fitting a Riemannian mean) keeps the map fully
    deterministic, which matters: it is one of the stochastic elements the
    redesign is trying to remove.
    """
    n, c, _ = covs.shape
    if c not in _TRI_CACHE:
        iu = np.triu_indices(c)
        w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
        _TRI_CACHE[c] = (iu[0], iu[1], w)
    r, cc, w = _TRI_CACHE[c]
    # Batched symmetric eigendecomposition -- an order of magnitude faster
    # than looping scipy.linalg.eigh, and numerically identical.
    wv, V = np.linalg.eigh(covs)
    wv = np.maximum(wv, 1e-12)
    # (V * log w) @ V^T as a batched matmul; einsum without optimize= falls
    # back to a naive loop and is ~30x slower here.
    L = (V * np.log(wv)[:, None, :]) @ np.swapaxes(V, 1, 2)
    return L[:, r, cc] * w


# --------------------------------------------------------------------------- #
# Per-subject, per-band covariance cache
# --------------------------------------------------------------------------- #
@dataclass
class SubjectBands:
    """
    Covariances for every trial of one subject, one entry per band.

    Leak-free to precompute before splitting: band-pass is a per-trial IIR
    with no fitted state and OAS covariance is a per-trial statistic. Nothing
    here is estimated across trials.
    """
    covs: Dict[Tuple[float, float], np.ndarray]
    y: np.ndarray
    subject: object

    @staticmethod
    def build(X: np.ndarray, y: np.ndarray, fs: int, subject,
              bands: Sequence[Tuple[float, float]] = DEFAULT_BANDS
              ) -> "SubjectBands":
        covs = {}
        for b in bands:
            covs[b] = oas_cov(bandpass(X, b[0], b[1], fs))
        return SubjectBands(covs=covs, y=np.asarray(y), subject=subject)


# --------------------------------------------------------------------------- #
# The method
# --------------------------------------------------------------------------- #
@dataclass
class ARTSConfig:
    bands: Sequence[Tuple[float, float]] = DEFAULT_BANDS
    use_alignment: bool = True         # Euclidean alignment
    use_transfer: bool = True          # cross-subject source views
    src_mode: str = "pooled"           # 'pooled' | 'per_subject'
    src_bands: Optional[Sequence[Tuple[float, float]]] = None
    use_csp: bool = False              # CSP + shrinkage-LDA views
    fusion: str = "stack"              # 'stack' | 'mean' | 'best'
    inner_folds: int = 5
    n_csp: int = 6
    C_self: float = 0.1                # L2 strength, subject-specific views
    C_src: float = 0.1                 # L2 strength, source views
    C_meta: float = 1.0                # L2 strength, meta-learner
    max_iter: int = 500


# --------------------------------------------------------------------------- #
# CSP computed from covariances (no raw signals needed)
# --------------------------------------------------------------------------- #
def csp_from_covs(covs: np.ndarray, y: np.ndarray, n_comp: int = 6
                  ) -> np.ndarray:
    """
    CSP spatial filters by generalised eigendecomposition of the class-mean
    covariances, taking the components whose generalised eigenvalue is
    farthest from 0.5 (i.e. most class-discriminative).

    Deriving CSP from the covariances the tangent-space views already use costs
    nothing extra and, more importantly, guarantees both view families see
    exactly the same signal -- so the ablation isolates the INDUCTIVE BIAS
    (rank-reduced log-variance vs full-rank tangent space) rather than any
    difference in preprocessing.

    For >2 classes, filters are stacked one-vs-rest.
    """
    cls = np.unique(y)
    if len(cls) > 2:
        return np.concatenate(
            [csp_from_covs(covs, (y == c).astype(int), n_comp) for c in cls],
            axis=0)
    A = covs[y == cls[0]].mean(axis=0)
    B = covs[y == cls[1]].mean(axis=0)
    A = A / np.trace(A)
    B = B / np.trace(B)
    w, V = eigh(A, A + B)
    order = np.argsort(np.abs(w - 0.5))[::-1]
    V = V[:, order]
    k = max(1, n_comp // 2)
    return np.concatenate([V[:, :k], V[:, -k:]], axis=1).T


def csp_logvar_from_covs(W: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """log-normalised variance of each CSP component, read off the covariance."""
    v = np.einsum("kc,ncd,kd->nk", W, covs, W)
    v = np.maximum(v, 1e-12)
    return np.log(v / v.sum(axis=1, keepdims=True))


def _fit_lr(Z, y, C, seed, max_iter):
    lr = LogisticRegression(C=C, max_iter=max_iter, random_state=seed,
                            class_weight="balanced")
    lr.fit(Z, y)
    return lr


def _logodds(P: np.ndarray) -> np.ndarray:
    """
    Meta-features are log-odds, not probabilities.

    A probability saturated at 0.99 and one at 0.999 are nearly identical to a
    linear meta-learner but differ by a factor of 10 in evidence. Log-odds
    keeps the meta-learner linear in the quantity the base learners are
    actually additive in.
    """
    P = np.clip(P, 1e-6, 1 - 1e-6)
    if P.shape[1] == 2:
        return np.log(P[:, 1:2] / P[:, 0:1])
    return np.log(P) - np.log(P).mean(axis=1, keepdims=True)


class ARTS:
    """
    Aligned Riemannian Transfer Stacking.

    fit(target_bands, train_idx, source_bands) -> self
    predict_proba(test_idx) -> (n_test, n_classes)

    Views, per band b:
        self_b   : LR on the target subject's own aligned tangent vectors
        src_b    : LR on every other subject's aligned tangent vectors
                   (only if use_transfer)

    Fusion weights are fit by multinomial L2 logistic regression on the
    out-of-fold log-odds of every view over the full training set.
    """

    def __init__(self, cfg: ARTSConfig = ARTSConfig(), seed: int = 0,
                 feature_cache: Optional[dict] = None):
        self.cfg = cfg
        self.seed = seed
        # Shared across the ARTS configurations evaluated on ONE unit (the
        # proposed method plus every ablation row). They differ in fusion,
        # transfer and band subset but re-derive the SAME alignment references
        # and tangent maps, which is where nearly all the time goes.
        # Keyed by (band, use_alignment, exact fitting index set), so a cache
        # hit is only possible when the reference is provably identical.
        self._fc = feature_cache if feature_cache is not None else {}
        self.classes_: Optional[np.ndarray] = None
        self.view_names_: List[str] = []
        self._src_models: Dict[Tuple[float, float], object] = {}
        self._self_models: Dict[Tuple[float, float], object] = {}
        self._ref: Dict[Tuple[float, float], np.ndarray] = {}
        self._meta = None
        self._uniform = False

    # -- feature construction ------------------------------------------ #
    def _ref_for(self, tb: SubjectBands, band, idx_fit):
        """Alignment reference, estimated from idx_fit only."""
        if not self.cfg.use_alignment:
            return None
        return euclidean_reference(tb.covs[band][idx_fit])

    def _aligned(self, tb: SubjectBands, band, ref, idx):
        C = tb.covs[band][idx]
        return C if ref is None else align(C, ref)

    def _cached(self, tb: SubjectBands, band, idx_fit):
        """
        (aligned covariances, tangent vectors) for ALL trials, under a
        reference estimated from idx_fit only.

        Computing all trials rather than just the ones needed costs a little
        extra per call and saves a great deal across the ablation ladder. It
        does not weaken the leakage guarantee: the only thing estimated from
        data is `ref`, and `ref` sees idx_fit alone.
        """
        key = (band, self.cfg.use_alignment, idx_fit.tobytes())
        hit = self._fc.get(key)
        if hit is None:
            ref = self._ref_for(tb, band, idx_fit)
            C = tb.covs[band] if ref is None else align(tb.covs[band], ref)
            hit = (ref, C, tangent(C))
            self._fc[key] = hit
        return hit

    def _target_feats(self, tb: SubjectBands, band, idx_fit, idx_apply):
        ref = self._ref_for(tb, band, idx_fit)
        return tangent(self._aligned(tb, band, ref, idx_apply)), ref

    def _target_feats_with_ref(self, tb, band, ref, idx_apply):
        return tangent(self._aligned(tb, band, ref, idx_apply))

    # -- source (transfer) views --------------------------------------- #
    def fit_sources_per_subject(self, source_bands: Sequence[SubjectBands],
                                bands: Optional[Sequence] = None):
        """
        One classifier per (source subject, band) instead of one per band over
        pooled sources.

        Rationale: pooling eight aligned subjects into a single logistic
        regression assumes their discriminative patterns superpose. They do
        not -- spatial patterns vary substantially between subjects even after
        Euclidean alignment. Keeping them separate lets the fusion layer
        discover WHICH source subjects resemble the target, which is the
        quantity that actually transfers.
        """
        bands = bands or self.cfg.bands
        self._src_per = {}
        for sb in source_bands:
            for band in bands:
                C = sb.covs[band]
                if self.cfg.use_alignment:
                    C = align(C, euclidean_reference(C))
                self._src_per[(sb.subject, band)] = _fit_lr(
                    tangent(C), sb.y, self.cfg.C_src, self.seed,
                    self.cfg.max_iter)
        return self

    def fit_sources(self, source_bands: Sequence[SubjectBands]):
        """
        Train one classifier per band on every other subject's trials.

        Each source subject is aligned by its OWN reference, computed from all
        of that subject's trials. This is not leakage: source subjects are
        disjoint from the target subject, and the target's test trials are
        never involved. It is also why these models are independent of the
        target's split and can be cached across seeds.
        """
        self._src_models = {}
        if not self.cfg.use_transfer or not source_bands:
            return self
        for band in self.cfg.bands:
            Zs, ys = [], []
            for sb in source_bands:
                C = sb.covs[band]
                if self.cfg.use_alignment:
                    C = align(C, euclidean_reference(C))
                Zs.append(tangent(C))
                ys.append(sb.y)
            Z = np.concatenate(Zs, 0)
            yy = np.concatenate(ys, 0)
            self._src_models[band] = _fit_lr(Z, yy, self.cfg.C_src,
                                             self.seed, self.cfg.max_iter)
        return self

    # -- fit ------------------------------------------------------------ #
    def fit(self, tb: SubjectBands, train_idx: np.ndarray,
            source_bands: Sequence[SubjectBands] = ()):
        cfg = self.cfg
        y_tr = tb.y[train_idx]
        self.classes_ = np.unique(y_tr)
        n_cls = len(self.classes_)

        if not self._src_models:
            self.fit_sources(source_bands)

        # ---- view inventory ------------------------------------------ #
        # Order matters: column blocks below are addressed by family offset.
        self.view_names_ = [f"self_{b[0]:g}-{b[1]:g}" for b in cfg.bands]
        self._has_src = bool(cfg.use_transfer and self._src_models)
        self._has_srcp = bool(cfg.use_transfer and cfg.src_mode == "per_subject"
                              and getattr(self, "_src_per", None))
        n_b = len(cfg.bands)
        self._off_src = None
        if self._has_src:
            self._off_src = len(self.view_names_)
            self.view_names_ += [f"src_{b[0]:g}-{b[1]:g}" for b in cfg.bands]
        self._srcp_keys = []
        if self._has_srcp:
            sbands = cfg.src_bands or cfg.bands
            self._off_srcp = len(self.view_names_)
            for (sub, band) in sorted(self._src_per,
                                      key=lambda k: (str(k[0]), k[1])):
                if band in tuple(sbands):
                    self._srcp_keys.append((sub, band))
                    self.view_names_.append(
                        f"srcS{sub}_{band[0]:g}-{band[1]:g}")
        self._off_csp = None
        if cfg.use_csp:
            self._off_csp = len(self.view_names_)
            self.view_names_ += [f"csp_{b[0]:g}-{b[1]:g}" for b in cfg.bands]

        # ---- out-of-fold meta features over the FULL training set ----- #
        width = 1 if n_cls == 2 else n_cls
        n_tr = len(train_idx)
        Zoof = np.zeros((n_tr, len(self.view_names_) * width))

        skf = StratifiedKFold(n_splits=min(cfg.inner_folds, np.bincount(
            y_tr - y_tr.min()).min()), shuffle=True, random_state=self.seed)
        n_b = len(cfg.bands)
        for tr_in, tr_out in skf.split(np.zeros(n_tr), y_tr):
            g_in = train_idx[tr_in]
            g_out = train_idx[tr_out]
            for j, band in enumerate(cfg.bands):
                # One alignment per (fold, band), reused by every view family.
                # The reference is estimated on g_in only, so nothing in the
                # meta-feature for a held-out trial depends on that trial.
                ref, Call, Zall = self._cached(tb, band, g_in)
                Ci, Co = Call[g_in], Call[g_out]
                Zi, Zo = Zall[g_in], Zall[g_out]

                m = _fit_lr(Zi, tb.y[g_in], cfg.C_self, self.seed, cfg.max_iter)
                c0 = j * width
                Zoof[np.ix_(tr_out, range(c0, c0 + width))] = \
                    _logodds(m.predict_proba(Zo))

                if self._has_src:
                    # Source models never see target labels at all, so their
                    # predictions on training trials are already out-of-fold.
                    c1 = (self._off_src + j) * width
                    Zoof[np.ix_(tr_out, range(c1, c1 + width))] = \
                        _logodds(self._src_models[band].predict_proba(Zo))

                if self._has_srcp:
                    for v, (sub, bd) in enumerate(self._srcp_keys):
                        if bd != band:
                            continue
                        c3 = (self._off_srcp + v) * width
                        Zoof[np.ix_(tr_out, range(c3, c3 + width))] = \
                            _logodds(self._src_per[(sub, bd)].predict_proba(Zo))

                if cfg.use_csp:
                    W = csp_from_covs(Ci, tb.y[g_in], cfg.n_csp)
                    lda = LDA(solver="lsqr", shrinkage="auto").fit(
                        csp_logvar_from_covs(W, Ci), tb.y[g_in])
                    c2 = (self._off_csp + j) * width
                    Zoof[np.ix_(tr_out, range(c2, c2 + width))] = _logodds(
                        lda.predict_proba(csp_logvar_from_covs(W, Co)))

        # ---- refit every view on the full training set ---------------- #
        self._self_models, self._ref, self._csp = {}, {}, {}
        self._full = {}
        for band in cfg.bands:
            ref, Call, Zall = self._cached(tb, band, train_idx)
            Ctr = Call[train_idx]
            self._ref[band] = ref
            self._full[band] = (Call, Zall)
            self._self_models[band] = _fit_lr(Zall[train_idx], y_tr,
                                              cfg.C_self, self.seed,
                                              cfg.max_iter)
            if cfg.use_csp:
                W = csp_from_covs(Ctr, y_tr, cfg.n_csp)
                lda = LDA(solver="lsqr", shrinkage="auto").fit(
                    csp_logvar_from_covs(W, Ctr), y_tr)
                self._csp[band] = (W, lda)

        # ---- fusion --------------------------------------------------- #
        if cfg.fusion == "stack":
            self._meta = _fit_lr(Zoof, y_tr, cfg.C_meta,
                                 self.seed, cfg.max_iter)
            self._uniform = False
        elif cfg.fusion == "mean":
            self._uniform = True
        elif cfg.fusion == "best":
            # Pick the single best view by OOF accuracy -- the "single-best
            # member" control, but selected on n_train rather than 30 trials.
            accs = []
            for v in range(len(self.view_names_)):
                sl = slice(v * width, (v + 1) * width)
                pred = self._view_argmax(Zoof[:, sl])
                accs.append((pred == y_tr).mean())
            self._best_view = int(np.argmax(accs))
            self._uniform = True
        else:
            raise ValueError(cfg.fusion)
        self._width = width
        return self

    def _view_argmax(self, Z):
        if Z.shape[1] == 1:
            return np.where(Z[:, 0] > 0, self.classes_[1], self.classes_[0])
        return self.classes_[np.argmax(Z, axis=1)]

    # -- predict --------------------------------------------------------- #
    def _meta_features(self, tb: SubjectBands, idx: np.ndarray):
        cfg = self.cfg
        width = self._width
        Z = np.zeros((len(idx), len(self.view_names_) * width))
        for j, band in enumerate(cfg.bands):
            Call, Zall = self._full[band]
            C, Zt = Call[idx], Zall[idx]
            c0 = j * width
            Z[:, c0:c0 + width] = _logodds(
                self._self_models[band].predict_proba(Zt))
            if self._has_src:
                c1 = (self._off_src + j) * width
                Z[:, c1:c1 + width] = _logodds(
                    self._src_models[band].predict_proba(Zt))
            if self._has_srcp:
                for v, (sub, bd) in enumerate(self._srcp_keys):
                    if bd != band:
                        continue
                    c3 = (self._off_srcp + v) * width
                    Z[:, c3:c3 + width] = _logodds(
                        self._src_per[(sub, bd)].predict_proba(Zt))
            if cfg.use_csp:
                W, lda = self._csp[band]
                c2 = (self._off_csp + j) * width
                Z[:, c2:c2 + width] = _logodds(
                    lda.predict_proba(csp_logvar_from_covs(W, C)))
        return Z

    def predict_proba(self, tb: SubjectBands, idx: np.ndarray) -> np.ndarray:
        Z = self._meta_features(tb, idx)
        if self.cfg.fusion == "stack":
            return self._meta.predict_proba(Z)
        w = self._width
        n_v = len(self.view_names_)
        if self.cfg.fusion == "best":
            sl = slice(self._best_view * w, (self._best_view + 1) * w)
            L = Z[:, sl]
        else:
            L = np.mean([Z[:, v * w:(v + 1) * w] for v in range(n_v)], axis=0)
        if w == 1:
            p1 = 1.0 / (1.0 + np.exp(-L[:, 0]))
            return np.column_stack([1 - p1, p1])
        e = np.exp(L - L.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, tb: SubjectBands, idx: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(tb, idx), axis=1)]
