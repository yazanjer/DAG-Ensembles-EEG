"""
selection.py — v2 selection machinery (improvement items 1, 2, 3 and 5).

The diagnosis in the revised manuscript is that DAG-SA's bottleneck is not the
optimiser or the pool but the *selection signal*: validation accuracy computed
on ~30 trials is used to rank ~4.4e11 candidate topologies, and the stacking
operator is additionally scored in-sample on that same split. This module
replaces that objective.

Components
----------
1. ``precompute_oof``            out-of-fold probabilities per pool member.
2. ``OOFScorer``                 topology objective computed on those, with
                                 out-of-fold stacking, an optional complexity
                                 penalty (item 2) and an optional diversity
                                 reward (item 5).
3. ``one_se_topology``           one-standard-error rule (item 2).
4. ``TopKAverage``               average of the k best topologies (item 3).

Nothing here touches the test split: the objective is computed entirely inside
the training partition, which is *stricter* than the published version (that
one consumed the separate validation split).
"""
from __future__ import annotations

import copy
import gc

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import dag_core
from dag_core import FeatureType, OperatorType


# --------------------------------------------------------------------------- #
# 1. Out-of-fold probabilities
# --------------------------------------------------------------------------- #
def precompute_oof(Xtr_raw, y_tr, fs, cfg, feats, component_options, seed,
                   n_folds=5, verbose=False):
    """Out-of-fold class probabilities for every pool member.

    For each inner fold the *entire* feature pipeline is refitted on the fold's
    training part -- CSP/CSSP filters, tangent-space reference, feature
    selection and the classifiers alike -- and applied to the held-out part.
    Refitting the extractors (rather than reusing filters fitted on the whole
    training split) is what makes the resulting matrix genuinely out-of-fold.

    Returns
    -------
    probs : dict member_id -> ndarray (n_train, n_classes)
    y     : ndarray (n_train,)  -- aligned with `probs`
    """
    import pool_ext

    y_tr = np.asarray(y_tr)
    n = len(y_tr)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    probs, n_classes = {}, len(np.unique(y_tr))

    for fold, (i_in, i_out) in enumerate(skf.split(np.zeros(n), y_tr)):
        proc = dag_core.DataProcessor(fs, component_options, cfg=cfg,
                                      feature_types=feats)
        Xa, Xb, _ = proc.process_splits(Xtr_raw[i_in], y_tr[i_in],
                                        Xtr_raw[i_out], Xtr_raw[i_out])
        pool = dag_core.ClassifierPool(component_options, cfg=cfg,
                                       feature_types=feats, seed=seed)
        pool.generate_pool()
        Xa, Xb, _, pool = pool_ext.build_views_and_pool(
            proc, pool, cfg, component_options,
            Xtr_raw[i_in], y_tr[i_in], Xtr_raw[i_out], Xtr_raw[i_out],
            Xa, Xb, dict(Xb), seed=seed)
        pool.pre_train_all(Xa, y_tr[i_in])

        for node in pool.pool:
            try:
                p = node.predict_proba(Xb)
            except Exception:
                continue
            key = _member_key(node)
            if key not in probs:
                probs[key] = np.full((n, n_classes), np.nan)
            probs[key][i_out] = p
        if verbose:
            print(f"      inner fold {fold + 1}/{n_folds} done")
        del proc, pool, Xa, Xb
        gc.collect()

    # members that failed on some fold are unusable as an objective
    probs = {k: v for k, v in probs.items() if not np.isnan(v).any()}
    return probs, y_tr


def _member_key(node):
    """Stable identity of a pool member (id already encodes the view)."""
    return f"{node.id}|{sorted(node.params.items())}"


# --------------------------------------------------------------------------- #
# 2. The out-of-fold objective
# --------------------------------------------------------------------------- #
class OOFScorer:
    """Score a candidate topology on the out-of-fold probability matrix.

    Parameters
    ----------
    complexity_penalty : float
        Subtracts ``lambda * n_leaves / members_ref`` from the score (item 2).
    diversity_weight : float
        Adds ``w * mean pairwise disagreement`` between the committee's
        members, measured on the out-of-fold predictions (item 5).
    stacking_cv : int
        Stacking nodes are fitted *and scored* by cross-validation inside the
        out-of-fold matrix, so a stacking node is never evaluated on data used
        to fit its meta-learner (item 1).
    """

    def __init__(self, probs, y, complexity_penalty=0.0, diversity_weight=0.0,
                 members_ref=4, stacking_cv=5, seed=42):
        self.probs = probs
        self.y = np.asarray(y)
        self.lam = float(complexity_penalty)
        self.w_div = float(diversity_weight)
        self.members_ref = max(1, int(members_ref))
        self.stacking_cv = int(stacking_cv)
        self.seed = seed
        self.n_calls = 0
        self._cache = {}

    # -- public ------------------------------------------------------------ #
    def __call__(self, dag):
        self.n_calls += 1
        key = str(dag.to_spec())
        if key in self._cache:
            return self._cache[key]
        leaves = dag.leaves()
        mats = [self.probs.get(_member_key(l)) for l in leaves]
        if any(m is None for m in mats):
            score = 0.0                       # member unusable -> reject
        else:
            fused = self._fuse(dag.root)
            acc = float(np.mean(np.argmax(fused, axis=1) == self.y))
            score = acc
            if self.lam:
                score -= self.lam * len(leaves) / self.members_ref
            if self.w_div:
                score += self.w_div * self._disagreement(mats)
        self._cache[key] = score
        return score

    def accuracy_of(self, dag):
        """Objective without the penalty/diversity terms (for reporting)."""
        mats = [self.probs.get(_member_key(l)) for l in dag.leaves()]
        if any(m is None for m in mats):
            return 0.0
        fused = self._fuse(dag.root)
        return float(np.mean(np.argmax(fused, axis=1) == self.y))

    def se(self, acc):
        """Standard error of an accuracy estimated on the OOF matrix."""
        n = len(self.y)
        return float(np.sqrt(max(acc * (1.0 - acc), 1e-12) / n))

    def fit_meta_learners(self, dag):
        """Fit the stacking meta-learners of `dag` on the out-of-fold matrix.

        This is the replacement for the published behaviour, where the ST
        meta-learner was fitted on the same validation split that scored the
        topology.
        """
        from sklearn.linear_model import LogisticRegression

        def rec(node):
            if isinstance(node, dag_core.EnsembleOperatorNode):
                parts = [rec(p) for p in node.parents]
                if node.op_type == OperatorType.ST:
                    Z = np.hstack(parts)
                    clf = LogisticRegression(random_state=self.seed,
                                             max_iter=1000)
                    clf.fit(Z, self.y)
                    node.meta_clf = clf
                    return clf.predict_proba(Z)
                return self._apply(node.op_type, parts)
            return self.probs[_member_key(node)]

        rec(dag.root)
        return dag

    # -- internals --------------------------------------------------------- #
    def _fuse(self, node):
        if isinstance(node, dag_core.EnsembleOperatorNode):
            parts = [self._fuse(p) for p in node.parents]
            if node.op_type == OperatorType.ST:
                return self._stack_cv(parts)
            return self._apply(node.op_type, parts)
        return self.probs[_member_key(node)]

    def _stack_cv(self, parts):
        """Predictions of the stacking meta-learner.

        With ``stacking_cv <= 1`` the meta-learner is fitted and scored on the
        same rows -- the published (in-sample, optimistically biased)
        behaviour, kept so that a variant can change *one* thing at a time.
        Otherwise the predictions are cross-validated, which is the fix.
        """
        from sklearn.linear_model import LogisticRegression
        Z = np.hstack(parts)
        clf = LogisticRegression(random_state=self.seed, max_iter=1000)
        if self.stacking_cv <= 1:
            clf.fit(Z, self.y)
            return clf.predict_proba(Z)
        try:
            return cross_val_predict(
                clf, Z, self.y, cv=StratifiedKFold(
                    n_splits=self.stacking_cv, shuffle=True,
                    random_state=self.seed), method="predict_proba")
        except Exception:
            clf.fit(Z, self.y)
            return clf.predict_proba(Z)

    @staticmethod
    def _apply(op, parts):
        stack = np.array(parts)                       # (m, n, c)
        if op == OperatorType.SV:
            s = stack.sum(axis=0)
            return s / np.clip(s.sum(axis=1, keepdims=True), 1e-12, None)
        if op == OperatorType.HV:
            return stack.max(axis=0)
        if op == OperatorType.MIN:
            m = stack.min(axis=0)
            return m / np.clip(m.sum(axis=1, keepdims=True), 1e-12, None)
        if op == OperatorType.MV:
            m, n, c = stack.shape
            hard = np.argmax(stack, axis=2)
            out = np.zeros((n, c))
            for j in range(n):
                counts = np.bincount(hard[:, j], minlength=c)
                top = counts.argmax()
                if counts[top] > m / 2:
                    out[j, top] = 1.0
                else:                                  # documented tie-break
                    s = stack[:, j, :].sum(axis=0)
                    out[j] = s / max(s.sum(), 1e-12)
            return out
        return stack.mean(axis=0)

    @staticmethod
    def _disagreement(mats):
        """Mean pairwise disagreement between members (0 = identical)."""
        hard = [np.argmax(m, axis=1) for m in mats]
        pairs, tot = 0, 0.0
        for i in range(len(hard)):
            for j in range(i + 1, len(hard)):
                tot += float(np.mean(hard[i] != hard[j]))
                pairs += 1
        return tot / pairs if pairs else 0.0


# --------------------------------------------------------------------------- #
# 3. One-standard-error rule (item 2)
# --------------------------------------------------------------------------- #
def one_se_topology(history, scorer):
    """Simplest topology whose score is within one SE of the best.

    `history` is the optimiser's list of (score, n_leaves, dag).
    """
    if not history:
        return None
    best_score = max(h[0] for h in history)
    tol = scorer.se(best_score)
    ok = [h for h in history if h[0] >= best_score - tol]
    ok.sort(key=lambda h: (h[1], -h[0]))          # fewest leaves, then best
    return ok[0][2]


# --------------------------------------------------------------------------- #
# 4. Top-k topology averaging (item 3)
# --------------------------------------------------------------------------- #
class TopKAverage:
    """Average the fused probabilities of the k best topologies visited.

    Turning the search's noisy argmax into an average over its k best states
    is the cheapest available variance reduction: no new components, only a
    different object returned at the end of the search.
    """

    def __init__(self, dags, weights=None):
        self.dags = list(dags)
        if weights is None:
            weights = np.ones(len(self.dags))
        w = np.asarray(weights, dtype=float)
        self.weights = w / w.sum()

    def predict_proba(self, X_dict):
        acc = None
        for w, d in zip(self.weights, self.dags):
            p = d.root.predict_proba(X_dict)
            acc = w * p if acc is None else acc + w * p
        return acc

    def predict(self, X_dict):
        return np.argmax(self.predict_proba(X_dict), axis=1)

    def to_spec(self):
        return {"top_k": len(self.dags),
                "members": [d.to_spec() for d in self.dags]}


# --------------------------------------------------------------------------- #
# 5. Convenience: build the final model from a finished search
# --------------------------------------------------------------------------- #
def finalise(opt, scorer, sel_cfg):
    """Turn a completed search into the model that will be tested.

    Applies, in order: the one-standard-error rule, top-k averaging, and
    out-of-fold fitting of any stacking meta-learners. Returns
    (model, info-dict).
    """
    info = {}
    history = list(opt.history_top)
    best = opt.best_dag

    if sel_cfg.get("one_se_rule") and history:
        cand = one_se_topology(history, scorer)
        if cand is not None:
            info["one_se_leaves"] = len(cand.leaves())
            info["argmax_leaves"] = len(best.leaves())
            best = cand

    k = int(sel_cfg.get("top_k_average", 1) or 1)
    if k > 1 and len(history) > 1:
        chosen = [copy.deepcopy(d) for _, _, d in history[:k]]
        if all(str(d.to_spec()) != str(best.to_spec()) for d in chosen):
            chosen[-1] = copy.deepcopy(best)
        for d in chosen:
            scorer.fit_meta_learners(d)
        info["top_k"] = len(chosen)
        return TopKAverage(chosen, [h[0] for h in history[:len(chosen)]]), info

    scorer.fit_meta_learners(best)
    info["top_k"] = 1
    return best, info
