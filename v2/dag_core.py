"""
dag_core.py
===========
Seed-parametrized refactor of the DAG-SA pipeline
(`implementation_with_test.py`), packaged for reuse by the multi-seed driver,
the baselines, the ablation runner and the Colab notebook.

What changed vs. the original monolith (scientific behaviour preserved):
  * Everything is a function/class taking an explicit `seed` — no module-level
    global seeding side-effects, so the 8-seed loops (R2-3) are honest.
  * DataProcessor supports CSP / CTP / CSSP feature views and reads all
    hyper-parameters from `config` (R2-6). CSP and Platt calibration are fit on
    TRAIN ONLY; topology is selected on VAL; TEST is untouched (R2-5 leakage).
  * SimulatedAnnealingOptimizer supports: configurable ensemble size M,
    reheating on/off, member-family constraint (same_family / unconstrained /
    partial, R2-8), periodic + best-so-far checkpointing with RNG state, and
    `resume=True` to continue after a Colab disconnect (Section A.3).
  * `count_search_space()` reports the size of the search space (R2-5a).
  * `EnsembleDAG.to_spec()` serialises the selected topology (R2-1).
"""

from __future__ import annotations

import copy
import gc
import math
import os
import pickle
import random
import itertools
import time
from collections import Counter
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter

# ---- sklearn compatibility patch (older calls pass force_writeable) --------
# v2 fix: the patch must be idempotent. Re-importing this module (e.g.
# `importlib.reload(dag_core)` in a notebook, or simply re-running the import
# cell in Colab) used to re-capture the ALREADY patched function as the
# "original", so the wrapper called itself: every classifier fit then died with
# RecursionError and the whole pool was discarded as unfittable.
import sklearn.utils.validation as _skv
if not getattr(_skv.check_X_y, "_dagsa_patched", False):
    _orig_check_X_y = _skv.check_X_y

    def _patched_check_X_y(X, y, **kwargs):
        kwargs.pop("force_writeable", None)
        return _orig_check_X_y(X, y, **kwargs)

    _patched_check_X_y._dagsa_patched = True
    _skv.check_X_y = _patched_check_X_y

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class FrequencyBand(Enum):
    ALPHA = (8, 13)
    BETA = (14, 30)
    FULL_MU = (8, 30)
    LOW_MU = (7, 15)


class FeatureType(Enum):
    CSP = "CSP"
    CTP = "CTP"
    CSSP = "CSSP"
    # ---- v2 additions (see pool_ext.py) ---------------------------------- #
    RIEM = "RIEM"   # covariance -> tangent-space projection (per band)
    FB = "FB"       # filter-bank CSP: CSP log-var concatenated across bands
    RAW = "RAW"     # unprocessed epochs: lets a whole baseline model be a member


class AlgorithmType(Enum):
    LDA = "LDA"
    SVM = "SVM"


class OperatorType(Enum):
    MV = "Majority Voting"
    HV = "Hard Voting"
    SV = "Soft Voting"
    MIN = "Min Probability"
    ST = "Stacking"


_OP_ABBR = {OperatorType.MV: "MV", OperatorType.HV: "HV", OperatorType.SV: "SV",
            OperatorType.MIN: "MIN", OperatorType.ST: "ST"}
_FAMILY = {FeatureType.CSP: "CSP", FeatureType.CSSP: "CSP", FeatureType.CTP: "CTP",
           # v2: the tangent-space views form their own family, so a
           # same-family committee can be entirely Riemannian -- i.e. the
           # search can now reach the strongest baseline of the paper.
           FeatureType.RIEM: "RIEM", FeatureType.FB: "FB",
           # members that ARE the strong baselines (exact B5, EEGNet) share one
           # family, so a same-family committee can be a committee of them.
           FeatureType.RAW: "STRONG"}


# --------------------------------------------------------------------------- #
# Signal processing
# --------------------------------------------------------------------------- #
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return lfilter(b, a, data, axis=2)


def cssp_transform(X, delay=2):
    """Common Spatio-Spatial Pattern: concatenate signal with a delayed copy."""
    if delay <= 0:
        return X
    shifted = np.roll(X, shift=delay, axis=2)
    shifted[:, :, :delay] = 0.0
    return np.concatenate([X, shifted], axis=1)


# --------------------------------------------------------------------------- #
# Data processing
# --------------------------------------------------------------------------- #
class DataProcessor:
    """Builds per-(band, feature, n_comp) feature views. CSP/Platt fit on TRAIN."""

    def __init__(self, fs, component_options, cfg=None,
                 feature_types=(FeatureType.CSP, FeatureType.CSSP, FeatureType.CTP)):
        self.fs = fs
        self.component_options = component_options
        self.cfg = cfg
        self.feature_types = tuple(feature_types)
        pp = (cfg["preprocessing"] if cfg is not None else {})
        self.order = pp.get("bandpass_order", 5)
        self.cssp_delay = pp.get("cssp_delay", 2)
        self.csp_reg = pp.get("csp_reg", None)
        self.csp_log = pp.get("csp_log", True)
        self.transformers = {}

    def _bandpass(self, X, low, high):
        # v2: single precision halves the peak memory of the per-band copies.
        # The filtered signal feeds CSP/covariance estimation only, where
        # float32 is numerically ample.
        out = butter_bandpass_filter(X, low, high, self.fs, order=self.order)
        return np.asarray(out, dtype=np.float32)

    def process_splits(self, X_tr_raw, y_tr, X_val_raw, X_test_raw):
        """Fit feature transformers on train; apply to val/test (no leakage)."""
        Xtr, Xval, Xte = {}, {}, {}
        for band in FrequencyBand:
            low, high = band.value
            tr = self._bandpass(X_tr_raw, low, high)
            va = self._bandpass(X_val_raw, low, high)
            te = self._bandpass(X_test_raw, low, high)
            for feat in self.feature_types:
                for n_comp in self.component_options:
                    key = (band, feat, n_comp)
                    if feat == FeatureType.CTP:
                        Xtr[key] = np.log(np.var(tr, axis=2) + 1e-10)
                        Xval[key] = np.log(np.var(va, axis=2) + 1e-10)
                        Xte[key] = np.log(np.var(te, axis=2) + 1e-10)
                        self.transformers[key] = "CTP"
                    else:
                        tr_in, va_in, te_in = tr, va, te
                        if feat == FeatureType.CSSP:
                            tr_in = cssp_transform(tr, self.cssp_delay)
                            va_in = cssp_transform(va, self.cssp_delay)
                            te_in = cssp_transform(te, self.cssp_delay)
                        csp = CSP(n_components=n_comp, reg=self.csp_reg,
                                  log=self.csp_log, norm_trace=False)
                        Xtr[key] = csp.fit_transform(tr_in, y_tr)   # TRAIN only
                        Xval[key] = csp.transform(va_in)
                        Xte[key] = csp.transform(te_in)
                        self.transformers[key] = csp
                        if feat == FeatureType.CSSP:
                            del tr_in, va_in, te_in     # v2: free the copies
            del tr, va, te
            gc.collect()
        return Xtr, Xval, Xte


def three_way_split(X, y, seed, test_size=0.15, val_size=0.15):
    """Stratified Train / Val / Test split (val/test are % of TOTAL)."""
    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    val_adj = val_size / (1.0 - test_size)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_adj, stratify=y_tmp, random_state=seed)
    return X_tr, y_tr, X_val, y_val, X_te, y_te


# --------------------------------------------------------------------------- #
# Model nodes
# --------------------------------------------------------------------------- #
class BaseClassifierNode:
    def __init__(self, band, feat_type, n_comp, algo_type, params, seed=42):
        self.band = band
        self.feat_type = feat_type
        self.n_comp = n_comp
        self.algo_type = algo_type
        self.params = params
        self.seed = seed
        self.id = f"{band.name}_{feat_type.name}_{n_comp}c_{algo_type.name}"
        self.model = self._build_pipeline()

    def _build_pipeline(self):
        if self.algo_type == AlgorithmType.LDA:
            clf = LinearDiscriminantAnalysis(**self.params)
        else:  # SVM — Platt calibration (probability=True) is fit on TRAIN only
            clf = SVC(probability=True, random_state=self.seed, **self.params)
        return Pipeline([("classifier", clf)])

    def _data(self, X_dict):
        return X_dict[(self.band, self.feat_type, self.n_comp)]

    def fit(self, X_dict, y):
        self.model.fit(self._data(X_dict), y)

    def predict_proba(self, X_dict):
        return self.model.predict_proba(self._data(X_dict))

    def predict(self, X_dict):
        return self.model.predict(self._data(X_dict))

    def __deepcopy__(self, memo):
        """Pool members are shared, never copied.

        The search deep-copies a candidate DAG on every perturbation, which
        used to copy the fitted leaf models too. That is wasteful (four fitted
        SVMs cloned per iteration) and, once a leaf wraps a torch model, fatal:
        ``EEGNetClassifier`` holds a reference to the ``torch`` module and
        ``copy.deepcopy`` raises ``TypeError: cannot pickle 'module' object``.

        Members are immutable after ``fit`` -- the search only ever swaps
        references to them -- so returning ``self`` is both correct and
        faster. Operator nodes are still copied properly, which is what
        matters, since their ``meta_clf`` is mutated during the search.
        """
        memo[id(self)] = self
        return self

    def spec(self):
        return {"band": self.band.name, "feature": self.feat_type.name,
                "n_comp": self.n_comp, "algo": self.algo_type.name,
                "params": self.params}


class ClassifierPool:
    def __init__(self, component_options, cfg=None,
                 feature_types=(FeatureType.CSP, FeatureType.CSSP, FeatureType.CTP), seed=42):
        self.pool = []
        self.component_options = component_options
        self.cfg = cfg
        self.feature_types = tuple(feature_types)
        self.seed = seed

    def generate_pool(self):
        self.pool = []
        svm_params = (self.cfg["svm_grid"] if self.cfg else [
            {"kernel": "linear", "C": 1.0}, {"kernel": "rbf", "C": 10.0, "gamma": "scale"}])
        lda_params = (self.cfg["lda_grid"] if self.cfg else [{"solver": "svd"}])
        for band, feat, n_comp in itertools.product(
                FrequencyBand, self.feature_types, self.component_options):
            for p in lda_params:
                self.pool.append(BaseClassifierNode(band, feat, n_comp,
                                                    AlgorithmType.LDA, dict(p), self.seed))
            for p in svm_params:
                self.pool.append(BaseClassifierNode(band, feat, n_comp,
                                                    AlgorithmType.SVM, dict(p), self.seed))

    def pre_train_all(self, X_train_dict, y_train):
        """Fit every member; drop the ones that fail.

        v2 fix: the published version counted failures but left the unfitted
        members in the pool, so the search could select one and then die with
        ``NotFittedError`` when the topology was scored. Dropping them keeps
        the pool self-consistent -- an unfittable member is not a candidate.
        """
        ok, kept, failed = 0, [], []
        for node in self.pool:
            try:
                node.fit(X_train_dict, y_train)
                kept.append(node)
                ok += 1
            except Exception as e:
                failed.append((node.id, type(e).__name__))
        if failed:
            print(f"    [pool] dropped {len(failed)} member(s) that failed to "
                  f"fit, e.g. {failed[0][0]} ({failed[0][1]})")
        self.pool = kept
        return ok

    def _family_of(self, node):
        return _FAMILY[node.feat_type]

    def get_random_distinct(self, k=2, exclude=(), family=None):
        cands = [c for c in self.pool if c not in exclude]
        if family is not None:
            cands = [c for c in cands if self._family_of(c) == family]
        if len(cands) < k:
            cands = [c for c in self.pool if c not in exclude]  # relax
        if len(cands) < k:
            raise ValueError("Pool too small for requested sample")
        return random.sample(cands, k)


class EnsembleOperatorNode:
    def __init__(self, op_type, parents, seed=42):
        self.op_type = op_type
        self.parents = parents
        self.seed = seed
        self.meta_clf = None

    def fit(self, X_dict, y):
        if self.op_type == OperatorType.ST:
            probs = [p.predict_proba(X_dict) for p in self.parents]
            self.meta_clf = LogisticRegression(random_state=self.seed, max_iter=1000)
            self.meta_clf.fit(np.hstack(probs), y)

    def predict_proba(self, X_dict):
        probs_list = [p.predict_proba(X_dict) for p in self.parents]
        stack = np.array(probs_list)
        N_clf, N_samp, N_cls = stack.shape

        if self.op_type == OperatorType.MV:
            out = np.zeros((N_samp, N_cls))
            hard = np.argmax(stack, axis=2)
            for i in range(N_samp):
                most, count = Counter(hard[:, i]).most_common(1)[0]
                if count > N_clf / 2:
                    out[i, most] = 1.0
                else:  # documented tie-break: fall back to summed probabilities
                    s = np.sum(stack[:, i, :], axis=0)
                    out[i, :] = s / (s.sum() + 1e-10)
            return out
        if self.op_type == OperatorType.HV:
            return np.max(stack, axis=0)
        if self.op_type == OperatorType.SV:
            s = np.sum(stack, axis=0)
            o = np.zeros_like(s)
            return np.divide(s, s.sum(axis=1, keepdims=True), out=o,
                             where=s.sum(axis=1, keepdims=True) != 0)
        if self.op_type == OperatorType.MIN:
            m = np.min(stack, axis=0)
            o = np.zeros_like(m)
            return np.divide(m, m.sum(axis=1, keepdims=True), out=o,
                             where=m.sum(axis=1, keepdims=True) != 0)
        if self.op_type == OperatorType.ST:
            return self.meta_clf.predict_proba(np.hstack(probs_list))
        return np.mean(stack, axis=0)

    def predict(self, X_dict):
        return np.argmax(self.predict_proba(X_dict), axis=1)

    def spec(self):
        return {"operator": _OP_ABBR[self.op_type],
                "parents": [p.spec() for p in self.parents]}


class EnsembleDAG:
    def __init__(self, root):
        self.root = root

    def fit_meta_learners(self, X_dict, y):
        def _rec(node):
            if isinstance(node, EnsembleOperatorNode):
                for p in node.parents:
                    _rec(p)
                if node.op_type == OperatorType.ST:
                    node.fit(X_dict, y)
        _rec(self.root)

    def accuracy(self, X_dict, y):
        return accuracy_score(y, self.root.predict(X_dict))

    def leaves(self):
        found = []
        def _rec(node):
            if isinstance(node, BaseClassifierNode):
                found.append(node)
            else:
                for p in node.parents:
                    _rec(p)
        _rec(self.root)
        return found

    def to_spec(self):
        """Serialise topology (members, edges, fusion operators) — R2-1."""
        return {"root": self.root.spec(), "n_members": len(self.leaves())}


# --------------------------------------------------------------------------- #
# Search-space accounting (R2-5a)
# --------------------------------------------------------------------------- #
def count_search_space(pool_size, members=4, n_operators=5):
    """
    Rough size of the DAG search space: choose `members` base classifiers from
    the pool (ordered into two pairs) × operator choices at the 3 fusion nodes.
    Reported as an order-of-magnitude figure for the manuscript.
    """
    from math import comb
    member_subsets = comb(pool_size, members) if pool_size >= members else 0
    operator_configs = n_operators ** 3  # two pair-operators + one root
    return {
        "pool_size": pool_size,
        "member_subsets": member_subsets,
        "operator_configs": operator_configs,
        "total_estimate": member_subsets * operator_configs,
    }


# --------------------------------------------------------------------------- #
# Simulated annealing
# --------------------------------------------------------------------------- #
class SimulatedAnnealingOptimizer:
    def __init__(self, pool, X_val, y_val, dataset_name, processor, seed=42,
                 members=4, member_constraint="same_family", use_reheat=True,
                 checkpoint_dir=None, checkpoint_minutes=10,
                 scorer=None, keep_history=0, init_family=None):
        """
        v2 additions
        ------------
        scorer : callable(EnsembleDAG) -> float, optional
            Replaces validation accuracy as the search objective. Used to plug
            in the out-of-fold objective of `selection.py`. When None the
            behaviour is identical to the published version.
        keep_history : int
            If > 0, retain the `keep_history` best distinct topologies visited
            (for one-standard-error selection and top-k averaging).
        """
        self.scorer = scorer
        # v2: seed the initial committee from one family (e.g. "STRONG", the
        # exact B5 / EEGNet members). Without this the strong members are 3-5
        # entries in a 560-member pool and are essentially never sampled, so
        # "the search can reach them" stays theoretical.
        self.init_family = init_family
        self.keep_history = int(keep_history)
        self.history_top = []          # list of (score, n_leaves, EnsembleDAG)
        self.pool = pool
        self.X_val = X_val
        self.y_val = y_val
        self.dataset_name = dataset_name
        self.processor = processor
        self.seed = seed
        self.members = members
        self.member_constraint = member_constraint
        self.use_reheat = use_reheat
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        # Wall-clock checkpoint interval in minutes (Section B). None/<=0 disables.
        self.checkpoint_minutes = checkpoint_minutes
        self.elapsed_seconds = 0.0     # accumulated compute time (survives resume)
        self.family = None
        self.start_iter = 0
        self.current_dag = self._create_dag()
        self.best_dag = copy.deepcopy(self.current_dag)
        self.best_acc = 0.0
        self.history = {"accuracy": [], "temperature": [], "best_accuracy": []}

    # -- topology helpers ---------------------------------------------------- #
    def _pick_family(self):
        # v2: with an init family AND the same-family constraint, the committee
        # is locked to that family -- e.g. a committee made only of the strong
        # baselines. Without the lock, unconstrained mutations drift away from
        # the warm start within a few dozen iterations.
        if self.init_family and self.member_constraint == "same_family":
            return self.init_family
        if self.member_constraint == "same_family":
            fams = list({_FAMILY[f] for f in self.pool.feature_types})
            return random.choice(fams)
        return None  # unconstrained / partial handled at sample time

    def _sample(self, k, exclude=()):
        fam = self.family if self.member_constraint == "same_family" else None
        return self.pool.get_random_distinct(k=k, exclude=list(exclude), family=fam)

    def _create_dag(self):
        self.family = self._pick_family()
        if self.init_family and self.start_iter == 0:
            bases = self.pool.get_random_distinct(k=self.members,
                                                  family=self.init_family)
        else:
            bases = self._sample(self.members)
        # Build `n_pairs` pair-operators then a root over them.
        pair_ops = []
        for i in range(0, len(bases) - 1, 2):
            pair_ops.append(EnsembleOperatorNode(
                random.choice(list(OperatorType)), [bases[i], bases[i + 1]], self.seed))
        if len(bases) % 2 == 1:  # odd member added to first pair
            pair_ops[0].parents.append(bases[-1])
        root = EnsembleOperatorNode(random.choice(list(OperatorType)), pair_ops, self.seed)
        dag = EnsembleDAG(root)
        if self.scorer is None:
            dag.fit_meta_learners(self.X_val, self.y_val)
        return dag

    def _score(self, dag):
        """Objective value of a candidate topology.

        Default: validation accuracy, exactly as published. With a `scorer`
        injected (v2) the candidate is scored out-of-fold on the training
        split instead, and the visited topology is recorded so that the
        one-standard-error rule and top-k averaging can use it.
        """
        if self.scorer is None:
            dag.fit_meta_learners(self.X_val, self.y_val)
            score = dag.accuracy(self.X_val, self.y_val)
        else:
            score = float(self.scorer(dag))
        if self.keep_history:
            self._remember(dag, score)
        return score

    def _remember(self, dag, score):
        key = str(dag.to_spec())
        for i, (s, n, d) in enumerate(self.history_top):
            if str(d.to_spec()) == key:
                if score > s:
                    self.history_top[i] = (score, n, d)
                break
        else:
            self.history_top.append((score, len(dag.leaves()),
                                     copy.deepcopy(dag)))
        self.history_top.sort(key=lambda r: -r[0])
        del self.history_top[self.keep_history:]

    def _perturb(self, dag):
        new = copy.deepcopy(dag)
        opts = ["swap_base", "change_op", "change_root", "add_base", "delete_base"]
        mut = random.choice(opts)
        target = new.root.parents[random.randint(0, len(new.root.parents) - 1)]
        if not isinstance(target, EnsembleOperatorNode):
            target = new.root
        if mut == "swap_base":
            idx = random.randint(0, len(target.parents) - 1)
            if isinstance(target.parents[idx], BaseClassifierNode):
                excl = target.parents[:idx] + target.parents[idx + 1:]
                target.parents[idx] = self._sample(1, exclude=excl)[0]
        elif mut == "change_op":
            target.op_type = random.choice(list(OperatorType))
        elif mut == "change_root":
            new.root.op_type = random.choice(list(OperatorType))
        elif mut == "add_base":
            try:
                target.parents.append(self._sample(1, exclude=target.parents)[0])
            except Exception:
                pass
        elif mut == "delete_base":
            base_children = [p for p in target.parents if isinstance(p, BaseClassifierNode)]
            if len(base_children) > 2:
                target.parents.remove(random.choice(base_children))
        return new

    # -- checkpointing ------------------------------------------------------- #
    def _ckpt_path(self):
        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir / f"ckpt_{self.dataset_name}_seed{self.seed}.pkl"

    def save_checkpoint(self, it):
        path = self._ckpt_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "subject": self.dataset_name, "seed": self.seed, "iter": it,
            "best_dag": self.best_dag, "best_dag_spec": self.best_dag.to_spec(),
            "best_val_acc": self.best_acc, "history": self.history,
            "family": self.family,
            "elapsed_seconds": self.elapsed_seconds,
            "rng_state": (random.getstate(), np.random.get_state()),
        }
        # Atomic write: dump to a temp file, then os.replace so a Colab
        # disconnect mid-write can never corrupt the resumable checkpoint.
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(bundle, f)
        os.replace(tmp, path)

    def try_resume(self):
        path = self._ckpt_path()
        if path is None or not path.exists():
            return False
        with open(path, "rb") as f:
            b = pickle.load(f)
        self.best_dag = b["best_dag"]
        self.best_acc = b["best_val_acc"]
        self.history = b["history"]
        self.family = b.get("family")
        self.elapsed_seconds = b.get("elapsed_seconds", 0.0)
        self.start_iter = b["iter"] + 1
        self.current_dag = copy.deepcopy(self.best_dag)
        rs, ns = b["rng_state"]
        random.setstate(rs)
        np.random.set_state(ns)
        print(f"    [resume] {self.dataset_name} seed{self.seed} @ iter {self.start_iter}")
        return True

    # -- main loop ----------------------------------------------------------- #
    def run(self, iterations=100, temp=5.0, cooling_rate=0.95, nreheat=20,
            checkpoint_every=25, checkpoint_minutes=None, resume=False,
            verbose=True):
        if checkpoint_minutes is None:
            checkpoint_minutes = self.checkpoint_minutes
        if resume:
            self.try_resume()
        curr_acc = self._score(self.current_dag)
        if self.best_acc == 0.0:
            self.best_acc = curr_acc
            self.best_dag = copy.deepcopy(self.current_dag)
        stagnant = 0
        # Wall-clock checkpoint bookkeeping (Section B.1): save at least every
        # `checkpoint_minutes`, independent of the iteration-count trigger.
        ckpt_interval = (checkpoint_minutes * 60.0
                         if checkpoint_minutes and checkpoint_minutes > 0 else None)
        loop_start = time.monotonic()
        last_ckpt_time = loop_start
        if verbose:
            print(f"    Init val acc: {curr_acc:.2%}")
        for it in range(self.start_iter, iterations):
            new = self._perturb(self.current_dag)
            new_acc = self._score(new)
            delta = curr_acc - new_acc
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-9)):
                self.current_dag = new
                curr_acc = new_acc
                if curr_acc > self.best_acc:
                    self.best_acc = curr_acc
                    self.best_dag = copy.deepcopy(new)
                    self.save_checkpoint(it)
                    if verbose:
                        print(f"    iter {it}: best {self.best_acc:.2%}")
                    stagnant = 0
            self.history["accuracy"].append(curr_acc)
            self.history["temperature"].append(temp)
            self.history["best_accuracy"].append(self.best_acc)
            temp *= cooling_rate
            stagnant += 1
            if self.use_reheat and stagnant > nreheat:
                temp *= 1.5
                stagnant = 0
            now = time.monotonic()
            self.elapsed_seconds += now - loop_start
            loop_start = now
            wall_due = ckpt_interval is not None and (now - last_ckpt_time) >= ckpt_interval
            if (checkpoint_every and it % checkpoint_every == 0) or wall_due:
                self.save_checkpoint(it)
                if wall_due:
                    last_ckpt_time = now
                    if verbose:
                        print(f"    [ckpt] wall-clock save @ iter {it} "
                              f"({self.elapsed_seconds/60:.1f} min elapsed)")
            if temp < 1e-4:
                break
        self.save_checkpoint(iterations)
        return self.best_dag
