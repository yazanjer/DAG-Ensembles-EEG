"""
baselines.py
============
All baselines, evaluated on the *identical* splits and seeds as DAG-SA so the
unified multi-seed table is fair (Reviewer 2 #3). Two families:

Feature-space baselines (share the DAG-SA CSP/CTP feature dicts):
  * B1 single_best            — best single base classifier (chosen on val)
  * B2 full_pool_soft_vote    — soft-vote over the whole pre-trained pool
  * B3 random_search          — random DAG search, budget matched to SA

Raw-epoch baselines on identical splits:
  * EEGNet (compact CNN)      — Reviewer 1 #4  (torch; CPU fallback)
  * Riemannian (non-CNN)      — Reviewer 1 #5  (pyriemann: cov→tangent→LDA)

Every baseline exposes the same call signature returning test-set predictions,
so the driver can score them with metrics_utils uniformly. Optional deps
(torch, pyriemann) are import-guarded: if missing, the baseline reports
`available=False` and is skipped rather than crashing the run.
"""

from __future__ import annotations

import numpy as np

from dag_core import SimulatedAnnealingOptimizer


# --------------------------------------------------------------------------- #
# Feature-space baselines
# --------------------------------------------------------------------------- #
def single_best(pool, X_val_dict, y_val, X_test_dict):
    """B1: pick the pool member with best validation accuracy; predict on test."""
    from sklearn.metrics import accuracy_score
    best_node, best_val = None, -1.0
    for node in pool.pool:
        try:
            acc = accuracy_score(y_val, node.predict(X_val_dict))
        except Exception:
            continue
        if acc > best_val:
            best_val, best_node = acc, node
    if best_node is None:
        raise RuntimeError("single_best: no usable pool member")
    return best_node.predict(X_test_dict), {"val_acc": best_val,
                                             "member": best_node.id}


def full_pool_soft_vote(pool, X_test_dict):
    """B2: average predict_proba across all trained pool members; argmax."""
    probs = []
    for node in pool.pool:
        try:
            probs.append(node.predict_proba(X_test_dict))
        except Exception:
            continue
    if not probs:
        raise RuntimeError("full_pool_soft_vote: no usable pool member")
    mean = np.mean(np.array(probs), axis=0)
    return np.argmax(mean, axis=1), {"n_members": len(probs)}


def random_search(pool, X_val_dict, y_val, X_test_dict, iterations,
                  seed, members=4, member_constraint="same_family"):
    """
    B3: sample `iterations` random DAGs (same budget as SA), keep the one with
    best validation accuracy, evaluate it on test. Uses the same DAG builder as
    the optimizer so the two are structurally comparable.
    """
    sampler = SimulatedAnnealingOptimizer(
        pool, X_val_dict, y_val, "randsearch", processor=None, seed=seed,
        members=members, member_constraint=member_constraint)
    best_dag, best_val = None, -1.0
    for _ in range(iterations):
        dag = sampler._create_dag()  # random, meta-learners fit on val
        acc = dag.accuracy(X_val_dict, y_val)
        if acc > best_val:
            best_val, best_dag = acc, dag
    return best_dag.root.predict(X_test_dict), {"val_acc": best_val}


# --------------------------------------------------------------------------- #
# EEGNet (compact CNN) — Reviewer 1 #4
# --------------------------------------------------------------------------- #
def eegnet_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


class EEGNetClassifier:
    """
    Compact EEGNet (Lawhern et al. 2018) on raw epochs (trials, channels,
    samples). Pure PyTorch so no braindecode dependency is required. Falls back
    to CPU automatically; uses GPU when available (GPU Colab, per your setup).
    """

    def __init__(self, n_classes=2, fs=250, epochs=100, lr=1e-3,
                 batch_size=32, F1=8, D=2, F2=16, dropout=0.5, seed=42,
                 verbose=False):
        import torch
        self.torch = torch
        self.n_classes = n_classes
        self.fs = fs
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.F1, self.D, self.F2 = F1, D, F2
        self.dropout = dropout
        self.seed = seed
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None

    def _build(self, C, T):
        torch = self.torch
        import torch.nn as nn
        kern = max(2, self.fs // 2)

        class Net(nn.Module):
            def __init__(s, F1, D, F2, C, T, kern, n_cls, p):
                super().__init__()
                s.block1 = nn.Sequential(
                    nn.Conv2d(1, F1, (1, kern), padding=(0, kern // 2), bias=False),
                    nn.BatchNorm2d(F1),
                    nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False),
                    nn.BatchNorm2d(F1 * D), nn.ELU(),
                    nn.AvgPool2d((1, 4)), nn.Dropout(p))
                s.block2 = nn.Sequential(
                    nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                              groups=F1 * D, bias=False),
                    nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                    nn.BatchNorm2d(F2), nn.ELU(),
                    nn.AvgPool2d((1, 8)), nn.Dropout(p))
                with torch.no_grad():
                    d = s.block2(s.block1(torch.zeros(1, 1, C, T)))
                s.classify = nn.Linear(int(np.prod(d.shape[1:])), n_cls)

            def forward(s, x):
                x = s.block2(s.block1(x))
                return s.classify(x.flatten(1))

        torch.manual_seed(self.seed)
        return Net(self.F1, self.D, self.F2, C, T, kern, self.n_classes,
                   self.dropout).to(self.device)

    def fit(self, X, y):
        torch = self.torch
        import torch.nn as nn
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        # per-channel standardisation (fit on train)
        self.mu = X.mean(axis=(0, 2), keepdims=True)
        self.sd = X.std(axis=(0, 2), keepdims=True) + 1e-7
        Xn = (X - self.mu) / self.sd
        C, T = Xn.shape[1], Xn.shape[2]
        self.net = self._build(C, T)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = nn.CrossEntropyLoss()
        Xt = torch.tensor(Xn[:, None, :, :], device=self.device)
        yt = torch.tensor(y, device=self.device)
        n = len(y)
        self.net.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                out = self.net(Xt[idx])
                loss = lossf(out, yt[idx])
                loss.backward()
                opt.step()
            if self.verbose and ep % 20 == 0:
                print(f"      EEGNet ep{ep} loss {loss.item():.3f}")
        return self

    def predict(self, X):
        torch = self.torch
        X = np.asarray(X, dtype=np.float32)
        Xn = (X - self.mu) / self.sd
        self.net.eval()
        with torch.no_grad():
            Xt = torch.tensor(Xn[:, None, :, :], device=self.device)
            out = self.net(Xt)
            return out.argmax(1).cpu().numpy()


# --------------------------------------------------------------------------- #
# Riemannian (non-CNN) — Reviewer 1 #5
# --------------------------------------------------------------------------- #
def riemannian_available():
    try:
        import pyriemann  # noqa: F401
        return True
    except Exception:
        return False


def build_riemannian(estimator="oas", seed=42):
    """
    Non-convolutional baseline: covariance → tangent space → LogisticRegression.
    Operates on raw epochs (trials, channels, samples).
    """
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("cov", Covariances(estimator=estimator)),
        ("ts", TangentSpace()),
        ("lr", LogisticRegression(max_iter=1000, random_state=seed)),
    ])
