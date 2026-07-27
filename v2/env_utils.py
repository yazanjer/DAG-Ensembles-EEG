"""
env_utils.py
============
Shared environment / path / reproducibility helpers for the DAG-SA EEG project.

Addresses Section A of the revision prompt:
 - Colab detection + Google Drive mount
 - single configurable PROJECT_ROOT (Colab default /content/drive/MyDrive/EEG_DAGSA/)
 - derived DATASET_DIR / RESULTS_DIR / CHECKPOINT_DIR (created on demand)
 - self-documenting: prints resolved paths
 - consistent seeding of python `random`, numpy and (if present) torch
 - a single loader for config.yaml so every run prints the hyper-parameters used

Works both as a plain ``.py`` import and inside Google Colab.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# YAML is a light dependency; degrade gracefully if it is missing.
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:  # pragma: no cover
    _HAS_YAML = False


# --------------------------------------------------------------------------- #
# Colab detection
# --------------------------------------------------------------------------- #
def in_colab() -> bool:
    """Return True when running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def mount_drive(mountpoint: str = "/content/drive") -> bool:
    """Mount Google Drive when in Colab. Returns True if a mount happened."""
    if not in_colab():
        return False
    try:
        from google.colab import drive  # type: ignore
        drive.mount(mountpoint)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"[env] WARNING: could not mount Drive: {exc}")
        return False


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
@dataclass
class ProjectPaths:
    """Resolved, created project directories for one run."""
    root: Path
    dataset_dir: Path
    results_dir: Path
    checkpoint_dir: Path

    def create(self) -> "ProjectPaths":
        for d in (self.results_dir, self.checkpoint_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def describe(self) -> None:
        print("[env] Resolved project paths:")
        print(f"      PROJECT_ROOT   = {self.root}")
        print(f"      DATASET_DIR    = {self.dataset_dir}")
        print(f"      RESULTS_DIR    = {self.results_dir}")
        print(f"      CHECKPOINT_DIR = {self.checkpoint_dir}")


def resolve_paths(project_root: Optional[str | os.PathLike] = None) -> ProjectPaths:
    """
    Resolve PROJECT_ROOT and derived directories.

    Priority:
      1. explicit ``project_root`` argument (e.g. top-of-notebook variable)
      2. environment variable EEG_DAGSA_ROOT
      3. Colab default /content/drive/MyDrive/EEG_DAGSA/
      4. local: the directory containing this file
    """
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
    elif os.environ.get("EEG_DAGSA_ROOT"):
        root = Path(os.environ["EEG_DAGSA_ROOT"]).expanduser().resolve()
    elif in_colab():
        root = Path("/content/drive/MyDrive/EEG_DAGSA")
    else:
        root = Path(__file__).resolve().parent

    return ProjectPaths(
        root=root,
        dataset_dir=root / "dataset",
        results_dir=root / "results",
        checkpoint_dir=root / "checkpoints",
    )


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def seed_everything(seed: int) -> None:
    """Seed python, numpy and (if importable) torch consistently."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional (only needed for the EEGNet baseline)
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic-ish; avoids surprising nondeterminism on GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
_DEFAULT_CONFIG = {
    "seed": 42,
    "seeds": [42, 43, 44, 45, 46, 47, 48, 49],
    "eval_protocol": "split",          # 'split' | 'cv'
    "cv": {"n_splits": 5, "n_repeats": 5},
    "window_sizes": [[0, 4], [1, 3]],
    "component_options": [[6, 7, 8], [4, 5, 6, 7, 8]],
    "preprocessing": {
        "bandpass_type": "butterworth",
        "bandpass_order": 5,
        "cssp_delay": 2,
        "csp_reg": None,               # None | float | 'ledoit_wolf' | 'oas'
        "csp_log": True,
        "downsample_to": None,
        "artefact_handling": "none",
        "trial_rejection": "none",
        "channel_selection": "all",
    },
    "sa": {
        "iterations": 300,
        "temp": 5.0,
        "cooling_rate": 0.97,
        "nreheat": 20,
        "checkpoint_every": 25,
        "checkpoint_minutes": 10,
    },
    "random_search": {"iterations": 300},  # matched to SA budget by default
    "svm_grid": [
        {"kernel": "linear", "C": 0.1},
        {"kernel": "linear", "C": 1.0},
        {"kernel": "linear", "C": 10.0},
        {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 100.0, "gamma": "auto"},
        {"kernel": "rbf", "C": 1.0, "gamma": 0.01},
        {"kernel": "rbf", "C": 1.0, "gamma": 0.1},
        {"kernel": "poly", "degree": 2, "C": 1.0},
        {"kernel": "poly", "degree": 3, "C": 1.0},
    ],
    "lda_grid": [
        {"solver": "svd"},
        {"solver": "lsqr", "shrinkage": "auto"},
        {"solver": "lsqr", "shrinkage": 0.1},
        {"solver": "lsqr", "shrinkage": 0.5},
        {"solver": "eigen", "shrinkage": "auto"},
    ],
    "hard_voting_tie_break": "sum_proba",  # documented tie rule for MV/HV
    "ensemble": {"members": 4, "operators": ["MV", "HV", "SV", "MIN", "ST"]},
    "member_constraint": "same_family",    # 'same_family' | 'unconstrained' | 'partial'
    "baselines": {
        "eegnet": {"enabled": True, "epochs": 100, "lr": 1e-3, "batch_size": 32},
        "riemannian": {"enabled": True, "estimator": "oas"},
        "single_best": {"enabled": True},
        "full_pool_soft_vote": {"enabled": True},
        "random_search": {"enabled": True},
    },
    "ci": {"method": "bootstrap", "n_boot": 2000, "alpha": 0.05},
    "subjects_ds1": ["a", "b", "f", "g"],
}


@dataclass
class Config:
    data: dict = field(default_factory=lambda: dict(_DEFAULT_CONFIG))

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def print_summary(self) -> None:
        print("[config] Hyper-parameters in use (R2-6 transparency):")
        for k in ("seed", "seeds", "eval_protocol", "cv", "preprocessing",
                  "sa", "random_search", "member_constraint", "ci"):
            print(f"      {k}: {self.data.get(k)}")

    def dump(self, path: str | os.PathLike) -> None:
        """Write the exact config used to config_used.yaml alongside results."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_YAML:
            with open(path, "w") as f:
                yaml.safe_dump(self.data, f, sort_keys=False)
        else:  # fall back to a readable repr
            with open(path, "w") as f:
                f.write(repr(self.data))


def load_config(path: Optional[str | os.PathLike] = None) -> Config:
    """
    Load config.yaml if present, else fall back to the built-in defaults.
    Unknown keys in the file override defaults (shallow merge at top level,
    deep merge one level down for dict values).
    """
    cfg = dict(_DEFAULT_CONFIG)
    if path is None:
        candidate = Path(__file__).resolve().parent / "config.yaml"
    else:
        candidate = Path(path)
    if candidate.exists() and _HAS_YAML:
        with open(candidate) as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                merged = dict(cfg[k]); merged.update(v); cfg[k] = merged
            else:
                cfg[k] = v
    elif candidate.exists() and not _HAS_YAML:
        print("[config] WARNING: PyYAML not installed; using built-in defaults.")
    return Config(cfg)


# --------------------------------------------------------------------------- #
# One-call setup
# --------------------------------------------------------------------------- #
def setup_environment(project_root: Optional[str | os.PathLike] = None,
                      config_path: Optional[str | os.PathLike] = None,
                      seed: Optional[int] = None):
    """
    Detect Colab, mount Drive, resolve+create dirs, load config, seed RNGs.
    Returns (paths, config).
    """
    if in_colab():
        print("[env] Google Colab detected.")
        mount_drive()
    else:
        print("[env] Local environment detected.")

    paths = resolve_paths(project_root).create()
    paths.describe()

    cfg = load_config(config_path)
    if seed is None:
        seed = cfg.get("seed", 42)
    seed_everything(seed)
    print(f"[env] Global seed frozen: {seed}")
    cfg.print_summary()
    return paths, cfg


if __name__ == "__main__":
    p, c = setup_environment()
    print("\n[env] Self-test OK.")
