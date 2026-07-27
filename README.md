# DAG–SA: Annealed Search over DAG Ensembles for Motor-Imagery EEG Decoding

Code for the manuscript *"Annealed Search over Directed-Acyclic-Graph Ensembles
for Subject-Specific Motor-Imagery EEG Decoding"* (MLWA-D-26-01081). DAG–SA uses
simulated annealing to select a small ensemble of CSP/CSSP/CTP + SVM/LDA base
classifiers and a fusion operator, evaluated on BCI Competition IV Datasets 1
and 2a.

This revision reorganises the original scripts into reusable modules, adds the
multi-seed / cross-validation / confidence-interval / baseline / ablation
machinery requested by the reviewers, and runs both locally and in Google Colab
with Google Drive.

## Layout

| File | Purpose |
|------|---------|
| `env_utils.py` | Colab detection, Drive mount, `PROJECT_ROOT`, seeding, config loader |
| `config.yaml` | Single source of truth for every hyper-parameter (printed + saved per run) |
| `datasets_io.py` | `load_dataset('ds1'/'ds2a', ...)` -> `(X, y, fs, ch_names, class_names)` |
| `dag_core.py` | Features (CSP/CSSP/CTP), classifier pool, DAG, simulated annealing + checkpoints |
| `baselines.py` | EEGNet (CNN), Riemannian (non-CNN), single-best, full-pool soft-vote, random search |
| `metrics_utils.py` | Class-wise metrics, kappa, bootstrap/normal CIs, McNemar, win/tie/loss |
| `run_experiments.py` | Unified multi-seed driver (`split` or nested `cv`), figures, audits |
| `ablations.py` | Member constraint, ensemble size, reheating, SA budget, selection frequency |
| `smoke_test.py` | Fast end-to-end validation on tiny settings (Section C) |
| `DAG_SA_Colab.ipynb` | Colab orchestration: install -> mount -> smoke -> full runs -> ablations |
| `implementation_with_test*.py` | Original scripts, kept for reference |

## Quick start (local)

```bash
pip install -r requirements.txt          # torch optional; see requirements.txt
# Put dataset files under ./dataset/  (BCICIV_calib_ds1?.mat, and/or A0?T.mat)
python smoke_test.py                      # must print SMOKE TEST PASSED
python run_experiments.py --dataset ds1 --experiment ds1_multiseed
```

## Full runs

```bash
# Dataset 1, all eight seeds, single-split protocol
python run_experiments.py --dataset ds1 --subjects a b f g --experiment ds1_multiseed

# Cross-validation protocol (nested repeated stratified k-fold)
python run_experiments.py --dataset ds1 --eval-protocol cv --experiment ds1_cv

# Dataset 2a, binary left/right subset, 9 subjects
python run_experiments.py --dataset ds2a --dataset-dir "../BCI Competition IV Dataset 2a" \
    --variant binary --experiment ds2a_multiseed

# Ablations
python ablations.py
```

Every method (DAG–SA, single-best, full-pool soft-vote, random search, EEGNet,
Riemannian) is evaluated on the **same eight seeds** and reported as
**mean ± std ± 95% CI**. Results go to `results/<experiment>/` (or Drive on Colab),
never the working directory.

## Checkpointing & resume (survives Colab disconnects)

The optimizer checkpoints on best-so-far improvement, every
`sa.checkpoint_every` iterations, **and at least every `sa.checkpoint_minutes`
of wall-clock time** (default 10). Checkpoints are written atomically to
`checkpoints/` with `{subject, seed, iter, best_dag_spec, best_val_acc,
rng_state, elapsed_seconds}`.

The driver also checkpoints at the run level: after each `(subject, seed)` unit
finishes, its rows are appended atomically to `per_seed_results.csv` /
`significance_summary.csv`. `run_experiment(..., resume=True)` (the default)
reads those CSVs on restart and **skips already-completed `(subject, seed)`
pairs**, so a disconnect never loses finished work. Pass `--no-resume` to force a
full recompute.

## Runtime estimates

At the start of every run the driver prints an ETA; after the first completed
unit it prints a calibrated total; and it refreshes
`results/<experiment>/runtime_report.txt` (per-unit timings + measured/projected
totals) after every unit. A "unit" is one `(subject, seed)` pair. Fill this
table from `runtime_report.txt` on your target hardware — the numbers are
machine-dependent (Colab CPU vs GPU), so they must come from a calibration run
rather than being hard-coded:

| Experiment | Units (subjects × seeds) | SA budget | Notes |
|------------|--------------------------|-----------|-------|
| Smoke test | 1 × 2 | 5 iters (tiny) | seconds; must print `SMOKE TEST PASSED` |
| Dataset 1 full (`split`) | 4 × 8 = 32 | 300 iters | EEGNet needs GPU for a fast run |
| Dataset 1 (`cv`) | 4 × 8 units × (n_splits·n_repeats) folds | 300 iters | ~25× the per-unit cost of `split` |
| Dataset 2a full (`split`) | 9 × 8 = 72 | 300 iters | binary left/right subset |
| Ablation suite | 1 subject × 2 seeds × ~13 variants | tiny→full | scale via `ablations.run_ablations` args |

> Estimation method (Section C): `total ≈ units × mean_unit_time`, where
> `mean_unit_time` is measured from the first completed unit at the real SA
> budget and refined as a rolling average. GPU vs CPU is auto-detected and
> stated in the printed assumptions.

## Reproducibility

Global seed defaults to 42; the eight seeds `[42..49]` live in `config.yaml`.
See `reproduce.md` for exact steps and `CHANGELOG_revision.md` for the mapping
from reviewer comments to changes.

## Data availability

BCI Competition IV Dataset 1 and Dataset 2a are public
(https://www.bbci.de/competition/iv/). Place the files under `dataset/` (Dataset
1) and a Dataset-2a folder (Dataset 2a); nothing is redistributed here.

## License

MIT — see `LICENSE`.
