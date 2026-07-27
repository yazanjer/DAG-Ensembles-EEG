# Reproduction guide

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# CPU-only torch (skip on Colab GPU, where torch is preinstalled):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

or `conda env create -f environment.yml`.

## 2. Data

* **Dataset 1** — download `BCICIV_calib_ds1{a..g}.mat` from
  https://www.bbci.de/competition/iv/ and place them in `./dataset/`.
* **Dataset 2a** — download the 9 subject files `A01T.mat .. A09T.mat`
  (Kaggle/BBCI `data`-cell export) or the raw `A0?T.gdf`, and place them in a
  folder (e.g. `../BCI Competition IV Dataset 2a`). Pass that folder with
  `--dataset-dir`.

No data is redistributed with this repository.

## 3. Seeds

The eight seeds are `[42, 43, 44, 45, 46, 47, 48, 49]` (`config.yaml: seeds`).
Global default seed is 42. Every method is evaluated on all eight, so the
`baseline_comparison_multiseed.csv` is a paired, like-for-like comparison.

## 4. Validate, then run

```bash
python smoke_test.py                       # gate: must pass
python run_experiments.py --dataset ds1  --subjects a b f g --experiment ds1_multiseed
python run_experiments.py --dataset ds1  --eval-protocol cv --experiment ds1_cv
python run_experiments.py --dataset ds2a --dataset-dir "../BCI Competition IV Dataset 2a" \
       --variant binary --experiment ds2a_multiseed
python ablations.py
```

## 5. Outputs (under `results/<experiment>/`)

* `baseline_comparison_multiseed.csv` — mean ± std ± 95% CI per method (R2-3/2-4)
* `significance_summary.csv` — McNemar p-values + discordant-pair counts (R1-1)
* `winloss_summary.csv` — win/tie/loss of DAG–SA vs each baseline (R2-2)
* `exp2b_baseline_comparison.pdf/png` — real Exp 2-B results figure (R1-9)
* `confusion_*.pdf`, `convergence_*.pdf`, `per_seed_results.csv`
* `config_used.yaml`, `leakage_audit.txt`, `search_space.txt`
* checkpoints under `checkpoints/` (resume with the driver after a disconnect)

## 6. Config overrides

Edit `config.yaml` (band-pass order, CSSP delay, CSP regularisation, SVM grid,
CV folds/repeats, SA/random-search budgets, ensemble size, member constraint,
CI method). The exact config is copied to `config_used.yaml` for every run.

## 7. Resuming after a Colab disconnect

Checkpoints (`checkpoints/ckpt_<subject>_seed<seed>.pkl`) store the best DAG,
its serialized spec, best validation accuracy, iteration index and RNG state.
The optimizer's `run(resume=True)` reloads the latest per (subject, seed) and
continues.
