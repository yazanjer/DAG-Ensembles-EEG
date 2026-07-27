# CHANGELOG — Major Revision (MLWA-D-26-01081)

Mapping of reviewer comments to concrete changes. Legend: **[CODE]** done in
code here; **[MANUSCRIPT]** handled in the LaTeX / response document (flagged,
not silently ignored); **[BOTH]** code produces the evidence, manuscript
reports it.

## Section A — Colab + Drive + reproducibility
- `env_utils.setup_environment()`: Colab detection, `drive.mount`, single
  `PROJECT_ROOT` (Colab default `/content/drive/MyDrive/EEG_DAGSA/`, overridable),
  derived `DATASET_DIR`/`RESULTS_DIR`/`CHECKPOINT_DIR` created on demand, paths
  printed. **[CODE]**
- Data read from `DATASET_DIR`; missing files fail fast with the exact expected
  path (`datasets_io`). **[CODE]**
- Checkpoints re-enabled and written to `CHECKPOINT_DIR`; periodic + best-so-far
  checkpoints store `{subject, seed, iter, best_dag_spec, best_val_acc,
  rng_state}`; `run(resume=True)` reloads the latest per (subject, seed). **[CODE]**
- Results re-enabled (`to_csv`, figures) under `RESULTS_DIR/<experiment>/`;
  nothing is written to the CWD. **[CODE]**
- `requirements.txt` pinned + `environment.yml`; notebook install cell guards
  `torch`/`mne` for Colab. **[CODE]**
- `SEED=42` kept as default; seed is a parameter and the eight seeds iterate in
  the driver; `seed_everything` seeds python/numpy/torch. **[CODE]**

## Editor / Associate Editor
- EIC #1/#2, AE #1/#2/#3: response-document structure, APA/single-column,
  colored-text revisions, institutional emails, verbatim EIC comments.
  **[MANUSCRIPT]** — addressed in the LaTeX and the Detailed Response file.

## Reviewer 1
- R1-1 significance: McNemar exact test per DAG–SA-vs-baseline comparison with
  discordant-pair counts + p-values in `significance_summary.csv`; effect
  reported alongside CIs. Printed output no longer overclaims. **[BOTH]**
- R1-2 CV: `--eval-protocol cv` runs nested repeated stratified k-fold
  (default 5×5, configurable); SA search runs inside the training portion of
  each outer fold; accuracy aggregated as mean ± std ± 95% CI. **[CODE]**
- R1-3 Dataset 2a: `datasets_io.load_dataset('ds2a', ...)` loads the 9-subject
  2a data (binary left/right subset or 4-class) behind the same interface as
  Dataset 1. **[CODE]** — 2a files supplied by the user.
- R1-4 DL baseline on identical splits: `EEGNet` (compact CNN, PyTorch) trained
  and evaluated on the same splits/seeds as DAG–SA; appears in the unified
  multi-seed table. **[CODE]**
- R1-5 non-convolutional baseline: `Riemannian` (pyriemann covariance ->
  tangent space -> logistic regression) on identical splits. **[CODE]**
- R1-6 public repo: clean `README.md`, pinned `requirements.txt` /
  `environment.yml`, documented `SEED`, `reproduce.md`, `LICENSE`; no absolute
  personal paths remain (replaced by `PROJECT_ROOT`). Actual GitHub/Zenodo
  upload is a user action. **[CODE + user action]**
- R1-7 highlights length: **[MANUSCRIPT]** (a length-check helper can be added
  if wanted).
- R1-8/1-10 figure sizing/overlap/duplication: **[MANUSCRIPT/CODE]**.
- R1-9 Figure 4 must show real results: `exp2b_baseline_comparison.pdf/png`
  plots actual per-method accuracy with 95% CI (not a flowchart). **[CODE]**
- R1-11 duplicate Algorithm 2 layer; R1-12 ethics statement: **[MANUSCRIPT]**.

## Reviewer 2
- R2-1 DAG novelty: `EnsembleDAG.to_spec()` serialises members/edges/fusion
  operators (saved in checkpoints); gives concrete, inspectable topology.
  **[BOTH]**
- R2-2 moderate claims: summaries report DAG–SA alongside all baselines;
  `winloss_summary.csv` tallies win/tie/loss and only counts a win when
  McNemar is significant. **[BOTH]**
- R2-3 consistent seeds (critical): the driver evaluates *every* method on the
  same eight seeds; one unified `baseline_comparison_multiseed.csv`. No
  single-seed-vs-averaged path remains. **[CODE]**
- R2-4 confidence intervals: bootstrap (default) and normal-approx CIs in
  `metrics_utils`; 95% CIs across seeds in the summary table. **[CODE]**
- R2-5 overfitting / search space / calibration: (a) `count_search_space()`
  logs the space size to `search_space.txt`; (b) nested `cv` protocol keeps
  topology selection and final test on disjoint folds; (c) `leakage_audit.txt`
  documents that CSP and Platt calibration are fit on TRAIN only. **[CODE]**
- R2-6 implementation details: all hyper-parameters live in `config.yaml`,
  printed at run start and copied to `config_used.yaml` (filter type/order,
  CSSP delay, CSP reg, SVM grid, calibration, MV/HV tie-break rule,
  random-search budget matched to SA). **[BOTH]**
- R2-7 fair Table 5: driver builds a same-dataset/same-splits comparison
  (CSP-LDA via single-best, Riemannian, EEGNet, soft-vote); cross-dataset
  context table stays in the manuscript, reframed. **[BOTH]**
- R2-8 same-family constraint: `member_constraint` switch
  (same_family / unconstrained / partial) with an ablation table. **[CODE]**
- R2-9 more ablations: `ablations.py` varies ensemble size M, reheating on/off,
  SA budget, and logs feature-family / fusion-operator selection frequency.
  **[CODE]**
- R2-10 class-wise metrics: per-class precision/recall/F1, balanced accuracy,
  Cohen's kappa in every results row. **[CODE]**
- R2-11 encoding artefacts; R2-12 ethics/data-availability: **[MANUSCRIPT]** —
  README states dataset source/access route to support the corrected statement.

## Hardening pass — 2026-07-24 (code review + Sections B–E)
See `CODE_REVIEW_LOG.md` for the full log and `REVIEWER_VERIFICATION.md` for the
comment-by-comment PASS/PARTIAL/FAIL audit. Summary:
- **Wall-clock checkpointing (B):** `sa.checkpoint_minutes: 10` — SA saves a
  resumable, atomically-written checkpoint at least every 10 min in addition to
  best-so-far / every-N-iter. `elapsed_seconds` added to the bundle. **[CODE]**
- **Driver-level partial progress (B.3):** each `(subject, seed)` unit is
  appended atomically to the results CSVs on completion; `resume=True` (default)
  skips already-completed pairs after a disconnect. **[CODE]**
- **Runtime estimation (C):** start ETA + first-unit calibration + live rolling
  ETA + `runtime_report.txt`; README runtime-estimate table added. **[CODE]**
- **R2-3 fairness bug fixed:** in the `cv` protocol, baselines previously ran on
  fold 0 only while DAG–SA ran on all folds; now every method runs on every
  fold, with per-fold McNemar. **[CODE]**
- **Packaging bug fixed:** `requirements.txt` was UTF-16 (unparseable by pip);
  re-encoded as UTF-8. **[CODE]**
- **Drive-only writes (D) verified:** all writes rooted at `RESULTS_DIR` /
  `CHECKPOINT_DIR`; none to CWD. **[CODE]**
- **Manuscript gaps flagged (not silently dropped):** Figure 4 still duplicates
  the flowchart, Table 3 still seed-42-vs-8-seed, EEGNet/Riemannian/Dataset-2a
  results and the ethics fix are not yet in `main.tex`. See
  `REVIEWER_VERIFICATION.md` §0 for copy-paste fixes. **[MANUSCRIPT]**

## Smoke test log
- 2026-07-19: `smoke_test.py` on Dataset 1 (ds1 subject `a`, seeds 42/43,
  tiny SA budget) — **PASSED 13/13**. Methods exercised: DAG–SA, single-best,
  full-pool soft-vote, random search, Riemannian. EEGNet path is import-guarded
  and skipped when torch is absent in the local sandbox; it runs on the Colab
  GPU runtime (and locally once `torch` is installed). Checkpoints verified to
  reload identically; no NaNs in metrics; all output files present and
  non-empty.
