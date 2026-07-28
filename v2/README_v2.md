# EEG_v2 — six proposed improvements to DAG-SA, set up as an incremental study

`EEG/` is untouched: it is the code that produced the results cited in the
revision of MLWA-D-26-01081, and it must stay reproducible. Everything here is
a copy plus the new components.

## The reasoning behind these six changes

The revision's own evidence says the bottleneck is **not** the optimiser or the
search budget:

* budgeted random search, at an identical budget in an identical space, scores
  69.0 % against DAG-SA's 66.4 % — so the annealed acceptance rule is not the
  active ingredient;
* the search optimises its objective successfully (the convergence traces climb)
  while test accuracy does not follow — so the problem is the objective;
* the objective is accuracy on ~30 validation trials used to rank ~10^11
  candidate topologies, and the stacking operator is fitted *and* scored on
  those same trials, which is why stacking is selected half the time.

So the changes attack two things: **make the selection signal less noisy and
unbiased** (items 1, 2, 3, 5) and **put the strong baselines inside the pool**
(items 4, 6), because at present the search cannot reach a Riemannian-level
solution by construction.

| Variant | Item | Change |
|---|---|---|
| `V0_published` | — | published method: validation objective, in-sample stacking |
| `V1_oof_objective` | 1 | out-of-fold objective (feature pipeline refitted per inner fold) + cross-validated stacking |
| `V2_regularised_selection` | 2 | complexity penalty + one-standard-error rule |
| `V3_topk_average` | 3 | average of the 5 best topologies instead of the single argmax |
| `V4_enriched_pool` | 4 | per-band Riemannian tangent-space views + filter-bank CSP views |
| `V5_diversity` | 5 | reward mean pairwise disagreement alongside accuracy |
| `V6_shrinkage_csp` | 6 | OAS shrinkage for the CSP covariance |
| `VALL_all_six` | 1–6 | everything at once |

Each variant differs from V0 in **exactly one** respect and runs on **identical
splits and seeds**, so every comparison is paired trial-by-trial and McNemar's
exact test applies directly.

## New files

| File | What it does |
|---|---|
| `selection.py` | out-of-fold probability matrix, the topology objective (accuracy, complexity penalty, diversity), out-of-fold stacking, the one-standard-error rule, top-k averaging |
| `pool_ext.py` | Riemannian tangent-space and FBCSP feature views and their pool members |
| `variants.py` | the eight variant definitions and the pool-sharing signature |
| `run_v2.py` | the campaign driver — writes per-seed results, a summary, McNemar against V0 and a win/tie/loss table |
| `analyse_v2.py` | turns a campaign into `v2_report.md` and `v2_table.tex` |
| `config_smoke.yaml` | a tiny configuration for wiring checks only |
| `DAG_SA_v2_Colab.ipynb` | Colab orchestration |

`dag_core.py` has three small additions, all inert unless switched on: two new
`FeatureType` members, an optional `scorer` on the annealer, and a history of
visited topologies. Single-precision band-pass filtering was added to halve peak
memory (the out-of-fold loop holds two feature pipelines at once).

## Just run everything (recommended)

`DAG_SA_v2_RunAll.ipynb` is the run-and-forget notebook: **Runtime -> Run all**,
click the Drive authorisation once, and leave it. It runs both strong-member
campaigns (Dataset 2a, 72 units; Dataset 1, 32 units), resumes after a
disconnect, survives a failing unit, mirrors everything to
`results/run_all.log`, and writes a single `results/SUMMARY.md` with the
accuracy table, the McNemar win/tie/loss column and the diagnostic that says
whether the search actually selected the strong members. Budget 4-7 h on a GPU
runtime.

`run_all.py` holds the orchestration, so it can be run headless too:

```bash
python run_all.py --project-root . --ds1-dir ../EEG/dataset \
                  --ds2a-dir "../BCI Competition IV Dataset 2a"
```

`DAG_SA_v2_Colab.ipynb` remains the step-by-step notebook for running single
campaigns interactively.

## Running it (step by step)

```bash
# wiring check, about a minute
python run_v2.py --dataset ds1 --subjects a --seeds 42 --tiny \
                 --config config_smoke.yaml --dataset-dir ../EEG/dataset

# the real campaign
python run_v2.py --dataset ds1 --subjects a b f g \
                 --seeds 42 43 44 45 46 47 48 49 --experiment ds1_v2

# read the outcome
python analyse_v2.py results/ds1_v2
```

Cost, measured on a 4-core CPU with the real 540-member pool: roughly 8–12
minutes per (subject, seed) unit for all eight variants, dominated by the
out-of-fold matrices (`V1`, `VALL`), which refit the whole feature pipeline once
per inner fold.

| Scope | Units | Wall clock (CPU) |
|---|---|---|
| 8 variants, 4 subjects × 8 seeds | 32 | 4–6 h |
| 8 variants, 4 subjects × 3 seeds | 12 | 1.5–2.5 h |
| cheap variants only (`V0 V2 V3 V4 V5 V6`) | 32 | 1–1.5 h |

Results are written after **every** unit, so a disconnect costs at most one
unit. Start with the cheap variants: they tell you whether the selection-rule
changes matter before you pay for the out-of-fold ones.

## Campaign 1 result (Dataset 1, 32 units x 8 variants, ~5 h)

None of the six changes improved on the published method. Accuracy, seed-level
mean over 4 subjects x 8 seeds, with McNemar against V0:

| Variant | Accuracy | Δ vs V0 | W/T/L |
|---|---|---|---|
| V5 diversity | 67.7 ± 4.3 | +1.0 | 0/32/0 |
| VALL | 67.2 ± 5.7 | +0.5 | 0/31/1 |
| V4 enriched pool | 67.0 ± 3.3 | +0.3 | 0/30/2 |
| V2 regularised selection | 66.9 ± 3.3 | +0.2 | 1/29/2 |
| **V0 published** | **66.7 ± 3.2** | — | — |
| V6 shrinkage CSP | 65.8 ± 7.7 | −0.8 | 2/29/1 |
| V3 top-k average | 64.8 ± 4.7 | −1.9 | 2/28/2 |
| V1 out-of-fold objective | 61.9 ± 4.9 | −4.8 | 2/28/2 |

**The number that governs how to read this.** Re-running the *identical*
published method with a different RNG trajectory (V0 here vs `results 3/`) gives
a unit-level standard deviation of **17 points** and a mean absolute difference
of 13. The standard error of any 32-unit mean difference is therefore about
**3 points** — larger than every effect in the table except V1. The method's own
run-to-run variability dwarfs all six interventions, which is a more interesting
finding than any of the deltas.

### Two defects found in that campaign, both now fixed

1. **The enriched pool was unreachable.** `_pick_family` draws the family from
   `pool.feature_types`, and `extend_pool` appended members without registering
   their families: 0 of 32 selected topologies contained a RIEM or FB member, so
   V4 was V0 plus 21 dead pool entries. Item 4 was never tested.
2. **The "strong baselines" were only cousins of them.** The paper's B5 runs on
   *unfiltered* epochs with a logistic-regression head; the members added were
   band-passed with SVM heads. EEGNet was not representable at all.

`pool_ext.py` now provides a `RAW` view plus `RiemannianExactNode` (verified to
reproduce B5 exactly: 0.767 vs 0.767 on subject a / seed 42) and `EEGNetNode`
(with the softmax head that `EEGNetClassifier` lacks). Both share the `STRONG`
family, so a committee can be a committee of strong baselines. `V4` embeds the
exact B5; the new opt-in **`V7_strong_members`** adds EEGNet.

The EEGNet node is code-reviewed but **unexecuted** — torch could not be
installed in the environment where this was written.

### Two further defects found while running campaign 2 (both fixed)

3. **`copy.deepcopy` crashed on torch-backed members.** The search deep-copies a
   candidate DAG on every perturbation, which copied the leaf models too;
   `EEGNetClassifier` holds a reference to the `torch` module, so the run died
   with `TypeError: cannot pickle 'module' object` the moment an EEGNet member
   entered a topology. `BaseClassifierNode.__deepcopy__` now returns `self` --
   members are immutable after `fit`, so sharing them is correct, and it also
   removes four cloned SVMs per iteration.
4. **The sklearn compatibility patch was not idempotent.** Re-importing
   `dag_core` (an `importlib.reload`, or simply re-running the import cell in
   Colab) re-captured the already patched `check_X_y` as the "original", so the
   wrapper called itself: every fit raised `RecursionError` and the entire pool
   was discarded as unfittable. The patch is now guarded by a sentinel.

Also: members that fail to fit are now **dropped** from the pool rather than
left selectable, and a unit that raises is logged and skipped instead of ending
the campaign. `run_campaign(..., resume=True)` skips `(subject, seed)` units
already present in the results CSV.

### The constraint problem, and the variants that address it

With `same_family` the strong members are effectively unreachable: the family is
drawn once at initialisation (~1 run in 5), and the `STRONG` family holds fewer
members than $M$, so `get_random_distinct` silently relaxes to the whole pool.
Three new variants give item 4 a fair test:

| Variant | What it does |
|---|---|
| `V4u_enriched_unconstrained` | V4 with the constraint lifted -- heterogeneous committees mixing CSP, CTP, tangent-space and FBCSP members |
| `V7u_strong_unconstrained` | EEGNet and the exact B5 as members, constraint lifted |
| `V7w_strong_warmstart` | initial committee drawn from the strong baselines, then free to drift |
| `V7l_strong_locked` | committee **locked** to the strong baselines: EEGNet and B5 fused by the searched operators |

`V7l` is the decisive one -- if fusing the strong baselines does not beat either
of them alone, item 4 is answered.

### Campaign 2 (set up, not yet run)

Dataset 2a, nine subjects, eight seeds, variants `V0_published`,
`V4u_enriched_unconstrained`, `V7u_strong_unconstrained`, `V7l_strong_locked`.
A first attempt stopped after 5 of 72 units on the deepcopy crash above; those
results are not usable. That is the dataset where EEGNet leads
by 6.4 points, so it is the only place where embedding the strong baselines can
plausibly change the answer. Section 6 of the notebook runs it and reports how
often the search actually selected a strong member.

## What has and has not been tested

Campaign 1 (Dataset 1, all eight variants, 32 units) has been run in full; the
result is above. Every code path of the *fixed* pool extension has been
exercised on real Dataset 1 and 2a data, including the exact-B5 member and the
family registration, and a short Dataset 2a run confirms the search now selects
the new families. The EEGNet member has not been executed anywhere — it needs
torch, which was unavailable in the authoring environment. Campaign 2 has not
been run.

## Before you read the results

Decide now that you will report the outcome either way, and read the
win/tie/loss column before the Δ column. On these cohorts a two- or three-point
mean difference will usually come out as a tie, and the tie is the honest
answer. Selecting the winning variant after the fact and reporting only that one
is precisely the practice the current revision exists to correct — it is how the
original submission ended up claiming an advantage that eight seeds erased.

A plausible and publishable outcome is that everything ties: that would say the
ceiling on this pool and these cohorts is set by the data rather than by the
search, which is a cleaner claim than a marginal accuracy gain.
