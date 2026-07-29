"""
make_notebook.py -- generate the self-contained Colab notebook.

The notebook embeds the source of every module verbatim, so it has no
dependency on this repository being present. Running it top to bottom on a
fresh Colab VM reproduces the whole study.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
MODULES = ["data.py", "pipeline.py", "baselines_v3.py", "protocol.py",
           "analysis.py"]


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


INTRO = """# DAG-SA v3 — confirmatory evaluation

This notebook runs the whole study end to end, unattended, and is safe to
re-run after a Colab disconnect: every completed unit is appended to
`predictions.jsonl` on Drive and skipped on the next run.

**What it produces**

1. `preregistration.json` — the protocol, frozen before the run
2. `predictions.jsonl` — one line per (dataset, subject, seed) unit, holding
   every method's predictions on the identical test trials
3. `report.md` / `report.json` — the comparison table (mean ± sd, 95% CI,
   Cohen's kappa), per-unit paired McNemar with a significance-gated
   win/tie/loss count, per-subject and class-wise metrics, the ablation, and
   the verdict against the pre-registered success criterion
4. `run.log` — a full log

**Before running**, set `DRIVE_ROOT` below. Datasets are expected at
`MyDrive/EEG_DAGSA/dataset` (BCI IV Dataset 1) and `MyDrive/EEG_DAGSA/dataset_2a`
(Dataset 2a).

**Runtime.** Everything except EEGNet is CPU-bound and takes roughly 40 minutes
for all three datasets. EEGNet dominates the rest; on a T4 the full study is
about 3–5 hours. Choose a GPU runtime.
"""

SETUP = '''# ---- configuration ------------------------------------------------------
DRIVE_ROOT = "/content/drive/MyDrive/EEG_DAGSA"
OUT_DIR    = f"{DRIVE_ROOT}/results_v3"
DS1_DIR    = f"{DRIVE_ROOT}/dataset"
DS2A_DIR   = f"{DRIVE_ROOT}/dataset_2a"

RUN_EEGNET   = True     # needs a GPU runtime to be practical
RUN_ABLATION = True
DATASETS     = ("ds2a_binary", "ds1", "ds2a_4class")

import os, sys, subprocess
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print("not on Colab or Drive already mounted:", e)

os.makedirs(OUT_DIR, exist_ok=True)
CODE_DIR = "/content/dagsa_v3"
os.makedirs(CODE_DIR, exist_ok=True)
sys.path.insert(0, CODE_DIR)
print("output ->", OUT_DIR)
'''

DEPS = '''# ---- dependencies -------------------------------------------------------
# scipy / scikit-learn / numpy are preinstalled on Colab. torch is needed only
# for the EEGNet baseline and is preinstalled on GPU runtimes.
import importlib
for mod, pip in [("numpy", "numpy"), ("scipy", "scipy"),
                 ("sklearn", "scikit-learn"), ("pandas", "pandas")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip],
                       check=True)

import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
if RUN_EEGNET and not torch.cuda.is_available():
    print("WARNING: no GPU. EEGNet will be very slow. "
          "Runtime > Change runtime type > GPU.")
'''

RUN = '''# ---- run (resumable) ----------------------------------------------------
# Safe to re-run after a disconnect: completed units are skipped. Results are
# written after EVERY unit and fsync'd, so an interrupted session loses at
# most the unit in progress.
import importlib
import protocol as PR
importlib.reload(PR)

cfg = PR.RunCfg(
    ds1_dir=DS1_DIR, ds2a_dir=DS2A_DIR, out_dir=OUT_DIR,
    datasets=DATASETS, run_eegnet=RUN_EEGNET, run_ablation=RUN_ABLATION,
)
PR.run(cfg)
'''

REPORT = '''# ---- analysis -----------------------------------------------------------
import importlib, json
import analysis as A
importlib.reload(A)

report = A.full_report(f"{OUT_DIR}/predictions.jsonl", OUT_DIR)
from IPython.display import Markdown, display
display(Markdown(A.render_markdown(report)))
'''

VERDICT = '''# ---- the pre-registered verdict, stated plainly -------------------------
for ds, d in report["datasets"].items():
    v = d["verdict"]
    print(f"\\n=== {ds} ===")
    print(f"  VERDICT: {v['verdict']}")
    print(f"  reason : {v['reason']}")
    print(f"  binding comparison: ARTS vs {v['binding_comparison']}  "
          f"{v['binding_diff']:+.2f} pts  "
          f"CI [{v['binding_ci'][0]:+.2f}, {v['binding_ci'][1]:+.2f}]")
    print("  --- ablation (which component earns the gain) ---")
    for r in sorted(d["ablation"], key=lambda x: x["method"]):
        print(f"    {r['method']:22s} {r['acc_mean']:6.2f}")

if "ds1_real_subjects_only" in report:
    v = report["ds1_real_subjects_only"]["verdict"]
    print("\\n=== ds1, four REAL subjects only (pre-specified secondary) ===")
    print(f"  VERDICT: {v['verdict']} ({v['reason']})")
'''

INTEGRITY = '''# ---- integrity checks ---------------------------------------------------
# These are cheap and catch the failure modes the audit found in the previous
# codebase: silently dropped methods, unequal unit counts, and duplicated rows.
import collections, json
recs = [json.loads(l) for l in open(f"{OUT_DIR}/predictions.jsonl") if l.strip()]

keys = [(r["dataset"], str(r["subject"]), r["seed"]) for r in recs]
dups = [k for k, c in collections.Counter(keys).items() if c > 1]
print("duplicate units:", len(dups), dups[:5])

per_method = collections.Counter()
for r in recs:
    for m, p in r["pred"].items():
        if p is not None:
            per_method[m] += 1
print("\\nunits per method (these must all be equal):")
for m, c in sorted(per_method.items(), key=lambda kv: -kv[1]):
    print(f"   {m:22s} {c}")

skipped = collections.Counter()
for r in recs:
    for m, p in r["pred"].items():
        if p is None:
            skipped[m] += 1
print("\\nexplicitly skipped (recorded, not silently dropped):", dict(skipped))

# every method must be scored on exactly the same test trials within a unit
bad = [k for r, k in zip(recs, keys)
       if any(len(p) != len(r["y_true"])
              for p in r["pred"].values() if p is not None)]
print("units with mismatched prediction lengths:", len(bad))
'''


def build():
    cells = [md(INTRO), code(SETUP), code(DEPS),
             md("## Source\\n\\nThe modules are written out verbatim so the "
                "notebook is self-contained.")]
    for mod in MODULES:
        src = (HERE / mod).read_text()
        # JSON-encode rather than embedding in a triple-quoted literal: a
        # stray ''' or a trailing backslash in the source would otherwise
        # produce a notebook that looks fine and fails to parse.
        cell = ("import json\n"
                "_src = json.loads(%s)\n"
                "with open(f'{CODE_DIR}/%s', 'w') as f:\n"
                "    f.write(_src)\n"
                "print('wrote %s', len(_src), 'chars')\n"
                % (repr(json.dumps(src)), mod, mod))
        cells.append(code(cell))
    cells += [md("## Run"), code(RUN),
              md("## Results"), code(REPORT), code(VERDICT),
              md("## Integrity checks"), code(INTEGRITY)]

    nb = {"cells": cells,
          "metadata": {"accelerator": "GPU",
                       "colab": {"provenance": [], "toc_visible": True},
                       "kernelspec": {"display_name": "Python 3",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = HERE / "DAG_SA_v3_Colab.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    build()
