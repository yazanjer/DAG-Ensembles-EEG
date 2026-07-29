"""
make_notebook.py -- generate the Colab notebook.

Design note: the notebook is a THIN DRIVER. It configures paths, fetches the
code, and calls three command-line scripts:

    run_study.py    the confirmatory run (resumable)
    make_report.py  tables + the pre-registered verdict
    verify.py       integrity checks

No science lives in the notebook. That matters for a paper: a notebook whose
cells contain the method is impossible to diff, impossible to review, and
diverges from the repository the moment either is edited. Here the notebook
and a terminal run execute byte-identical code, and `git log` on v3/ is the
authoritative history.

Regenerate with:  python make_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "DAG_SA_v3_Colab.ipynb"

REPO_URL = "https://github.com/yazanjer/DAG-Ensembles-EEG.git"
BRANCH = "v3-audit-redesign"


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


# --------------------------------------------------------------------------- #
INTRO = """# DAG-SA v3 — confirmatory evaluation

Thin driver. Every cell below calls a script in `v3/`; none of them contain
method code, so this notebook and a terminal run execute identical code and
the repository stays the single source of truth.

| step | script | produces |
|---|---|---|
| 1 | `run_study.py` | `predictions.jsonl`, `preregistration.json`, `run.log` |
| 2 | `make_report.py` | `report.md`, `report.json`, the verdict |
| 3 | `verify.py` | integrity checks (exits non-zero on failure) |

**Before running**

1. Datasets on Drive as `MyDrive/EEG_DAGSA/dataset/BCICIV_calib_ds1{a..g}.mat`
   and `MyDrive/EEG_DAGSA/dataset_2a/A0{1..9}T.mat`
2. **Runtime → Change runtime type → GPU.** Without one, EEGNet is impractical;
   it is the baseline most likely to be competitive on Dataset 2a, so leaving
   it out leaves the comparison incomplete.
3. Run all.

**Resumable.** Each completed unit is appended to `predictions.jsonl` and
fsync'd. After a disconnect, just run all again — finished units are skipped
and at most the unit in progress is lost.

**Runtime.** Everything except EEGNet is CPU-bound, roughly 40 minutes for all
three datasets. EEGNet dominates the rest; on a T4 expect 3–5 hours total.
"""

CONFIG = '''#@title Configuration { display-mode: "form" }
DRIVE_ROOT   = "/content/drive/MyDrive/EEG_DAGSA"  #@param {type:"string"}
DATASETS     = "ds2a_binary,ds1,ds2a_4class"       #@param {type:"string"}
RUN_EEGNET   = True   #@param {type:"boolean"}
RUN_ABLATION = True   #@param {type:"boolean"}

CODE_SOURCE  = "github"  #@param ["github", "drive"]
REPO_URL     = "%(repo)s"  #@param {type:"string"}
BRANCH       = "%(branch)s"  #@param {type:"string"}
DRIVE_CODE   = "/content/drive/MyDrive/EEG_DAGSA/code_v3"  #@param {type:"string"}

OUT_DIR  = f"{DRIVE_ROOT}/results_v3"
CODE_DIR = "/content/dagsa_v3"
print("results ->", OUT_DIR)
''' % {"repo": REPO_URL, "branch": BRANCH}

MOUNT = '''# Mount Drive
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print("not on Colab, or already mounted:", e)

import os
os.makedirs(OUT_DIR, exist_ok=True)
'''

FETCH = '''# Fetch the code. CODE_SOURCE="github" clones the repo; "drive" copies
# a folder you placed on Drive yourself (use that if the branch is not pushed).
import os, shutil, subprocess, sys

def sh(*cmd):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-2000:] or "", end="")
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)
    return r.returncode

if os.path.isdir(CODE_DIR):
    shutil.rmtree(CODE_DIR)

if CODE_SOURCE == "github":
    rc = sh("git", "clone", "--depth", "1", "--branch", BRANCH,
            REPO_URL, "/content/_repo")
    if rc != 0:
        raise SystemExit(
            f"clone of branch '{BRANCH}' failed.\\n"
            "If the branch is not pushed yet, upload the v3/ folder to Drive "
            "and set CODE_SOURCE='drive' in the config cell.")
    shutil.copytree("/content/_repo/v3", CODE_DIR)
    sh("git", "-C", "/content/_repo", "log", "--oneline", "-1")
else:
    if not os.path.isdir(DRIVE_CODE):
        raise SystemExit(f"DRIVE_CODE not found: {DRIVE_CODE}")
    shutil.copytree(DRIVE_CODE, CODE_DIR)

sys.path.insert(0, CODE_DIR)
print("\\ncode in", CODE_DIR)
print(sorted(f for f in os.listdir(CODE_DIR) if f.endswith(".py")))
'''

DEPS = '''# Dependencies. numpy/scipy/scikit-learn/pandas ship with Colab; torch is
# needed only for the EEGNet baseline and ships with GPU runtimes.
import importlib, subprocess, sys
for mod, pip in [("numpy","numpy"), ("scipy","scipy"),
                 ("sklearn","scikit-learn"), ("pandas","pandas")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.run([sys.executable,"-m","pip","install","-q",pip], check=True)

try:
    import torch
    print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
    if RUN_EEGNET and not torch.cuda.is_available():
        print("\\n!! No GPU. EEGNet will be extremely slow.")
        print("   Runtime > Change runtime type > GPU, then re-run.")
except ImportError:
    print("!! torch not available -- EEGNet will be recorded as skipped.")
'''

RUN = '''# Step 1 -- the confirmatory run. Resumable: re-run this cell after a
# disconnect and completed units are skipped.
import shlex, subprocess, sys

args = ["--drive-root", DRIVE_ROOT, "--out", OUT_DIR, "--datasets", DATASETS]
if not RUN_EEGNET:
    args.append("--no-eegnet")
if not RUN_ABLATION:
    args.append("--no-ablation")

cmd = [sys.executable, "-u", f"{CODE_DIR}/run_study.py"] + args
print("$", " ".join(shlex.quote(c) for c in cmd), "\\n", flush=True)

# Stream output live so a long run shows progress rather than going silent.
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    print(line, end="")
proc.wait()
print("\\nexit code:", proc.returncode)
'''

REPORT = '''# Step 2 -- tables and the pre-registered verdict.
import subprocess, sys
r = subprocess.run([sys.executable, f"{CODE_DIR}/make_report.py",
                    "--results", OUT_DIR], capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print(r.stderr, file=sys.stderr)
'''

VERIFY = '''# Step 3 -- integrity checks.
# Non-zero exit means something is wrong. The key line is "units per scored
# method": it must be equal for every row. An unequal row means a method
# failed on some units and the comparison is not like-for-like (audit B5).
import subprocess, sys
r = subprocess.run([sys.executable, f"{CODE_DIR}/verify.py",
                    "--results", OUT_DIR], capture_output=True, text=True)
print(r.stdout)
if r.stderr:
    print(r.stderr, file=sys.stderr)
print("exit code:", r.returncode)
'''

REPORT_DISPLAY = '''# Render report.md inline
from IPython.display import Markdown, display
from pathlib import Path
p = Path(OUT_DIR) / "report.md"
display(Markdown(p.read_text() if p.exists() else "*no report.md yet*"))
'''

FILES = '''# What was written
import os
for f in sorted(os.listdir(OUT_DIR)):
    p = os.path.join(OUT_DIR, f)
    if os.path.isfile(p):
        print(f"{os.path.getsize(p)/1024:9.1f} KB  {f}")
'''

OUTRO = """## Reading the result

`make_report.py` prints a verdict per dataset against the criterion frozen in
`preregistration.json`: superiority requires, **against every baseline**, a 95%
paired CI lower bound above zero *and* a point estimate of at least 2.0
accuracy points. A CI lying entirely within ±2.0 points is a declared tie.
Anything else is `INCONCLUSIVE`.

Two things to resist once the numbers appear:

* **Do not re-tune after seeing them.** That converts a pre-registered result
  into a post-hoc one — the exact objection two reviewers already raised.
* **Do not drop a baseline, subject, seed or metric.** The pre-registration
  forbids it and the table is meant to be reported in full.

If EEGNet lands above roughly 83 on 2a binary, ARTS no longer leads that
dataset and the framing has to change: report EEGNet as strongest on 2a and
ARTS as the strongest non-deep method.
"""


def build():
    cells = [
        md(INTRO),
        code(CONFIG),
        md("## Setup"),
        code(MOUNT),
        code(FETCH),
        code(DEPS),
        md("## 1. Run the study"),
        code(RUN),
        md("## 2. Report"),
        code(REPORT),
        code(REPORT_DISPLAY),
        md("## 3. Integrity checks"),
        code(VERIFY),
        code(FILES),
        md(OUTRO),
    ]

    nb = {"cells": cells,
          "metadata": {"accelerator": "GPU",
                       "colab": {"provenance": [], "toc_visible": True},
                       "kernelspec": {"display_name": "Python 3",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT}  ({len(cells)} cells, "
          f"{OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    build()
