# Pushing this project to GitHub (then running on Colab from GitHub)

The **`EEG/` folder is the repo root** — it holds all the code, the notebook,
and `.gitignore`. Datasets, `results/`, and `checkpoints/` are git-ignored on
purpose (they are large and belong on Google Drive, not in git).

## 1. Create the repo on GitHub
Make an **empty** repository (no README/License/.gitignore — this folder already
has them). Copy its URL, e.g. `https://github.com/<user>/<repo>.git`.

## 2. Push from Terminal
Run these from inside the `EEG/` folder (each line, paste + Enter):

```bash
cd "/Users/yazanaljeroudi/Documents/Salwa article 19 July/EEG"
git init
git add .
git status                 # confirm NO .mat/.gdf/results/ files are listed
git commit -m "DAG-SA revision: multi-seed driver, baselines, CV, ablations, Colab notebook"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

If `git status` shows any `*.mat`, `*.gdf`, `results/`, or `checkpoints/`
entries, stop — the `.gitignore` isn't being picked up; re-check you are in the
`EEG/` folder.

## 3. Run on Colab from GitHub
1. Open [colab.research.google.com](https://colab.research.google.com) →
   *File → Open notebook → GitHub* → paste your repo URL → open
   `DAG_SA_Colab.ipynb`.
2. Runtime → Change runtime type → **GPU**.
3. In cell **0**, set `REPO_URL` to your repo (for a private repo use
   `https://<TOKEN>@github.com/<user>/<repo>.git`).
4. Upload the datasets to Drive so Colab can read them:
   ```
   MyDrive/EEG_DAGSA/dataset/BCICIV_calib_ds1?.mat     (Dataset 1)
   MyDrive/EEG_DAGSA/dataset_2a/A0?T.mat               (Dataset 2a)
   ```
5. Run the cells top to bottom. Results are written to
   `MyDrive/EEG_DAGSA/results/<experiment>/`.

## Run-on-Colab checklist (from a fresh GitHub clone)
1. Colab → *File → Open notebook → GitHub* → paste repo URL → `DAG_SA_Colab.ipynb`.
2. Runtime → Change runtime type → **GPU** (EEGNet path).
3. Cell 0: set `REPO_URL`. Cells install deps and mount Drive.
4. **Drive folder layout to create first:**
   ```
   MyDrive/EEG_DAGSA/dataset/BCICIV_calib_ds1a.mat  … ds1g   (Dataset 1)
   MyDrive/EEG_DAGSA/dataset_2a/A01T.mat … A09T.mat          (Dataset 2a)
   ```
   Results/checkpoints are written under `MyDrive/EEG_DAGSA/results/` and
   `…/checkpoints/` automatically.
5. Run the smoke cell (must print `SMOKE TEST PASSED`), then the full-run cells.
6. **If Colab disconnects, just re-run the same cell** — the driver resumes
   (`resume=True`), skips completed `(subject, seed)` units, and continues from
   the latest per-seed checkpoint. Watch `results/<exp>/runtime_report.txt` for
   the live ETA and measured total.

## Notes
* **Authentication:** GitHub needs a Personal Access Token (not your password)
  for HTTPS pushes. Create one at *GitHub → Settings → Developer settings →
  Personal access tokens*. When `git push` asks for a password, paste the token.
* **Data stays private:** because data is git-ignored, your `.mat`/`.gdf` files
  are never uploaded to GitHub; they live only on your Drive.
