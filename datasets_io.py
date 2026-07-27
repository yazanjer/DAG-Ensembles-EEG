"""
datasets_io.py
==============
Dataset abstraction layer (Reviewer 1 comment #3).

Exposes a single interface::

    X, y, fs, channel_names, class_names = load_dataset(name, dataset_dir, subject=...)

`X` has shape (n_trials, n_channels, n_samples), `y` is an integer label vector.
Two datasets are supported behind the same interface:

  * "ds1"  — BCI Competition IV Dataset 1 (the original 2-class MI data, .mat).
  * "ds2a" — BCI Competition IV Dataset 2a (9 subjects). Supports the binary
             left/right-hand subset (for comparability with Dataset 1) or the
             full 4-class problem. The 2a files must be supplied by the user
             (see the load_ds2a_subject docstring); this loader fails fast with
             an explicit message listing the expected file/path if absent.

Nothing here fabricates data: missing files raise FileNotFoundError with the
exact path that was expected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io


ArrayTuple = Tuple[np.ndarray, np.ndarray, int, list, list]


# --------------------------------------------------------------------------- #
# Shared epoching / filtering helpers
# --------------------------------------------------------------------------- #
def create_epochs(cnt: np.ndarray, mrk_pos: np.ndarray, mrk_y: np.ndarray,
                  fs: int, window_start: float = 0.0,
                  window_end: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """Cut continuous EEG into (trials, channels, samples) epochs."""
    start_offset = int(window_start * fs)
    end_offset = int(window_end * fs)
    epoch_length = end_offset - start_offset
    num_trials = len(mrk_pos)
    num_channels = cnt.shape[1]
    X = np.zeros((num_trials, num_channels, epoch_length))
    valid = []
    for i in range(num_trials):
        t_start = mrk_pos[i] + start_offset
        t_end = mrk_pos[i] + end_offset
        if t_end <= cnt.shape[0]:
            X[i] = cnt[t_start:t_end, :].T
            valid.append(i)
    return X[valid], mrk_y[valid]


# --------------------------------------------------------------------------- #
# Dataset 1
# --------------------------------------------------------------------------- #
def load_ds1_subject(dataset_dir: Path, subject: str,
                     window: Tuple[float, float] = (0.0, 4.0)) -> ArrayTuple:
    """
    Load one BCI IV Dataset 1 subject (a..g) from the calibration .mat file.
    Returns binary labels mapped to {0, 1}.
    """
    dataset_dir = Path(dataset_dir)
    f_path = dataset_dir / f"BCICIV_calib_ds1{subject}.mat"
    if not f_path.exists():
        raise FileNotFoundError(
            f"[ds1] Expected file not found: {f_path}\n"
            f"      Place BCICIV_calib_ds1{subject}.mat in {dataset_dir}."
        )
    data = scipy.io.loadmat(f_path, struct_as_record=True)
    raw_eeg = data["cnt"].astype(float)
    mrk_pos = data["mrk"]["pos"][0][0].flatten()
    mrk_y = data["mrk"]["y"][0][0].flatten()
    fs = int(data["nfo"]["fs"][0][0].flatten()[0])
    class_names = [str(c[0]) for c in data["nfo"]["classes"][0][0].flatten()]
    try:
        ch_names = [str(c[0]) for c in data["nfo"]["clab"][0][0].flatten()]
    except Exception:
        ch_names = [f"ch{i}" for i in range(raw_eeg.shape[1])]

    X, y_raw = create_epochs(raw_eeg, mrk_pos, mrk_y, fs, window[0], window[1])
    y = np.where(y_raw == -1, 0, 1).astype(int)
    return X, y, fs, ch_names, class_names


# --------------------------------------------------------------------------- #
# Dataset 2a
# --------------------------------------------------------------------------- #
# BCI IV 2a MI event codes (GDF): 769=left hand, 770=right hand,
# 771=feet, 772=tongue. Binary subset uses left(769) vs right(770).
_DS2A_MI_EVENTS = {769: 0, 770: 1, 771: 2, 772: 3}
_DS2A_CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]


def load_ds2a_subject(dataset_dir: Path, subject: int,
                      window: Tuple[float, float] = (2.0, 6.0),
                      variant: str = "binary",
                      session: str = "T") -> ArrayTuple:
    """
    Load one BCI IV Dataset 2a subject (1..9).

    Expected files (place any one of these in <dataset_dir>):
      * MAT  : A0{subject}{session}.mat  — the widely used Kaggle/BBCI export:
               a `data` cell of per-run structs, each with fields
               X (samples x 25: 22 EEG + 3 EOG), trial (onset sample indices),
               y (class 1..4), fs (250), classes, artifacts.
      * GDF  : A0{subject}{session}.gdf  — raw GDF (needs `mne`).

    `window` is expressed in **seconds relative to each trial onset**; the
    default (2.0, 6.0) takes the 4 s motor-imagery period that begins ~2 s after
    the cue, matching common 2a practice.

    variant : 'binary' (left vs right hand → labels 0/1) or '4class'.
    session : 'T' (training) or 'E' (evaluation).

    Fails fast with the expected path if the file is missing (nothing is
    fabricated). EOG channels (last 3) are dropped, leaving 22 EEG channels.
    """
    dataset_dir = Path(dataset_dir)
    stem = f"A0{subject}{session}"
    gdf = dataset_dir / f"{stem}.gdf"
    mat = dataset_dir / f"{stem}.mat"

    if mat.exists():
        X, y, fs, ch_names = _load_ds2a_mat(mat, window)
    elif gdf.exists():
        X, y, fs, ch_names = _load_ds2a_gdf(gdf, window)
    else:
        raise FileNotFoundError(
            f"[ds2a] No file for subject {subject}. Expected one of:\n"
            f"      {mat}\n      {gdf}\n"
            f"      Download BCI IV Dataset 2a and place it in {dataset_dir}."
        )

    if variant == "binary":
        keep = np.isin(y, [0, 1])
        X, y = X[keep], y[keep].astype(int)
        class_names = _DS2A_CLASS_NAMES[:2]
    else:
        class_names = _DS2A_CLASS_NAMES
    return X, y, fs, ch_names, class_names


def _load_ds2a_gdf(gdf_path: Path, window: Tuple[float, float]):
    try:
        import mne
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "[ds2a] Reading GDF requires mne. `pip install mne`."
        ) from exc
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose="ERROR")
    # Drop EOG channels if present.
    eeg_picks = mne.pick_types(raw.info, eeg=True, eog=False)
    raw.pick(eeg_picks)
    fs = int(raw.info["sfreq"])
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    # Map MI annotation codes to 0..3.
    wanted = {k: v for k, v in event_id.items()
              if _annotation_to_mi(k) is not None}
    if not wanted:
        raise ValueError(f"[ds2a] No MI events found in {gdf_path}.")
    picks_ev = {code: _annotation_to_mi(name) for name, code in wanted.items()}
    tmin, tmax = window
    epochs = mne.Epochs(raw, events, event_id=[c for c in picks_ev],
                        tmin=tmin, tmax=tmax, baseline=None,
                        preload=True, verbose="ERROR")
    X = epochs.get_data()  # (trials, channels, samples)
    y = np.array([picks_ev[c] for c in epochs.events[:, -1]])
    ch_names = epochs.ch_names
    return X, y, fs, ch_names


def _annotation_to_mi(name: str):
    """Map an mne annotation description to a 0..3 MI class, else None."""
    for code, cls in _DS2A_MI_EVENTS.items():
        if str(code) in str(name):
            return cls
    return None


def _load_ds2a_mat(mat_path: Path, window: Tuple[float, float]):
    """
    Read the Kaggle/BBCI `data`-cell export of Dataset 2a.

    Each run struct: X (samples x 25), trial (onsets), y (1..4), fs, artifacts.
    Only runs that actually contain trials (the 6 MI runs) are epoched.
    Channels 23-25 are EOG and are dropped. `window` is seconds after onset.
    """
    d = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    if "cnt" in d and "mrk" in d:  # ds1-style fallback
        raw_eeg = d["cnt"].astype(float)
        mrk_pos = np.asarray(d["mrk"].pos).flatten()
        mrk_y = np.asarray(d["mrk"].y).flatten()
        fs = int(np.asarray(d["nfo"].fs).flatten()[0])
        X, y = create_epochs(raw_eeg, mrk_pos, mrk_y, fs, window[0], window[1])
        return X, (y.astype(int) - int(y.min())), fs, \
            [f"ch{i}" for i in range(raw_eeg.shape[1])]

    if "data" not in d:
        raise ValueError(f"[ds2a] Unrecognised .mat layout in {mat_path}.")

    runs = np.atleast_1d(d["data"])
    X_all, y_all, fs = [], [], None
    for run in runs:
        trials = np.atleast_1d(getattr(run, "trial", []))
        labels = np.atleast_1d(getattr(run, "y", []))
        if trials.size == 0 or labels.size == 0:
            continue  # eye-movement / baseline run, no MI trials
        fs = int(run.fs)
        Xr = np.asarray(run.X, dtype=float)          # (samples, 25)
        Xr = Xr[:, :22]                              # drop 3 EOG channels
        start = int(round(window[0] * fs))
        length = int(round((window[1] - window[0]) * fs))
        for onset, cls in zip(trials.astype(int), labels.astype(int)):
            s = onset + start
            e = s + length
            if e <= Xr.shape[0]:
                X_all.append(Xr[s:e, :].T)           # (channels, samples)
                y_all.append(cls)
    if not X_all:
        raise ValueError(f"[ds2a] No MI trials parsed from {mat_path}.")
    X = np.stack(X_all)
    y = np.asarray(y_all, dtype=int) - 1             # 1..4 -> 0..3
    ch_names = [f"eeg{i:02d}" for i in range(22)]
    return X, y, fs, ch_names


# --------------------------------------------------------------------------- #
# Unified entry point
# --------------------------------------------------------------------------- #
def load_dataset(name: str, dataset_dir, subject,
                 window: Optional[Tuple[float, float]] = None,
                 variant: str = "binary", session: str = "T") -> ArrayTuple:
    """
    Unified loader.

    name    : 'ds1' or 'ds2a'
    subject : 'a'..'g' for ds1; 1..9 for ds2a
    """
    dataset_dir = Path(dataset_dir)
    name = name.lower()
    if name == "ds1":
        return load_ds1_subject(dataset_dir, str(subject),
                                window=window or (0.0, 4.0))
    if name == "ds2a":
        return load_ds2a_subject(dataset_dir, int(subject),
                                 window=window or (0.5, 4.5),
                                 variant=variant, session=session)
    raise ValueError(f"Unknown dataset '{name}'. Use 'ds1' or 'ds2a'.")


def available_subjects(name: str, dataset_dir) -> list:
    """Return the subjects for which files are actually present on disk."""
    dataset_dir = Path(dataset_dir)
    found = []
    if name.lower() == "ds1":
        for s in "abcdefg":
            if (dataset_dir / f"BCICIV_calib_ds1{s}.mat").exists():
                found.append(s)
    elif name.lower() == "ds2a":
        for s in range(1, 10):
            if (dataset_dir / f"A0{s}T.gdf").exists() or \
               (dataset_dir / f"A0{s}T.mat").exists():
                found.append(s)
    return found
