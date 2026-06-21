"""EEGMAT (PhysioNet eegmat 1.0.0) — a SECOND device for true cross-device transfer.

Zyma et al. 2019: 36 subjects, Neurocom 23-channel 10-20 EEG @500 Hz. Each subject has
two recordings:
  * Subject<NN>_1.edf  — 3 min eyes-open REST            -> our "Relaxed"
  * Subject<NN>_2.edf  — mental ARITHMETIC (serial subtraction) -> our "Concentrated"

This mirrors the custom Emotiv Relaxed-vs-Concentrated contrast EXACTLY, but on a
different device and cohort — so it is the missing piece that turns the flagship's
transfer experiment (R2) from a cross-paradigm stress test into a genuine same-contrast
cross-device test. We extract the same frontal pair (F7/F8), resample 500->128 Hz, and
feed the identical feature/covariance pipeline.
"""
from __future__ import annotations

import urllib.request

import numpy as np

import mne

from yoga_impact import config
from yoga_impact.io_custom import Recording

EEGMAT_ROOT = config.DATA_ROOT / "eegmat"
PHYSIONET_BASE = "https://physionet.org/files/eegmat/1.0.0/"
N_SUBJECTS = 36
# condition file suffix -> (state label used by the project, recording tag)
COND = {1: ("Relaxed", "rest"), 2: ("Concentrated", "arith")}
PICK = ("EEG F7", "EEG F8")


def ensure_downloaded() -> int:
    """Fetch any missing EEGMAT EDFs from PhysioNet so the pipeline runs from a clean
    checkout. Idempotent: existing non-empty files are skipped. Returns #present."""
    EEGMAT_ROOT.mkdir(parents=True, exist_ok=True)
    present = 0
    for s in range(N_SUBJECTS):
        for c in COND:
            fn = f"Subject{s:02d}_{c}.edf"
            out = EEGMAT_ROOT / fn
            if out.exists() and out.stat().st_size > 1000:
                present += 1
                continue
            try:
                print(f"  [eegmat] downloading {fn} ...", flush=True)
                urllib.request.urlretrieve(PHYSIONET_BASE + fn, out)
                present += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [eegmat] FAILED {fn}: {exc}")
    return present


def _load_edf(path, resample: float = 128.0):
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    if not all(ch in raw.ch_names for ch in PICK):
        return None
    raw.pick(list(PICK))
    raw.load_data(verbose="ERROR")
    if resample and raw.info["sfreq"] != resample:
        raw.resample(resample, verbose="ERROR")
    data = raw.get_data() * 1e6  # V -> microvolts
    return data.astype(np.float32), float(raw.info["sfreq"])


def load_recordings(resample: float = 128.0):
    """One Recording per (subject, condition); F7/F8 @128 Hz, device 'Neurocom'."""
    ensure_downloaded()
    recs = []
    for s in range(N_SUBJECTS):
        for c, (state, tag) in COND.items():
            path = EEGMAT_ROOT / f"Subject{s:02d}_{c}.edf"
            if not path.exists():
                continue
            loaded = _load_edf(path, resample)
            if loaded is None:
                continue
            data, fs = loaded
            recs.append(Recording(
                subject=f"eegmat{s:02d}", state=state, device="Neurocom", path=path,
                recording_id=f"eegmat{s:02d}_{tag}", fs=fs,
                channels=["F7", "F8"], data=data))
    return recs


def _load_edf_full(path, resample: float = 128.0):
    """Load the full scalp montage (all 'EEG *' channels except the A2-A1 reference)."""
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    picks = [c for c in raw.ch_names if c.startswith("EEG ") and c != "EEG A2-A1"]
    if len(picks) < 8:
        return None
    raw.pick(picks)
    raw.load_data(verbose="ERROR")
    if resample and raw.info["sfreq"] != resample:
        raw.resample(resample, verbose="ERROR")
    data = raw.get_data() * 1e6  # V -> microvolts
    names = [c.replace("EEG ", "") for c in picks]
    return data.astype(np.float32), float(raw.info["sfreq"]), names


def load_recordings_full(resample: float = 128.0):
    """One Recording per (subject, condition) with the FULL 19-channel 10-20 montage.

    Used by the multi-channel Riemannian experiment, where the spatial covariance across
    all electrodes (functional connectivity) is exactly the structure that per-channel
    band power cannot represent — so the manifold can genuinely outperform the classical
    baseline here, unlike the 2-channel frontal case."""
    ensure_downloaded()
    recs = []
    for s in range(N_SUBJECTS):
        for c, (state, tag) in COND.items():
            path = EEGMAT_ROOT / f"Subject{s:02d}_{c}.edf"
            if not path.exists():
                continue
            loaded = _load_edf_full(path, resample)
            if loaded is None:
                continue
            data, fs, names = loaded
            recs.append(Recording(
                subject=f"eegmat{s:02d}", state=state, device="Neurocom", path=path,
                recording_id=f"eegmat{s:02d}_{tag}", fs=fs,
                channels=names, data=data))
    return recs


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    rr = load_recordings()
    print(f"loaded {len(rr)} recordings from {EEGMAT_ROOT}")
    from collections import Counter
    print("by state:", Counter(r.state for r in rr))
    print("subjects:", len({r.subject for r in rr}))
