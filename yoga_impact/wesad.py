"""WESAD autonomic axis — loading + physiological feature extraction.

Chest signals @700 Hz (ECG, EMG, EDA, Resp, Temp) are windowed (60 s, 50 % overlap)
into pure-label segments, and each window is reduced to interpretable autonomic
features: HRV (time + frequency), EDA tonic/phasic (SCR), EMG tension, respiration
rate, skin temperature.

Labels: 1 baseline, 2 stress, 3 amusement, 4 meditation (0/5/6/7 ignored).

Features are cached to outputs/wesad_features.csv (extraction is the slow part).
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.stats import linregress

import neurokit2 as nk

from yoga_impact import config

FS = 700.0
WIN_SEC = 60.0
STEP_SEC = 30.0
LABELS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
CHEST = ("ECG", "EMG", "EDA", "Resp", "Temp")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def subjects() -> list[str]:
    return sorted(
        p.name for p in config.WESAD_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("S") and (p / f"{p.name}.pkl").exists()
    )


def load_subject(sid: str):
    with open(config.WESAD_ROOT / sid / f"{sid}.pkl", "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    chest = d["signal"]["chest"]
    sig = {k: np.asarray(chest[k], dtype=np.float64).ravel() for k in CHEST}
    label = np.asarray(d["label"], dtype=int).ravel()
    return sig, label


# ---------------------------------------------------------------------------
# Per-signal features
# ---------------------------------------------------------------------------
def hrv_features(ecg: np.ndarray, fs: float) -> dict:
    keys = ["hrv_meanNN", "hrv_SDNN", "hrv_RMSSD", "hrv_pNN50",
            "hrv_meanHR", "hrv_LF", "hrv_HF", "hrv_LFHF"]
    out = {k: np.nan for k in keys}
    try:
        clean = nk.ecg_clean(ecg, sampling_rate=fs)
        _, info = nk.ecg_peaks(clean, sampling_rate=fs)
        rp = np.asarray(info["ECG_R_Peaks"])
        rr = np.diff(rp) / fs * 1000.0
        rr = rr[(rr > 300) & (rr < 2000)]
        if rr.size < 4:
            return out
        dd = np.diff(rr)
        out["hrv_meanNN"] = float(np.mean(rr))
        out["hrv_SDNN"] = float(np.std(rr))
        out["hrv_RMSSD"] = float(np.sqrt(np.mean(dd ** 2)))
        out["hrv_pNN50"] = float(np.mean(np.abs(dd) > 50) * 100)
        out["hrv_meanHR"] = float(60000.0 / np.mean(rr))
        t = np.cumsum(rr) / 1000.0
        tt = np.arange(t[0], t[-1], 0.25)
        if tt.size > 8:
            rri = np.interp(tt, t, rr) - np.mean(rr)
            f, pxx = sps.welch(rri, fs=4.0, nperseg=min(256, rri.size))
            lf = float(np.trapezoid(pxx[(f >= 0.04) & (f < 0.15)], f[(f >= 0.04) & (f < 0.15)]))
            hf = float(np.trapezoid(pxx[(f >= 0.15) & (f < 0.4)], f[(f >= 0.15) & (f < 0.4)]))
            out["hrv_LF"], out["hrv_HF"] = lf, hf
            out["hrv_LFHF"] = lf / hf if hf > 0 else np.nan
    except Exception:  # noqa: BLE001
        pass
    return out


def eda_features(eda: np.ndarray, fs: float) -> dict:
    x = eda.astype(float)
    t = np.arange(x.size) / fs
    out = {
        "eda_mean": float(np.mean(x)), "eda_std": float(np.std(x)),
        "eda_min": float(np.min(x)), "eda_max": float(np.max(x)),
        "eda_range": float(np.ptp(x)),
    }
    try:
        out["eda_slope"] = float(linregress(t, x).slope)
    except Exception:  # noqa: BLE001
        out["eda_slope"] = np.nan
    try:
        sos = sps.butter(2, 0.05, "hp", fs=fs, output="sos")
        ph = sps.sosfiltfilt(sos, x)
        pk, props = sps.find_peaks(ph, height=0.01, distance=int(fs))
        out["eda_scr_rate"] = float(pk.size / (x.size / fs) * 60)
        out["eda_scr_amp"] = float(np.mean(props["peak_heights"])) if pk.size else 0.0
    except Exception:  # noqa: BLE001
        out["eda_scr_rate"] = np.nan
        out["eda_scr_amp"] = np.nan
    return out


def emg_features(emg: np.ndarray, fs: float) -> dict:
    x = emg.astype(float)
    return {
        "emg_rms": float(np.sqrt(np.mean(x ** 2))),
        "emg_mae": float(np.mean(np.abs(x))),
        "emg_std": float(np.std(x)),
        "emg_p90": float(np.percentile(np.abs(x), 90)),
    }


def resp_features(resp: np.ndarray, fs: float) -> dict:
    x = resp.astype(float)
    out = {"resp_rate": np.nan, "resp_rate_std": np.nan,
           "resp_amp": np.nan, "resp_sig_std": float(np.std(x))}
    try:
        sos = sps.butter(2, [0.1, 0.5], "bp", fs=fs, output="sos")
        xf = sps.sosfiltfilt(sos, x)
        pk, _ = sps.find_peaks(xf, distance=int(fs * 1.5))
        if pk.size >= 2:
            ibi = np.diff(pk) / fs
            out["resp_rate"] = float(60.0 / np.mean(ibi))
            out["resp_rate_std"] = float(np.std(60.0 / ibi))
            out["resp_amp"] = float(np.mean(xf[pk]))
    except Exception:  # noqa: BLE001
        pass
    return out


def temp_features(temp: np.ndarray, fs: float) -> dict:
    x = temp.astype(float)
    t = np.arange(x.size) / fs
    out = {"temp_mean": float(np.mean(x)), "temp_std": float(np.std(x)),
           "temp_min": float(np.min(x)), "temp_max": float(np.max(x))}
    try:
        out["temp_slope"] = float(linregress(t, x).slope)
    except Exception:  # noqa: BLE001
        out["temp_slope"] = np.nan
    return out


# ---------------------------------------------------------------------------
# Windowing + table build
# ---------------------------------------------------------------------------
def _window_feature_rows(sid, sig, label):
    win, step = int(WIN_SEC * FS), int(STEP_SEC * FS)
    n = label.size
    rows = []
    for s in range(0, n - win + 1, step):
        lab = label[s:s + win]
        vals, cnts = np.unique(lab, return_counts=True)
        modal = int(vals[np.argmax(cnts)])
        if modal not in LABELS or cnts.max() / win < 0.9:
            continue
        feat = {"subject": sid, "label": modal, "condition": LABELS[modal]}
        feat.update(hrv_features(sig["ECG"][s:s + win], FS))
        feat.update(eda_features(sig["EDA"][s:s + win], FS))
        feat.update(emg_features(sig["EMG"][s:s + win], FS))
        feat.update(resp_features(sig["Resp"][s:s + win], FS))
        feat.update(temp_features(sig["Temp"][s:s + win], FS))
        rows.append(feat)
    return rows


def build_features(cache: bool = True) -> pd.DataFrame:
    out_csv = config.OUTPUT_ROOT / "wesad_features.csv"
    if cache and out_csv.exists():
        return pd.read_csv(out_csv)
    rows = []
    for sid in subjects():
        sig, label = load_subject(sid)
        srows = _window_feature_rows(sid, sig, label)
        rows.extend(srows)
        print(f"  {sid}: +{len(srows)} windows (total {len(rows)})", flush=True)
    df = pd.DataFrame(rows)
    if cache:
        df.to_csv(out_csv, index=False)
        print(f"WESAD features -> {out_csv}")
    return df


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    d = build_features()
    print("\nWindows per condition:")
    print(d["condition"].value_counts().to_string())
    print("\nWindows per subject:")
    print(d["subject"].value_counts().sort_index().to_string())
