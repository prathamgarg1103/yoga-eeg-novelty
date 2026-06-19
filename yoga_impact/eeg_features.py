"""Device-agnostic EEG feature extraction (2 frontal channels: left, right).

Works identically on Emotiv F7/F8, Muse AF7/AF8, and BioSemi frontal pairs so the
same features describe the custom yoga data and the ds001787 meditation data.

Pipeline per recording:
  bandpass 1-45 Hz + 50 Hz notch -> 8 s windows (50 % overlap) -> per-window features:
    * absolute + RELATIVE band power (delta/theta/alpha/beta/gamma) per channel,
    * band ratios (alpha/theta, theta/beta, alpha/beta),
    * spectral entropy, Hjorth (activity/mobility/complexity),
    * frontal alpha/theta asymmetry (ln right - ln left).

Relative power + ratios + entropy + Hjorth mobility/complexity are amplitude-scale
invariant, which is what lets features transfer across devices with different gains.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sps

from yoga_impact import config

META = ("subject", "state", "recording_id", "label", "device")


def _preprocess(x: np.ndarray, fs: float) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float))
    lo, hi = config.EEG_BANDPASS
    hi = min(hi, fs / 2 - 1)
    sos = sps.butter(4, [lo, hi], "bp", fs=fs, output="sos")
    y = sps.sosfiltfilt(sos, x)
    if fs / 2 > config.NOTCH_FREQ:
        b, a = sps.iirnotch(config.NOTCH_FREQ, 30, fs)
        y = sps.filtfilt(b, a, y)
    return y


def _psd(x: np.ndarray, fs: float):
    nper = int(min(x.size, fs * 2))
    f, pxx = sps.welch(x, fs=fs, nperseg=nper)
    return f, pxx


def _band_powers(f, pxx) -> dict:
    out = {}
    band_mask = (f >= 1) & (f <= 45)
    total = np.trapezoid(pxx[band_mask], f[band_mask]) + 1e-12
    for name, (lo, hi) in config.BANDS.items():
        idx = (f >= lo) & (f < hi)
        bp = float(np.trapezoid(pxx[idx], f[idx]))
        out[f"{name}_abs"] = bp
        out[f"{name}_rel"] = bp / total
    return out


def _spectral_entropy(f, pxx) -> float:
    p = pxx[(f >= 1) & (f <= 45)]
    p = p / (p.sum() + 1e-12)
    return float(-np.sum(p * np.log2(p + 1e-12)) / np.log2(p.size))


def _hjorth(x: np.ndarray):
    dx = np.diff(x)
    ddx = np.diff(dx)
    v0 = np.var(x) + 1e-12
    v1 = np.var(dx) + 1e-12
    v2 = np.var(ddx) + 1e-12
    mobility = np.sqrt(v1 / v0)
    complexity = (np.sqrt(v2 / v1) / mobility) if mobility > 0 else 0.0
    return float(v0), float(mobility), float(complexity)


def _channel_features(x: np.ndarray, fs: float, prefix: str):
    f, pxx = _psd(x, fs)
    bp = _band_powers(f, pxx)
    feat = {f"{prefix}_{k}": v for k, v in bp.items()}
    a, th, be = bp["alpha_abs"], bp["theta_abs"], bp["beta_abs"]
    feat[f"{prefix}_alpha_theta"] = a / (th + 1e-12)
    feat[f"{prefix}_theta_beta"] = th / (be + 1e-12)
    feat[f"{prefix}_alpha_beta"] = a / (be + 1e-12)
    feat[f"{prefix}_spec_ent"] = _spectral_entropy(f, pxx)
    act, mob, comp = _hjorth(x)
    feat[f"{prefix}_hjorth_act"] = act
    feat[f"{prefix}_hjorth_mob"] = mob
    feat[f"{prefix}_hjorth_comp"] = comp
    return feat, bp


def window_features(seg: np.ndarray, fs: float) -> dict:
    left = _preprocess(seg[0], fs)
    right = _preprocess(seg[1], fs)
    fL, bpL = _channel_features(left, fs, "L")
    fR, bpR = _channel_features(right, fs, "R")
    feat = {**fL, **fR}
    feat["alpha_asym"] = float(np.log(bpR["alpha_abs"] + 1e-12) - np.log(bpL["alpha_abs"] + 1e-12))
    feat["theta_asym"] = float(np.log(bpR["theta_abs"] + 1e-12) - np.log(bpL["theta_abs"] + 1e-12))
    feat["mean_alpha_rel"] = (bpL["alpha_rel"] + bpR["alpha_rel"]) / 2
    feat["mean_theta_rel"] = (bpL["theta_rel"] + bpR["theta_rel"]) / 2
    feat["mean_beta_rel"] = (bpL["beta_rel"] + bpR["beta_rel"]) / 2
    feat["mean_alpha_theta"] = (feat["L_alpha_theta"] + feat["R_alpha_theta"]) / 2
    return feat


def extract_recording(rec) -> list[dict]:
    fs = rec.fs
    win = int(config.WIN_SEC * fs)
    step = max(1, int(win * (1 - config.WIN_OVERLAP)))
    data = rec.data
    rows = []
    for s in range(0, data.shape[1] - win + 1, step):
        seg = data[:, s:s + win]
        try:
            feat = window_features(seg, fs)
        except Exception:  # noqa: BLE001
            continue
        feat.update(
            subject=rec.subject, state=rec.state, recording_id=rec.recording_id,
            label=int(rec.state == config.POSITIVE_CLASS), device=rec.device,
        )
        rows.append(feat)
    return rows


def build_feature_matrix(recordings) -> pd.DataFrame:
    rows = []
    for rec in recordings:
        rows.extend(extract_recording(rec))
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META]
