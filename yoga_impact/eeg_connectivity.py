"""C1 — connectivity / coupling / aperiodic EEG features (scipy/numpy only).

These extend the per-window feature set with quantities the 2023 journal *hypothesised*
(inter-hemispheric connectivity) but never computed, plus a modern arousal biomarker
(the 1/f aperiodic spectral exponent). Every feature here is **amplitude-scale
invariant** so it transfers across devices with different gains (Emotiv F7/F8,
BioSemi A7/B10), matching the design of ``eeg_features``:

  * coherence_pair  — inter-hemispheric magnitude-squared coherence (band-limited)
  * plv_pair        — inter-hemispheric phase-locking value (phase only)
  * pac_channel     — within-channel phase-amplitude coupling (Tort modulation index)
  * aperiodic_slope — robust log-log 1/f fit -> exponent (invariant) + offset (gain)
  * alpha_peak_freq — alpha-band spectral centre of gravity

All inputs are signals that have ALREADY been band-passed + notched by
``eeg_features._preprocess`` (we only sub-band them further here).
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps

from yoga_impact import config


def _bandpass(x: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    hi = min(hi, fs / 2 - 1)
    if hi <= lo:
        return x.astype(float)
    sos = sps.butter(4, [lo, hi], "bp", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def _welch(x: np.ndarray, fs: float):
    nper = int(min(x.size, fs * 2))
    return sps.welch(x, fs=fs, nperseg=nper)


# ---------------------------------------------------------------------------
# Inter-hemispheric connectivity
# ---------------------------------------------------------------------------
def coherence_pair(xL: np.ndarray, xR: np.ndarray, fs: float,
                   bands=config.CONN_BANDS) -> dict:
    """Magnitude-squared coherence (0-1) averaged within each band. Gain-invariant."""
    out = {f"coh_{b}": np.nan for b in bands}
    try:
        nper = int(min(xL.size, fs * 2))
        f, cxy = sps.coherence(xL, xR, fs=fs, nperseg=nper)
        for b in bands:
            lo, hi = config.BANDS[b]
            idx = (f >= lo) & (f < hi)
            if idx.any():
                out[f"coh_{b}"] = float(np.mean(cxy[idx]))
    except Exception:  # noqa: BLE001
        pass
    return out


def plv_pair(xL: np.ndarray, xR: np.ndarray, fs: float,
             bands=config.CONN_BANDS) -> dict:
    """Phase-locking value between the two channels per band. Amplitude-invariant."""
    out = {f"plv_{b}": np.nan for b in bands}
    for b in bands:
        try:
            lo, hi = config.BANDS[b]
            pl = np.angle(sps.hilbert(_bandpass(xL, fs, lo, hi)))
            pr = np.angle(sps.hilbert(_bandpass(xR, fs, lo, hi)))
            out[f"plv_{b}"] = float(np.abs(np.mean(np.exp(1j * (pl - pr)))))
        except Exception:  # noqa: BLE001
            pass
    return out


# ---------------------------------------------------------------------------
# Phase-amplitude coupling (within channel)
# ---------------------------------------------------------------------------
def _tort_mi(phase: np.ndarray, amp: np.ndarray, n_bins: int) -> float:
    """Tort modulation index: normalised KL divergence of amplitude-by-phase."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.digitize(phase, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    m = np.array([amp[idx == k].mean() if np.any(idx == k) else 0.0 for k in range(n_bins)])
    s = m.sum()
    if s <= 0:
        return np.nan
    p = m / s
    p = np.clip(p, 1e-12, None)
    h = -np.sum(p * np.log(p))
    return float((np.log(n_bins) - h) / np.log(n_bins))   # 0 = no coupling, 1 = max


def pac_channel(x: np.ndarray, fs: float,
                phase_band: str = config.PAC_PHASE_BAND,
                amp_bands=config.PAC_AMP_BANDS,
                n_bins: int = config.PAC_N_BINS) -> dict:
    """Theta-phase -> {beta,gamma}-amplitude coupling. Amplitude z-scored => gain-invariant."""
    out = {f"pac_{phase_band}_{a}": np.nan for a in amp_bands}
    try:
        plo, phi = config.BANDS[phase_band]
        phase = np.angle(sps.hilbert(_bandpass(x, fs, plo, phi)))
        for a in amp_bands:
            alo, ahi = config.BANDS[a]
            if min(ahi, fs / 2 - 1) <= alo:
                continue
            amp = np.abs(sps.hilbert(_bandpass(x, fs, alo, ahi)))
            sd = amp.std()
            if sd < 1e-12:
                continue
            amp = (amp - amp.mean()) / sd        # z-score removes amplitude gain
            out[f"pac_{phase_band}_{a}"] = _tort_mi(phase, amp - amp.min(), n_bins)
    except Exception:  # noqa: BLE001
        pass
    return out


# ---------------------------------------------------------------------------
# Aperiodic (1/f) spectral slope + alpha peak
# ---------------------------------------------------------------------------
def aperiodic_slope(f: np.ndarray, pxx: np.ndarray,
                    fit_range=config.APERIODIC_FIT_RANGE) -> dict:
    """Robust log-log 1/f fit, EXCLUDING the alpha bump, with one residual-trim pass.

    Returns ``aperiodic_exponent`` (= -slope, gain-invariant) and ``aperiodic_offset``
    (intercept, gain-dependent -> excluded from cross-device transfer set).
    """
    out = {"aperiodic_exponent": np.nan, "aperiodic_offset": np.nan}
    try:
        lo, hi = fit_range
        alo, ahi = config.BANDS["alpha"]
        mask = (f >= lo) & (f <= hi) & (pxx > 0) & ~((f >= alo) & (f < ahi))
        if mask.sum() < 5:
            return out
        lf = np.log10(f[mask])
        lp = np.log10(pxx[mask])
        slope, intercept = np.polyfit(lf, lp, 1)
        resid = lp - (slope * lf + intercept)            # one robust trim pass
        keep = np.abs(resid) <= np.quantile(np.abs(resid), 0.9)
        if keep.sum() >= 5:
            slope, intercept = np.polyfit(lf[keep], lp[keep], 1)
        out["aperiodic_exponent"] = float(-slope)
        out["aperiodic_offset"] = float(intercept)
    except Exception:  # noqa: BLE001
        pass
    return out


def alpha_peak_freq(f: np.ndarray, pxx: np.ndarray, band=(7.0, 13.0)) -> dict:
    """Alpha-band spectral centre of gravity (Hz). Gain-robust (ratio of weighted sums)."""
    out = {"alpha_peak_hz": np.nan}
    try:
        idx = (f >= band[0]) & (f <= band[1])
        p = pxx[idx]
        if p.sum() > 0:
            out["alpha_peak_hz"] = float(np.sum(f[idx] * p) / np.sum(p))
    except Exception:  # noqa: BLE001
        pass
    return out


# ---------------------------------------------------------------------------
# Orchestrator: one dict per window
# ---------------------------------------------------------------------------
def window_connectivity_features(seg: np.ndarray, fs: float) -> dict:
    """``seg`` = (2, n) already-filtered (left, right). Returns the C1 feature block."""
    left, right = seg[0], seg[1]
    feat: dict[str, float] = {}
    feat.update(coherence_pair(left, right, fs))
    feat.update(plv_pair(left, right, fs))

    # PAC + aperiodic + alpha peak: per channel, averaged across the two
    pac_keys = pac_channel(left, fs).keys()
    pacs = [pac_channel(left, fs), pac_channel(right, fs)]
    for k in pac_keys:
        vals = [d[k] for d in pacs if np.isfinite(d.get(k, np.nan))]
        feat[k] = float(np.mean(vals)) if vals else np.nan

    aps = []
    peaks = []
    for x in (left, right):
        f, pxx = _welch(x, fs)
        aps.append(aperiodic_slope(f, pxx))
        peaks.append(alpha_peak_freq(f, pxx))
    for k in ("aperiodic_exponent", "aperiodic_offset"):
        vals = [d[k] for d in aps if np.isfinite(d.get(k, np.nan))]
        feat[k] = float(np.mean(vals)) if vals else np.nan
    vals = [d["alpha_peak_hz"] for d in peaks if np.isfinite(d.get("alpha_peak_hz", np.nan))]
    feat["alpha_peak_hz"] = float(np.mean(vals)) if vals else np.nan
    return feat
