"""C2 — 64-channel brain-network analysis on ds001787 (expert vs novice meditators).

The rebuild's ``ds_meditation`` deliberately used only the F7/F8 (A7/B10) pair to match
the 2-channel custom Emotiv device. Here we exploit the FULL 64-channel BioSemi montage
to compute band-limited connectivity graphs and network metrics — the kind of "brain
connectivity" the 2023 journal hypothesised but never measured.

Pipeline per subject:
  read BDF (64 scalp ch) -> average reference -> crop -> resample 256->128 ->
  band-limited connectivity matrix (wPLI / PLV / coherence), averaged over windows ->
  graph metrics (efficiency, clustering, modularity, frontal strength).

One feature row per subject -> leak-free LOSO (group_cv_evaluate) + permutation test
(n=24, so the permutation p — not the point AUROC — is the real significance check).
Connectivity matrices are cached to outputs/ds_network_connectivity.npz.

Caveat reported in the output: meditation expertise is age-confounded in ds001787
(experts skew younger), so contrasts are flagged, not over-claimed.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy.stats import mannwhitneyu

import mne
from sklearn.linear_model import LogisticRegression

from yoga_impact import config
from yoga_impact.graph_metrics import graph_features
from yoga_impact.ds_meditation import _participants, _first_bdf
from yoga_impact.modeling import group_cv_evaluate, permutation_test


def _scalp_channels(ch_names) -> list[str]:
    """The 64 BioSemi scalp channels are A1..A32, B1..B32 (EXG*/Status excluded)."""
    return [c for c in ch_names if len(c) >= 2 and c[0] in "AB" and c[1:].isdigit()]


def _frontal_indices(montage_names) -> list[int]:
    return [i for i, c in enumerate(montage_names)
            if c.upper().startswith(("FP", "AF", "F"))]


# ---------------------------------------------------------------------------
# Loading (full montage)
# ---------------------------------------------------------------------------
def load_full_recordings(resample: float = 128.0,
                         crop_sec: float | None = config.NETWORK_CROP_SEC):
    parts = _participants()
    recs = []
    montage = mne.channels.make_standard_montage(config.NETWORK_MONTAGE)
    for _, row in parts.iterrows():
        sub, grp = row["participant_id"], row["group"]
        bdf = _first_bdf(sub)
        if bdf is None:
            continue
        raw = mne.io.read_raw_bdf(bdf, preload=False, verbose="ERROR")
        scalp = _scalp_channels(raw.ch_names)
        if len(scalp) < 64:
            print(f"  [skip] {sub}: only {len(scalp)} scalp channels")
            continue
        raw.pick(scalp[:64])
        # positional rename A1..A32,B1..B32 -> standard biosemi64 10-20 names
        raw.rename_channels(dict(zip(scalp[:64], montage.ch_names)))
        raw.set_montage(montage, on_missing="ignore", verbose="ERROR")
        if crop_sec:
            raw.crop(tmax=min(crop_sec, raw.times[-1]))
        raw.load_data(verbose="ERROR")
        raw.set_eeg_reference("average", verbose="ERROR")
        if resample and raw.info["sfreq"] != resample:
            raw.resample(resample, verbose="ERROR")
        data = raw.get_data() * 1e6  # V -> uV
        recs.append(dict(subject=sub, group=grp, ch_names=list(montage.ch_names),
                         fs=float(raw.info["sfreq"]), data=data.astype(np.float32)))
        print(f"  {sub} ({grp}): {data.shape} @ {raw.info['sfreq']:.0f} Hz", flush=True)
    return recs


# ---------------------------------------------------------------------------
# Connectivity matrices (vectorised over channel pairs)
# ---------------------------------------------------------------------------
def _bandpass(data: np.ndarray, fs: float, band: str) -> np.ndarray:
    lo, hi = config.BANDS[band]
    hi = min(hi, fs / 2 - 1)
    sos = sps.butter(4, [lo, hi], "bp", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, data, axis=1)


def _window_connectivity(Z: np.ndarray, method: str) -> np.ndarray:
    """Connectivity matrix for one window's analytic signal Z (C, w)."""
    C, w = Z.shape
    if method == "plv":
        P = np.exp(1j * np.angle(Z))
        M = (P @ P.conj().T) / w
        out = np.abs(M)
    elif method == "coherence":
        S = (Z @ Z.conj().T) / w                 # cross-spectral density
        d = np.real(np.diag(S))
        denom = np.sqrt(np.outer(d, d)) + 1e-24
        out = (np.abs(S) ** 2) / (denom ** 2)
    else:  # wpli (default, robust to volume conduction)
        A, B = np.real(Z), np.imag(Z)
        # Im(Z_i Z_j*) = B_i A_j - A_i B_j  over time -> (C,C,w)
        IM = (B[:, None, :] * A[None, :, :]) - (A[:, None, :] * B[None, :, :])
        num = np.abs(IM.mean(axis=2))
        den = np.abs(IM).mean(axis=2) + 1e-24
        out = num / den
    np.fill_diagonal(out, 0.0)
    return out


def subject_connectivity(rec: dict, band: str, method: str,
                         win_sec: float = config.NETWORK_WIN_SEC) -> np.ndarray:
    data = rec["data"]
    fs = rec["fs"]
    xb = _bandpass(data, fs, band)
    analytic = sps.hilbert(xb, axis=1)
    win = int(win_sec * fs)
    n = analytic.shape[1]
    mats = []
    for s in range(0, n - win + 1, win):
        mats.append(_window_connectivity(analytic[:, s:s + win], method))
    if not mats:
        mats.append(_window_connectivity(analytic, method))
    return np.mean(mats, axis=0)


# ---------------------------------------------------------------------------
# Build per-subject network feature table
# ---------------------------------------------------------------------------
def build_network_table(cache: bool = True):
    out_csv = config.OUTPUT_ROOT / "ds_network_features.csv"
    if cache and out_csv.exists():
        return pd.read_csv(out_csv), {}, None
    npz = config.OUTPUT_ROOT / "ds_network_connectivity.npz"
    bands = list(config.NETWORK_BANDS)
    method = config.NETWORK_METHOD
    recs = load_full_recordings()
    if not recs:
        return pd.DataFrame(), {}, []

    frontal = _frontal_indices(recs[0]["ch_names"])
    rows = []
    saved = {}
    for rec in recs:
        feat = {"subject": rec["subject"], "group": rec["group"],
                "label": int(rec["group"] == "expert")}
        for band in bands:
            W = subject_connectivity(rec, band, method)
            saved[f"{rec['subject']}__{band}"] = W.astype(np.float32)
            gf = graph_features(W, frontal_idx=frontal)
            for k, v in gf.items():
                feat[f"{band}_{k}"] = v
        rows.append(feat)
        print(f"  network features: {rec['subject']} done", flush=True)

    df = pd.DataFrame(rows)
    if cache:
        np.savez_compressed(npz, **saved)
        df.to_csv(config.OUTPUT_ROOT / "ds_network_features.csv", index=False)
        print(f"connectivity matrices -> {npz}")
    return df, saved, frontal


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("subject", "group", "label")]


def run() -> dict:
    print("=" * 66)
    print(f"ds001787 64-CHANNEL BRAIN-NETWORK ANALYSIS ({config.NETWORK_METHOD})")
    print("=" * 66)
    df, _, _ = build_network_table(cache=True)
    if df.empty:
        print("[no recordings - skipped]")
        return {}

    feats = feature_columns(df)
    X = df[feats].to_numpy(float)
    y = df["label"].to_numpy(int)
    groups = df["subject"].to_numpy()
    print(f"subjects={len(df)}  features={len(feats)}  "
          f"experts={int(y.sum())}  novices={int((1 - y).sum())}")

    results: dict[str, dict] = {}

    # 1) expert vs novice (LOSO) on network metrics. calibrate=None: with one row per
    # subject (n=24, LOO) probability calibration is unreliable, so we report the
    # uncalibrated discrimination AUROC (consistent with the permutation test below).
    res = group_cv_evaluate(X, y, groups, lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", C=0.5), calibrate=None)
    print("\n[1] expert vs novice (LOSO, network features)")
    print("  " + res.summary().replace("\n", "\n  "))
    results["expert_vs_novice_network"] = res.pooled

    perm = permutation_test(X, y, groups, lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", C=0.5), n_perm=500)
    print(f"  permutation test: observed AUROC={perm['observed_auroc']:.3f}  "
          f"perm-mean={perm['perm_mean']:.3f}  p={perm['perm_p']:.3f}  (n=24)")
    results["permutation"] = perm

    # 2) per-metric expert vs novice contrast (Mann-Whitney U)
    print("\n[2] expert vs novice network-metric contrasts (Mann-Whitney U)")
    contrasts = {}
    ex = df["label"] == 1
    for c in feats:
        a, b = df.loc[ex, c].dropna(), df.loc[~ex, c].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        try:
            u, p = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        contrasts[c] = dict(expert_mean=float(a.mean()), novice_mean=float(b.mean()), p=float(p))
    top = sorted(contrasts.items(), key=lambda kv: kv[1]["p"])[:8]
    for c, d in top:
        print(f"  {c:28s} expert={d['expert_mean']:.3f} novice={d['novice_mean']:.3f} p={d['p']:.3f}")
    results["metric_contrasts"] = contrasts

    results["note"] = ("Expertise is age-confounded in ds001787 (experts skew younger); "
                       "network contrasts are reported as associations, not causal expertise effects.")

    out = config.OUTPUT_ROOT / "ds_network_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    try:
        _plot(df, feats)
    except Exception as exc:  # noqa: BLE001
        print(f"[plot skipped] {exc}")
    print(f"\nMetrics -> {out}")
    print("=" * 66)
    return results


def _plot(df, feats) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # pick the alpha global-efficiency metric if present, else the first feature
    col = next((c for c in feats if c.endswith("alpha_global_efficiency")), None) \
        or next((c for c in feats if "global_efficiency" in c), feats[0])
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [df.loc[df["label"] == lab, col] for lab in (0, 1)]
    ax.boxplot(data, tick_labels=["novice", "expert"])
    for i, lab in enumerate((0, 1), 1):
        xs = np.random.default_rng(0).normal(i, 0.04, size=len(data[i - 1]))
        ax.scatter(xs, data[i - 1], alpha=0.7, s=30)
    ax.set_ylabel(col)
    ax.set_title(f"ds001787 network — {col} (expert vs novice)")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / "ds_network_efficiency.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    run()
