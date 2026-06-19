"""ds001787 meditation EEG — external validation of the EEG relaxation axis.

BioSemi-64 uses A/B electrode labels; the standard layout maps A7->F7 and B10->F8,
so we extract exactly the F7/F8 pair used by the custom Emotiv data, resample 256->128 Hz,
and run the SAME feature pipeline. This lets us check that frontal relaxation signatures
generalize to an independent device, cohort, and practice (seated meditation).

Analyses:
  1. expert vs novice meditator classification (LOSO) from frontal EEG features,
  2. relaxation-feature contrasts (alpha / alpha-theta) expert vs novice,
  3. transfer: apply the custom-trained relaxation model to this data and compare the
     Relaxation Index of experts vs novices.

Features cached to outputs/ds_meditation_features.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import mne

from yoga_impact import config
from yoga_impact.io_custom import Recording
from yoga_impact.eeg_features import build_feature_matrix, feature_columns
from yoga_impact.modeling import group_cv_evaluate, make_pipeline
from sklearn.linear_model import LogisticRegression

# BioSemi-64 standard layout -> 10-20 (the frontal pair we need)
BIOSEMI_F7 = "A7"
BIOSEMI_F8 = "B10"


def _participants() -> pd.DataFrame:
    return pd.read_csv(config.MEDITATION_ROOT / "participants.tsv", sep="\t")


def _first_bdf(sub: str):
    cands = sorted((config.MEDITATION_ROOT / sub).rglob("*_task-meditation_eeg.bdf"))
    return cands[0] if cands else None


def load_recordings(resample: float = 128.0, crop_sec: float | None = 1200.0):
    """Load F7/F8 (A7/B10) per subject, resampled to 128 Hz, as Recording objects."""
    parts = _participants()
    recs = []
    for _, row in parts.iterrows():
        sub, grp = row["participant_id"], row["group"]
        bdf = _first_bdf(sub)
        if bdf is None:
            continue
        raw = mne.io.read_raw_bdf(bdf, preload=False, verbose="ERROR")
        if BIOSEMI_F7 not in raw.ch_names or BIOSEMI_F8 not in raw.ch_names:
            print(f"  [skip] {sub}: A7/B10 missing")
            continue
        raw.pick([BIOSEMI_F7, BIOSEMI_F8])
        if crop_sec:
            raw.crop(tmax=min(crop_sec, raw.times[-1]))
        raw.load_data(verbose="ERROR")
        if resample and raw.info["sfreq"] != resample:
            raw.resample(resample, verbose="ERROR")
        data = raw.get_data() * 1e6  # V -> microvolts
        recs.append(Recording(
            subject=sub, state=grp, device="BioSemi", path=bdf,
            recording_id=sub, fs=float(raw.info["sfreq"]),
            channels=["F7", "F8"], data=data.astype(np.float32),
        ))
        print(f"  {sub} ({grp}): {data.shape[1]} samples @ {raw.info['sfreq']:.0f} Hz", flush=True)
    return recs


def build_features(cache: bool = True) -> pd.DataFrame:
    out_csv = config.OUTPUT_ROOT / "ds_meditation_features.csv"
    if cache and out_csv.exists():
        return pd.read_csv(out_csv)
    recs = load_recordings()
    df = build_feature_matrix(recs)
    # 'label' from eeg_features is meaningless here (state != Relaxed); recompute
    df["label"] = (df["state"] == "expert").astype(int)   # 1 = expert
    if cache:
        df.to_csv(out_csv, index=False)
        print(f"ds features -> {out_csv}")
    return df


def run() -> dict:
    df = build_features(cache=True)
    feats = feature_columns(df)
    X = df[feats].to_numpy(float)
    y = (df["state"] == "expert").astype(int).to_numpy()
    groups = df["subject"].to_numpy()

    print("=" * 66)
    print("ds001787 MEDITATION EEG — external validation (F7/F8 = A7/B10)")
    print("=" * 66)
    print(f"subjects={df['subject'].nunique()}  windows={len(df)}  features={len(feats)}")
    print("windows per group:", df["state"].value_counts().to_dict())

    results: dict[str, dict] = {}

    # 1) expert vs novice (LOSO)
    res = group_cv_evaluate(X, y, groups, lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", C=0.5), calibrate="sigmoid")
    print("\n[1] expert vs novice (LOSO, LogReg)")
    print("  " + res.summary().replace("\n", "\n  "))
    results["expert_vs_novice"] = res.pooled

    # 2) relaxation-feature contrasts
    print("\n[2] relaxation-feature contrasts (mean over windows)")
    contrast_cols = ["mean_alpha_rel", "mean_theta_rel", "mean_alpha_theta",
                     "alpha_asym", "L_alpha_rel", "R_alpha_rel"]
    contrast = df.groupby("state")[contrast_cols].mean()
    print(contrast.round(3).to_string())
    results["feature_contrasts"] = contrast.round(4).to_dict()

    # 3) transfer: custom-trained relaxation model applied to ds001787
    print("\n[3] transfer: custom relaxation model -> ds001787 Relaxation Index")
    try:
        from yoga_impact.clean_custom import load_clean_recordings
        cdf = build_feature_matrix(load_clean_recordings())
        common = [c for c in feature_columns(cdf) if c in feats]
        model = make_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
        model.fit(cdf[common].to_numpy(float), cdf["label"].to_numpy(int))
        ri = model.predict_proba(df[common].to_numpy(float))[:, 1] * 100
        df_ri = df[["subject", "state"]].copy()
        df_ri["relaxation_index"] = ri
        by_group = df_ri.groupby("state")["relaxation_index"].mean().round(1)
        print("  Relaxation Index (custom model) by group:")
        print("  " + by_group.to_string().replace("\n", "\n  "))
        results["transfer_relaxation_index"] = by_group.to_dict()
    except Exception as exc:  # noqa: BLE001
        print(f"  [transfer skipped] {exc}")

    out = config.OUTPUT_ROOT / "ds_meditation_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nMetrics -> {out}")
    print("=" * 66)
    return results


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    run()
