"""Custom yoga EEG relaxation model -> Relaxation Index (0-100).

Trains a calibrated Relaxed-vs-Concentrated classifier on the 19 clean recordings.
Because windows within a recording are correlated, the honest unit of analysis is the
RECORDING: window-level OOF probabilities are aggregated per recording.

Two leak-free validations:
  * LORO  - leave-one-RECORDING-out  (19 folds; primary, given only ~4 subjects)
  * LOPO  - leave-one-SUBJECT-out    (stricter generalization to unseen people)

Relaxation Index = calibrated P(Relaxed) x 100.

NOTE: this dataset is tiny (19 recordings, effectively 3-4 usable subjects). Results
are reported as an honest case study, not a headline accuracy claim.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from yoga_impact import config
from yoga_impact.clean_custom import load_clean_recordings
from yoga_impact.eeg_features import build_feature_matrix, feature_columns
from yoga_impact.modeling import default_rf, group_cv_evaluate, permutation_test


def _logit():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)


def _recording_level(df, oof_prob):
    """Aggregate window OOF probs -> one prediction per recording."""
    tmp = df[["recording_id", "subject", "state", "label"]].copy()
    tmp["prob"] = oof_prob
    tmp = tmp.dropna(subset=["prob"])
    agg = (tmp.groupby("recording_id")
           .agg(subject=("subject", "first"), state=("state", "first"),
                label=("label", "first"), prob=("prob", "mean"))
           .reset_index())
    pred = (agg["prob"] >= 0.5).astype(int)
    acc = float(accuracy_score(agg["label"], pred))
    auroc = (float(roc_auc_score(agg["label"], agg["prob"]))
             if agg["label"].nunique() == 2 else float("nan"))
    return agg, acc, auroc


def run() -> dict:
    recs = load_clean_recordings()
    df = build_feature_matrix(recs)
    feats = feature_columns(df)
    X = df[feats].to_numpy(float)
    y = df["label"].to_numpy(int)

    print("=" * 66)
    print("CUSTOM YOGA EEG — Relaxed vs Concentrated (Relaxation Index)")
    print("=" * 66)
    print(f"recordings={df['recording_id'].nunique()}  windows={len(df)}  "
          f"features={len(feats)}  subjects={df['subject'].nunique()}")
    print("windows per state:", df["state"].value_counts().to_dict())

    results: dict[str, dict] = {}
    models = [("LogReg", _logit), ("RandomForest", lambda: default_rf())]

    for cv_name, group_col in [("LORO(recording)", "recording_id"),
                               ("LOPO(subject)", "subject")]:
        groups = df[group_col].to_numpy()
        print(f"\n--- {cv_name} ---")
        for mname, fac in models:
            res = group_cv_evaluate(X, y, groups, fac, calibrate="sigmoid")
            agg, racc, rauroc = _recording_level(df, res.oof_prob)
            print(f"  {mname:13s} window: {res.summary().splitlines()[0]}")
            print(f"  {'':13s} RECORDING-level: acc={racc:.3f}  auroc={rauroc:.3f}  "
                  f"(n={len(agg)} recordings)")
            results[f"{cv_name}::{mname}"] = dict(
                window=res.pooled, recording=dict(acc=racc, auroc=rauroc, n=len(agg)))

    # permutation test (recording-grouped) using the linear model
    perm = permutation_test(X, y, df["recording_id"].to_numpy(), _logit, n_perm=200)
    print(f"\npermutation test (LORO, LogReg): observed AUROC={perm['observed_auroc']:.3f}  "
          f"perm-mean={perm['perm_mean']:.3f}  p={perm['perm_p']:.3f}")
    results["permutation_LORO"] = perm

    # ---- Relaxation Index per recording (LORO LogReg, calibrated) -----------
    groups = df["recording_id"].to_numpy()
    res = group_cv_evaluate(X, y, groups, _logit, calibrate="sigmoid")
    agg, racc, rauroc = _recording_level(df, res.oof_prob)
    agg["relaxation_index"] = (agg["prob"] * 100).round(1)
    agg = agg.sort_values("relaxation_index", ascending=False)
    out_scores = config.OUTPUT_ROOT / "custom_relaxation_index.csv"
    agg.to_csv(out_scores, index=False)

    print("\nRelaxation Index by true state (mean, 0-100):")
    print(agg.groupby("state")["relaxation_index"].mean().round(1).to_string())
    print("\nPer-recording Relaxation Index:")
    print(agg[["subject", "state", "relaxation_index"]].to_string(index=False))

    out = config.OUTPUT_ROOT / "custom_eeg_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    try:
        _plot(agg)
    except Exception as exc:  # noqa: BLE001
        print(f"[plot skipped] {exc}")
    print(f"\nMetrics -> {out}\nScores  -> {out_scores}")
    print("=" * 66)
    return results


def _plot(agg: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [agg.loc[agg["state"] == s, "relaxation_index"] for s in ("Concentrated", "Relaxed")]
    ax.boxplot(data, tick_labels=["Concentrated", "Relaxed"])
    for i, s in enumerate(("Concentrated", "Relaxed"), 1):
        xs = np.random.default_rng(0).normal(i, 0.04, size=len(data[i - 1]))
        ax.scatter(xs, data[i - 1], alpha=0.7, s=30)
    ax.set_ylabel("Relaxation Index (0-100)")
    ax.set_title("Custom yoga EEG — Relaxation Index by state (LORO, calibrated)")
    ax.axhline(50, ls="--", c="grey", lw=1)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / "custom_relaxation_index.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    run()
