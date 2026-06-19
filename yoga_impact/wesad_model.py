"""WESAD autonomic relaxation model (LOSO).

Two targets:
  1. stress vs non-stress  -- the standard WESAD benchmark (literature comparability).
  2. calm vs stress        -- calm = {meditation, baseline}; the yoga-relevant relaxation
                              contrast. Calibrated P(calm) = the AUTONOMIC RELAXATION SCORE
                              that feeds the Composite Yoga Index.

Validation is strictly leave-one-subject-out. Saves metrics + per-condition score
ordering (a built-in sanity check: meditation should score most relaxed, stress least).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from yoga_impact import config
from yoga_impact.modeling import default_rf, group_cv_evaluate, make_pipeline, permutation_test
from yoga_impact.wesad import build_features

META_COLS = {"subject", "label", "condition"}


def _light_rf(seed: int = 42):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=120, min_samples_leaf=3, class_weight="balanced",
        n_jobs=-1, random_state=seed,
    )


def _lgbm(seed: int = 42):
    try:
        from lightgbm import LGBMClassifier
    except Exception:  # noqa: BLE001
        return None
    return LGBMClassifier(
        n_estimators=300, num_leaves=31, learning_rate=0.05,
        class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1,
    )


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def run() -> dict:
    df = build_features(cache=True)
    feats = feature_columns(df)
    X_all = df[feats].to_numpy(float)
    groups = df["subject"].to_numpy()
    results: dict[str, dict] = {}

    print("=" * 66)
    print("WESAD AUTONOMIC AXIS — leave-one-subject-out")
    print("=" * 66)
    print(f"subjects={df['subject'].nunique()}  windows={len(df)}  features={len(feats)}")
    print(df["condition"].value_counts().to_string())

    # ---- Target 1: stress vs non-stress (benchmark) -------------------------
    y_stress = (df["label"] == 2).astype(int).to_numpy()
    print("\n[1] stress vs non-stress")
    for name, fac in [("RandomForest", lambda: default_rf()),
                      ("LightGBM", lambda: _lgbm())]:
        if fac() is None:
            continue
        res = group_cv_evaluate(X_all, y_stress, groups, fac, calibrate="sigmoid")
        print(f"  {name:13s} {res.summary().splitlines()[0]}")
        results[f"stress_vs_rest::{name}"] = res.pooled

    # ---- Target 2: calm vs stress (relaxation score) ------------------------
    mask = df["label"].isin([1, 2, 4]).to_numpy()        # baseline, stress, meditation
    Xr = X_all[mask]
    gr = groups[mask]
    y_calm = df.loc[mask, "label"].isin([1, 4]).astype(int).to_numpy()   # 1 = calm
    print("\n[2] calm (meditation+baseline) vs stress  ->  relaxation score")
    res_calm = group_cv_evaluate(Xr, y_calm, gr, lambda: default_rf(), calibrate="sigmoid")
    print("  " + res_calm.summary().replace("\n", "\n  "))
    results["calm_vs_stress::RandomForest"] = res_calm.pooled

    perm = permutation_test(Xr, y_calm, gr, lambda: _light_rf(), n_perm=150)
    print(f"  permutation test: observed AUROC={perm['observed_auroc']:.3f}  "
          f"perm-mean={perm['perm_mean']:.3f}  p={perm['perm_p']:.3f}")
    results["calm_vs_stress::permutation"] = perm

    # ---- Autonomic relaxation score per condition (sanity ordering) ---------
    relax = np.full(len(df), np.nan)
    relax[mask] = res_calm.oof_prob
    df_score = df.copy()
    df_score["relaxation_score"] = relax * 100.0
    order = (df_score.dropna(subset=["relaxation_score"])
             .groupby("condition")["relaxation_score"].mean().sort_values(ascending=False))
    print("\n  Autonomic relaxation score by condition (OOF, 0-100):")
    print("  " + order.round(1).to_string().replace("\n", "\n  "))
    results["relaxation_score_by_condition"] = order.round(2).to_dict()

    # ---- feature importance (interpretation only) ---------------------------
    imp_model = make_pipeline(default_rf()).fit(Xr, y_calm)
    importances = imp_model.named_steps["clf"].feature_importances_
    top = pd.Series(importances, index=feats).sort_values(ascending=False).head(12)
    print("\n  Top autonomic features (calm vs stress):")
    print("  " + top.round(3).to_string().replace("\n", "\n  "))
    results["top_features"] = top.round(4).to_dict()

    out = config.OUTPUT_ROOT / "wesad_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    df_score[["subject", "condition", "relaxation_score"]].to_csv(
        config.OUTPUT_ROOT / "wesad_relaxation_scores.csv", index=False)
    print(f"\nMetrics -> {out}")
    print("=" * 66)
    return results


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    run()
