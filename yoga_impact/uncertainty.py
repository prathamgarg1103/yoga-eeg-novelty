"""C3 — uncertainty-aware Relaxation Index + cross-modal convergent validity.

Turns the 0-100 Relaxation Index into a *calibrated biomarker with a confidence band*,
and formally tests that the same relaxation axis agrees across the autonomic (WESAD) and
cortical (EEG) modalities. Everything here consumes OUT-OF-FOLD predictions only, so no
test-group information leaks:

  * split_conformal_intervals — leave-group-out (Mondrian) conformal prediction: the
    quantile for a held-out group is taken over the OTHER groups' nonconformity scores.
  * expected_calibration_error / reliability_curve — is the calibrated probability honest?
  * convergent_validity — Spearman agreement between the expected relaxation ordering of
    conditions and the model's observed mean index, pooled across modalities.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LogisticRegression

from yoga_impact import config
from yoga_impact.modeling import group_cv_evaluate, make_pipeline, default_rf


# ---------------------------------------------------------------------------
# Conformal prediction (leave-group-out / Mondrian)
# ---------------------------------------------------------------------------
def split_conformal_intervals(prob, y, groups, alpha: float = config.CONFORMAL_ALPHA) -> dict:
    """Per-sample prediction band on P(relaxed). qhat for group g uses only groups != g."""
    prob = np.asarray(prob, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    s = np.abs(y - prob)                       # nonconformity scores
    lo = np.empty_like(prob)
    hi = np.empty_like(prob)
    q = np.empty_like(prob)
    for g in np.unique(groups):
        m = groups == g
        cal = s[~m]
        if cal.size == 0:
            qhat = 1.0
        else:
            n = cal.size
            level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
            qhat = float(np.quantile(cal, level, method="higher"))
        q[m] = qhat
        lo[m] = np.clip(prob[m] - qhat, 0, 1)
        hi[m] = np.clip(prob[m] + qhat, 0, 1)
    covered = (np.abs(y - prob) <= q).astype(int)
    return dict(lo=lo, hi=hi, qhat=q, covered=covered,
                empirical_coverage=float(covered.mean()), nominal=1 - alpha)


def expected_calibration_error(prob, y, n_bins: int = config.ECE_BINS) -> dict:
    prob = np.asarray(prob, float)
    y = np.asarray(y, int)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    mce = 0.0
    table = []
    N = len(prob)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc, wt = prob[m].mean(), y[m].mean(), m.sum() / N
        gap = abs(acc - conf)
        ece += wt * gap
        mce = max(mce, gap)
        table.append(dict(conf=float(conf), acc=float(acc), n=int(m.sum())))
    return dict(ece=float(ece), mce=float(mce), bins=table)


def reliability_curve(prob, y, n_bins: int = config.ECE_BINS):
    prob = np.asarray(prob, float)
    y = np.asarray(y, int)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    centers, frac, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        centers.append(prob[m].mean())
        frac.append(y[m].mean())
        counts.append(int(m.sum()))
    return np.array(centers), np.array(frac), np.array(counts)


def convergent_validity(conditions) -> dict:
    """conditions: list of (name, expected_rank, observed_index). Spearman + Kendall."""
    exp = [c[1] for c in conditions]
    obs = [c[2] for c in conditions]
    rho, p = spearmanr(exp, obs)
    tau, pt = kendalltau(exp, obs)
    return dict(spearman_rho=float(rho), spearman_p=float(p),
                kendall_tau=float(tau), kendall_p=float(pt),
                conditions=[dict(name=c[0], expected_rank=c[1], observed_index=round(float(c[2]), 1))
                            for c in conditions])


# ---------------------------------------------------------------------------
# Per-stage OOF reconstruction (self-contained; same leak-free CV as the models)
# ---------------------------------------------------------------------------
def _custom_oof():
    from yoga_impact.clean_custom import load_clean_recordings
    from yoga_impact.eeg_features import build_feature_matrix, feature_columns
    df = build_feature_matrix(load_clean_recordings())
    feats = feature_columns(df)
    res = group_cv_evaluate(df[feats].to_numpy(float), df["label"].to_numpy(int),
                            df["recording_id"].to_numpy(),
                            lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
                            calibrate="sigmoid")
    return df, res


def _wesad_calm_oof():
    from yoga_impact.wesad import build_features
    df = build_features(cache=True)
    feats = [c for c in df.columns if c not in {"subject", "label", "condition"}]
    mask = df["label"].isin([1, 2, 4]).to_numpy()
    res = group_cv_evaluate(df[feats].to_numpy(float)[mask],
                            df.loc[mask, "label"].isin([1, 4]).astype(int).to_numpy(),
                            df["subject"].to_numpy()[mask],
                            lambda: default_rf(), calibrate="sigmoid")
    return df, mask, res, feats


def _wesad_condition_scores(df, feats) -> dict:
    """LOSO: train calm(1,4)-vs-stress(2), score ALL of held-out subject's conditions.

    Gives an OOF relaxation score for amusement (label 3) too, so all four conditions
    can be ranked. Training labels are unchanged; amusement is never in training.
    """
    from sklearn.model_selection import LeaveOneGroupOut
    X = df[feats].to_numpy(float)
    cond = df["condition"].to_numpy()
    subj = df["subject"].to_numpy()
    train_mask = df["label"].isin([1, 2, 4]).to_numpy()
    y_calm = df["label"].isin([1, 4]).astype(int).to_numpy()
    score = np.full(len(df), np.nan)
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, np.zeros(len(X)), subj):
        tr_use = tr[train_mask[tr]]
        if len(np.unique(y_calm[tr_use])) < 2:
            continue
        model = make_pipeline(default_rf()).fit(X[tr_use], y_calm[tr_use])
        score[te] = model.predict_proba(X[te])[:, 1] * 100.0
    s = pd.Series(score)
    return s.groupby(cond).mean().dropna().round(2).to_dict()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run() -> dict:
    print("=" * 66)
    print("UNCERTAINTY-AWARE INDEX + CROSS-MODAL CONVERGENT VALIDITY")
    print("=" * 66)
    results: dict[str, dict] = {}

    # ---- Custom EEG (illustrative; ~4 subjects) ----------------------------
    cdf, cres = _custom_oof()
    # recording-level aggregation
    tmp = cdf[["recording_id", "subject", "state", "label"]].copy()
    tmp["prob"] = cres.oof_prob
    tmp = tmp.dropna(subset=["prob"])
    agg = (tmp.groupby("recording_id")
           .agg(subject=("subject", "first"), state=("state", "first"),
                label=("label", "first"), prob=("prob", "mean")).reset_index())
    conf = split_conformal_intervals(agg["prob"].to_numpy(), agg["label"].to_numpy(),
                                     agg["subject"].to_numpy())
    agg["relaxation_index"] = (agg["prob"] * 100).round(1)
    agg["index_lo"] = (conf["lo"] * 100).round(1)
    agg["index_hi"] = (conf["hi"] * 100).round(1)
    agg["covered"] = conf["covered"]
    ece_c = expected_calibration_error(cres.oof_prob[~np.isnan(cres.oof_prob)],
                                       cres.y[~np.isnan(cres.oof_prob)])
    results["custom"] = dict(ece=ece_c["ece"], mce=ece_c["mce"],
                             conformal_coverage=conf["empirical_coverage"],
                             nominal=conf["nominal"])
    agg.sort_values("relaxation_index", ascending=False).to_csv(
        config.OUTPUT_ROOT / "custom_relaxation_index.csv", index=False)
    print(f"\n[custom EEG] ECE={ece_c['ece']:.3f}  conformal coverage="
          f"{conf['empirical_coverage']:.2f} (nominal {conf['nominal']:.2f}, ~4 subjects -> illustrative)")

    # ---- WESAD (credible; 15 subjects) -------------------------------------
    wdf, wmask, wres, wfeats = _wesad_calm_oof()
    wsub = wdf["subject"].to_numpy()[wmask]
    wconf = split_conformal_intervals(wres.oof_prob, wres.y, wsub)
    ece_w = expected_calibration_error(wres.oof_prob, wres.y)
    results["wesad"] = dict(ece=ece_w["ece"], mce=ece_w["mce"],
                            conformal_coverage=wconf["empirical_coverage"],
                            nominal=wconf["nominal"])
    print(f"[WESAD] ECE={ece_w['ece']:.3f}  conformal coverage="
          f"{wconf['empirical_coverage']:.2f} (nominal {wconf['nominal']:.2f})")

    # ---- Cross-modal convergent validity -----------------------------------
    cond_scores = _wesad_condition_scores(wdf, wfeats)
    expected = {"meditation": 4, "baseline": 3, "amusement": 2, "stress": 1}
    auto_rows = [(f"WESAD:{k}", expected[k], cond_scores[k]) for k in expected if k in cond_scores]
    cv_auto = convergent_validity(auto_rows)
    # EEG cortical ordering (Relaxed > Concentrated)
    eeg_rel = agg.groupby("state")["relaxation_index"].mean()
    eeg_rows = [("EEG:Concentrated", 1, eeg_rel.get("Concentrated", np.nan)),
                ("EEG:Relaxed", 2, eeg_rel.get("Relaxed", np.nan))]
    # pooled cross-modal: rank all conditions on the shared 0-100 relaxation axis
    pooled = auto_rows + [("EEG:Concentrated", 1.5, eeg_rel.get("Concentrated", np.nan)),
                          ("EEG:Relaxed", 4.5, eeg_rel.get("Relaxed", np.nan))]
    pooled = [r for r in pooled if np.isfinite(r[2])]
    cv_pooled = convergent_validity(pooled)
    results["convergent_validity"] = dict(autonomic=cv_auto, pooled_cross_modal=cv_pooled,
                                          eeg_ordering={k: round(float(v), 1) for k, v in eeg_rel.items()})
    print(f"\n[convergent validity] autonomic Spearman rho={cv_auto['spearman_rho']:.3f} "
          f"(p={cv_auto['spearman_p']:.3f}); pooled cross-modal rho={cv_pooled['spearman_rho']:.3f}")
    print("  WESAD condition scores:", {k: round(v, 1) for k, v in cond_scores.items()})
    print("  EEG ordering:", {k: round(float(v), 1) for k, v in eeg_rel.items()})

    out = config.OUTPUT_ROOT / "uncertainty_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    try:
        _plot_reliability(cres.oof_prob[~np.isnan(cres.oof_prob)], cres.y[~np.isnan(cres.oof_prob)],
                          "reliability_custom.png", "Custom EEG")
        _plot_reliability(wres.oof_prob, wres.y, "reliability_wesad.png", "WESAD")
    except Exception as exc:  # noqa: BLE001
        print(f"[plot skipped] {exc}")
    print(f"\nMetrics -> {out}")
    print("=" * 66)
    return results


def _plot_reliability(prob, y, fname, title) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centers, frac, counts = reliability_curve(prob, y)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect")
    ax.plot(centers, frac, "o-", label="model")
    ax.set_xlabel("predicted P(relaxed)")
    ax.set_ylabel("empirical fraction relaxed")
    ax.set_title(f"Reliability — {title}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / fname, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    run()
