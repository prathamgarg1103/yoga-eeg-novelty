"""C4 — leak-free few-shot subject adaptation.

The 2023 journal claimed "subject-specific adaptability" but obtained it by fine-tuning /
normalising on the subject's own LATER-SCORED task windows — a leak. Here we deliver the
same idea honestly: adapt to a subject using ONLY a disjoint slice of that subject's own
*baseline* windows, then score DIFFERENT windows.

Testbed = WESAD (15 subjects). Task = baseline vs amusement — the one WESAD condition pair
with real headroom (subject-independent AUROC ~0.73; every other pair saturates near 1.0 on
chest autonomics, leaving nothing to personalize). Per subject t (LOSO):
  * adaptation pool = first fraction of t's MEDITATION (label 4) windows — a within-subject
    physiological anchor that is DISJOINT from both the scored and trained classes, so it
    cannot leak,
  * scoring set     = t's baseline (1) vs amusement (3) windows — DISJOINT from the anchor,
  * compare a subject-independent model (raw features) against a subject-adaptive one
    (features re-centred / z-scored by each subject's own anchor statistics).

The adaptation slice never overlaps the scored windows, and the global model never trains
on t — so the lift (if any) is an honest measure of personalization value. A null lift is
reported as null (that itself rebuts the inflated 2023 personalization claim).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

from yoga_impact import config
from yoga_impact.modeling import make_pipeline


def _rf(seed: int = 42):
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                  class_weight="balanced", n_jobs=-1, random_state=seed)


def _subject_baseline_stats(X, subj, label, frac, min_win, ref_label):
    """Per-subject (mean, std) from the first `frac` of that subject's anchor windows."""
    stats = {}
    for s in np.unique(subj):
        bwin = X[(subj == s) & (label == ref_label)]
        if len(bwin) == 0:
            stats[s] = (np.zeros(X.shape[1]), np.ones(X.shape[1]), 0)
            continue
        k = max(min_win, int(round(frac * len(bwin))))
        k = min(k, len(bwin))
        adapt = bwin[:k]
        mu = np.nanmean(adapt, axis=0)
        sd = np.nanstd(adapt, axis=0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        stats[s] = (mu, sd, k)
    return stats


def _normalize(X, subj, stats, mode):
    Xn = X.astype(float).copy()
    for s in np.unique(subj):
        m = subj == s
        mu, sd, _ = stats[s]
        if mode == "recenter":
            Xn[m] = Xn[m] - mu
        elif mode == "zscore":
            Xn[m] = (Xn[m] - mu) / sd
    return Xn


def run() -> dict:
    from yoga_impact.wesad import build_features
    df = build_features(cache=True)
    feats = [c for c in df.columns if c not in {"subject", "label", "condition"}]
    X = df[feats].to_numpy(float)
    subj = df["subject"].to_numpy()
    label = df["label"].to_numpy(int)

    task = list(config.PERSON_TASK_LABELS)
    pos = config.PERSON_POS_LABEL
    ref = config.PERSON_REF_LABEL
    print("=" * 66)
    print("LEAK-FREE PERSONALIZATION - WESAD (baseline vs amusement, LOSO)")
    print("=" * 66)

    stats = _subject_baseline_stats(X, subj, label, config.PERSON_ADAPT_FRACTION,
                                    config.PERSON_MIN_ADAPT_WINDOWS, ref)
    train_mask = np.isin(label, task)
    y_calm = (label == pos).astype(int)                # positive class = amusement
    methods = [m for m in config.PERSON_METHODS if m in ("recenter", "zscore")]

    Xnorm = {m: _normalize(X, subj, stats, m) for m in methods}
    logo = LeaveOneGroupOut()
    per_subject = []

    for tr, te in logo.split(X, y_calm, subj):
        t = subj[te][0]
        tr_use = tr[train_mask[tr]]
        if len(np.unique(y_calm[tr_use])) < 2:
            continue
        # scoring set: t's baseline (1) vs amusement (3) — disjoint from the meditation anchor
        score_idx = te[np.isin(label[te], task)]
        ys = (label[score_idx] == pos).astype(int)      # 1 = amusement, 0 = baseline
        if len(np.unique(ys)) < 2:
            continue

        row = {"subject": str(t), "n_score": int(len(score_idx)),
               "n_adapt": int(stats[t][2])}

        # subject-independent baseline (raw features)
        m_ind = make_pipeline(_rf()).fit(X[tr_use], y_calm[tr_use])
        p_ind = m_ind.predict_proba(X[score_idx])[:, 1]
        row["auroc_independent"] = float(roc_auc_score(ys, p_ind))
        row["balacc_independent"] = float(balanced_accuracy_score(ys, (p_ind >= 0.5).astype(int)))

        # subject-adaptive (per-subject baseline normalization)
        for mth in methods:
            Xn = Xnorm[mth]
            m_ad = make_pipeline(_rf()).fit(Xn[tr_use], y_calm[tr_use])
            p_ad = m_ad.predict_proba(Xn[score_idx])[:, 1]
            row[f"auroc_{mth}"] = float(roc_auc_score(ys, p_ad))
        per_subject.append(row)

    pdf = pd.DataFrame(per_subject)
    results = {"per_subject": per_subject, "methods": methods}
    print(f"\nsubjects evaluated: {len(pdf)}")
    print(f"  subject-independent  mean AUROC = {pdf['auroc_independent'].mean():.3f}")
    for mth in methods:
        lift = pdf[f"auroc_{mth}"] - pdf["auroc_independent"]
        helped = int((lift > 0).sum())
        hurt = int((lift < 0).sum())
        try:
            wstat, wp = wilcoxon(pdf[f"auroc_{mth}"], pdf["auroc_independent"])
            wp = float(wp)
        except ValueError:
            wp = float("nan")
        results[mth] = dict(mean_auroc=float(pdf[f"auroc_{mth}"].mean()),
                            mean_lift=float(lift.mean()), helped=helped, hurt=hurt,
                            wilcoxon_p=wp)
        print(f"  adapted [{mth:8s}] mean AUROC = {pdf[f'auroc_{mth}'].mean():.3f}  "
              f"mean lift = {lift.mean():+.3f}  (helped {helped} / hurt {hurt}, Wilcoxon p={wp:.3f})")

    results["independent_mean_auroc"] = float(pdf["auroc_independent"].mean())
    out = config.OUTPUT_ROOT / "personalization_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    pdf.to_csv(config.OUTPUT_ROOT / "personalization_per_subject.csv", index=False)
    try:
        _plot(pdf, methods)
    except Exception as exc:  # noqa: BLE001
        print(f"[plot skipped] {exc}")
    print(f"\nMetrics -> {out}")
    print("=" * 66)
    return results


def _plot(pdf, methods) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = max(methods, key=lambda m: (pdf[f"auroc_{m}"] - pdf["auroc_independent"]).mean()) \
        if methods else None
    if best is None:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    a = pdf["auroc_independent"].to_numpy()
    b = pdf[f"auroc_{best}"].to_numpy()
    for i in range(len(a)):
        ax.plot([0, 1], [a[i], b[i]], color="grey", alpha=0.5)
    ax.scatter(np.zeros_like(a), a, label="independent")
    ax.scatter(np.ones_like(b), b, label=f"adapted ({best})")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["independent", f"adapted\n({best})"])
    ax.set_ylabel("per-subject AUROC (baseline vs amusement)")
    ax.set_title("Leak-free personalization lift (WESAD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / "personalization_lift.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    run()
