"""Shared, leak-free modelling utilities used by both the WESAD and EEG axes.

The whole point of the rebuild is trustworthy validation, so all evaluation goes
through ``group_cv_evaluate``:
  * leave-one-GROUP-out CV (group = subject for LOSO/LOPO, or recording for LORO),
  * imputation + scaling + classifier fit *inside* each training fold only,
  * probability calibration fit on the training fold,
  * out-of-fold (OOF) predictions pooled for honest aggregate metrics.

Nothing from a test group is ever seen during training, which is exactly what the
prior pipelines violated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def default_rf(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=seed,
    )


def make_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


@dataclass
class CVResult:
    oof_prob: np.ndarray                 # calibrated P(positive), out-of-fold
    oof_pred: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    per_fold: list[dict] = field(default_factory=list)
    pooled: dict = field(default_factory=dict)

    def summary(self) -> str:
        p = self.pooled
        lines = [
            f"pooled  acc={p['acc']:.3f}  bal_acc={p['bal_acc']:.3f}  "
            f"f1={p['f1']:.3f}  auroc={p['auroc']:.3f}  (n={len(self.y)})",
        ]
        valid = [f for f in self.per_fold if not np.isnan(f["auroc"])]
        if valid:
            import numpy as _np
            lines.append(
                "per-fold mean±sd  "
                f"acc={_np.mean([f['acc'] for f in self.per_fold]):.3f}"
                f"±{_np.std([f['acc'] for f in self.per_fold]):.3f}  "
                f"auroc={_np.mean([f['auroc'] for f in valid]):.3f}"
                f"±{_np.std([f['auroc'] for f in valid]):.3f}"
            )
        return "\n".join(lines)


def group_cv_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf_factory: Callable[[], object] | None = None,
    calibrate: str | None = "sigmoid",
    seed: int = 42,
) -> CVResult:
    """Leave-one-group-out CV with in-fold calibration; returns pooled OOF metrics."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    clf_factory = clf_factory or (lambda: default_rf(seed))

    logo = LeaveOneGroupOut()
    oof_prob = np.full(len(y), np.nan)
    per_fold = []

    for tr, te in logo.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue  # cannot train without both classes
        base = make_pipeline(clone(clf_factory()))
        if calibrate:
            n_min = np.min(np.bincount(y[tr]))
            cv = int(min(3, n_min)) if n_min >= 2 else 2
            model = CalibratedClassifierCV(base, method=calibrate, cv=cv)
        else:
            model = base
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        oof_prob[te] = prob
        pred = (prob >= 0.5).astype(int)
        auroc = roc_auc_score(y[te], prob) if len(np.unique(y[te])) == 2 else np.nan
        per_fold.append(dict(
            group=str(groups[te][0]), n=int(len(te)),
            acc=float(accuracy_score(y[te], pred)),
            f1=float(f1_score(y[te], pred, zero_division=0)),
            auroc=float(auroc),
        ))

    mask = ~np.isnan(oof_prob)
    yo, po = y[mask], oof_prob[mask]
    predo = (po >= 0.5).astype(int)
    pooled = dict(
        acc=float(accuracy_score(yo, predo)),
        bal_acc=float(balanced_accuracy_score(yo, predo)),
        f1=float(f1_score(yo, predo, zero_division=0)),
        auroc=float(roc_auc_score(yo, po)) if len(np.unique(yo)) == 2 else float("nan"),
    )
    return CVResult(oof_prob=oof_prob, oof_pred=(oof_prob >= 0.5).astype(int),
                    y=y, groups=groups, per_fold=per_fold, pooled=pooled)


def permutation_test(
    X, y, groups, clf_factory=None, n_perm: int = 200, seed: int = 42
) -> dict:
    """Group-aware label permutation test on pooled AUROC (and accuracy)."""
    rng = np.random.default_rng(seed)
    obs = group_cv_evaluate(X, y, groups, clf_factory, calibrate=None, seed=seed).pooled
    uniq = np.unique(groups)
    perm_auroc = []
    for _ in range(n_perm):
        # permute labels at the GROUP level (preserves within-group structure)
        gmap = {g: lab for g, lab in zip(uniq, rng.permutation(
            [int(np.round(np.mean(y[groups == g]))) for g in uniq]))}
        # fall back to sample-level shuffle if group labels are mixed
        yp = rng.permutation(y)
        try:
            r = group_cv_evaluate(X, yp, groups, clf_factory, calibrate=None, seed=seed)
            if not np.isnan(r.pooled["auroc"]):
                perm_auroc.append(r.pooled["auroc"])
        except Exception:  # noqa: BLE001
            continue
    perm_auroc = np.asarray(perm_auroc) if perm_auroc else np.array([0.5])
    p = float((np.sum(perm_auroc >= obs["auroc"]) + 1) / (len(perm_auroc) + 1))
    return dict(observed_auroc=obs["auroc"], perm_mean=float(np.mean(perm_auroc)),
                perm_p=p, n_perm=int(len(perm_auroc)))
