"""C5 — deep vs calibrated-classical head-to-head (honest).

The 2023 journal's headline was an LSTM. This stage asks the question its design never
answered under leak-free validation: *does a neural model actually beat the calibrated
classical baseline on the same subject/recording-independent folds?* We adopt deep ONLY
if it wins; otherwise we report that it lost (a legitimate finding — on 19 recordings a
deep net overfits, and WESAD's classical bar is ~0.98).

Two comparisons, both on the IDENTICAL folds as the classical models:
  * WESAD calm-vs-stress (LOSO)            — neural-MLP vs RandomForest
  * custom Relaxed-vs-Concentrated (LORO)  — neural-MLP vs LogReg, recording-level

Backends:
  * 'mlp'   — sklearn MLPClassifier (no extra dependency); routed through the SAME
              group_cv_evaluate (impute/scale/calibrate in-fold) for a fair comparison.
  * 'torch' — if torch is importable, additionally trains a small LSTM over each custom
              recording's window sequence (the true temporal model answering the 2023 LSTM).
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from yoga_impact import config
from yoga_impact.modeling import group_cv_evaluate, default_rf


def _backend() -> str:
    if config.DEEP_BACKEND == "mlp":
        return "mlp"
    try:
        import torch  # noqa: F401
        return "torch"
    except Exception:  # noqa: BLE001
        return "mlp"


def _mlp(seed: int = 42):
    return MLPClassifier(hidden_layer_sizes=(config.DEEP_HIDDEN, config.DEEP_HIDDEN),
                         activation="relu", alpha=1e-3, max_iter=400,
                         early_stopping=True, n_iter_no_change=20, random_state=seed)


# ---------------------------------------------------------------------------
# Optional torch LSTM over custom-recording window sequences
# ---------------------------------------------------------------------------
def _torch_lstm_custom(df, feats) -> dict | None:
    try:
        import torch
        from torch import nn
    except Exception:  # noqa: BLE001
        return None
    torch.manual_seed(config.SEED)
    rng = np.random.default_rng(config.SEED)

    # build per-recording sequences (ordered windows)
    recs = []
    for rid, g in df.groupby("recording_id"):
        X = g[feats].to_numpy(np.float32)
        recs.append((rid, str(g["subject"].iloc[0]), int(g["label"].iloc[0]), X))

    class LSTM(nn.Module):
        def __init__(self, n_in, hid):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hid, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    n_in = len(feats)
    ids = [r[0] for r in recs]
    oof = {}
    for held in ids:                                   # leave-one-RECORDING-out
        tr = [r for r in recs if r[0] != held]
        te = [r for r in recs if r[0] == held][0]
        # in-fold standardisation
        allX = np.concatenate([r[3] for r in tr], 0)
        mu, sd = allX.mean(0), allX.std(0) + 1e-6
        net = LSTM(n_in, config.DEEP_HIDDEN)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        net.train()
        for _ in range(config.DEEP_EPOCHS):
            order = rng.permutation(len(tr))
            for i in order:
                rid, subj, lab, X = tr[i]
                xb = torch.tensor((X - mu) / sd).unsqueeze(0)
                yb = torch.tensor([float(lab)])
                opt.zero_grad()
                loss = lossf(net(xb), yb)
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            xb = torch.tensor((te[3] - mu) / sd).unsqueeze(0)
            oof[held] = float(torch.sigmoid(net(xb)).item())

    from sklearn.metrics import roc_auc_score, accuracy_score
    y = np.array([r[2] for r in recs])
    p = np.array([oof[r[0]] for r in recs])
    return dict(recording_auroc=float(roc_auc_score(y, p)),
                recording_acc=float(accuracy_score(y, (p >= 0.5).astype(int))), n=len(recs))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run() -> dict:
    backend = _backend()
    print("=" * 66)
    print(f"DEEP vs CLASSICAL HEAD-TO-HEAD  (backend: {backend})")
    print("=" * 66)
    results: dict[str, dict] = {"backend": backend}

    # ---- WESAD calm vs stress (LOSO): MLP vs RandomForest ------------------
    from yoga_impact.wesad import build_features
    wdf = build_features(cache=True)
    wfeats = [c for c in wdf.columns if c not in {"subject", "label", "condition"}]
    wmask = wdf["label"].isin([1, 2, 4]).to_numpy()
    Xw = wdf[wfeats].to_numpy(float)[wmask]
    yw = wdf.loc[wmask, "label"].isin([1, 4]).astype(int).to_numpy()
    gw = wdf["subject"].to_numpy()[wmask]
    cls = group_cv_evaluate(Xw, yw, gw, lambda: default_rf(), calibrate="sigmoid")
    mlp = group_cv_evaluate(Xw, yw, gw, lambda: _mlp(), calibrate="sigmoid")
    results["wesad"] = dict(classical_auroc=cls.pooled["auroc"], deep_auroc=mlp.pooled["auroc"],
                            classical_balacc=cls.pooled["bal_acc"], deep_balacc=mlp.pooled["bal_acc"],
                            deep_wins=bool(mlp.pooled["auroc"] > cls.pooled["auroc"]))
    print(f"\n[WESAD calm vs stress, LOSO]")
    print(f"  classical (RF)  AUROC={cls.pooled['auroc']:.3f}  bal_acc={cls.pooled['bal_acc']:.3f}")
    print(f"  deep (MLP)      AUROC={mlp.pooled['auroc']:.3f}  bal_acc={mlp.pooled['bal_acc']:.3f}")
    print(f"  -> deep {'WINS' if results['wesad']['deep_wins'] else 'does NOT beat'} classical")

    # ---- custom Relaxed vs Concentrated (LORO, recording-level) ------------
    from yoga_impact.clean_custom import load_clean_recordings
    from yoga_impact.eeg_features import build_feature_matrix, feature_columns
    from yoga_impact.eeg_model_custom import _recording_level
    cdf = build_feature_matrix(load_clean_recordings())
    cfeats = feature_columns(cdf)
    Xc = cdf[cfeats].to_numpy(float)
    yc = cdf["label"].to_numpy(int)
    gc = cdf["recording_id"].to_numpy()
    cls_c = group_cv_evaluate(Xc, yc, gc, lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", C=0.5), calibrate="sigmoid")
    mlp_c = group_cv_evaluate(Xc, yc, gc, lambda: _mlp(), calibrate="sigmoid")
    _, cls_acc, cls_au = _recording_level(cdf, cls_c.oof_prob)
    _, mlp_acc, mlp_au = _recording_level(cdf, mlp_c.oof_prob)
    results["custom_mlp"] = dict(classical_recording_auroc=cls_au, deep_recording_auroc=mlp_au,
                                 deep_wins=bool(mlp_au > cls_au))
    print(f"\n[custom Relaxed vs Concentrated, LORO recording-level]")
    print(f"  classical (LogReg)  AUROC={cls_au:.3f}  acc={cls_acc:.3f}")
    print(f"  deep (MLP)          AUROC={mlp_au:.3f}  acc={mlp_acc:.3f}")

    # ---- optional torch LSTM (true temporal model, answers the 2023 LSTM) ---
    if backend == "torch":
        lstm = _torch_lstm_custom(cdf, cfeats)
        if lstm:
            lstm["deep_wins"] = bool(lstm["recording_auroc"] > cls_au)
            results["custom_lstm"] = lstm
            print(f"\n[custom LSTM over window sequences, LORO recording-level]")
            print(f"  classical (LogReg)  AUROC={cls_au:.3f}")
            print(f"  deep (LSTM)         AUROC={lstm['recording_auroc']:.3f}  acc={lstm['recording_acc']:.3f}")
            print(f"  -> LSTM {'WINS' if lstm['deep_wins'] else 'does NOT beat'} classical")

    # ---- verdict -----------------------------------------------------------
    any_win = (results["wesad"]["deep_wins"] or results["custom_mlp"]["deep_wins"]
               or results.get("custom_lstm", {}).get("deep_wins", False))
    results["verdict"] = ("deep beats classical on at least one axis" if any_win
                          else "deep does NOT beat the calibrated classical baseline "
                               "(classical retained); honest given small N and a high classical bar")
    print(f"\nVERDICT: {results['verdict']}")

    out = config.OUTPUT_ROOT / "deep_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nMetrics -> {out}")
    print("=" * 66)
    return results


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    run()
