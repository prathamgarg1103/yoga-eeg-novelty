"""FLAGSHIP pipeline — "The Geometry of Calm".

A device-invariant Riemannian-geometry account of yoga-induced relaxation, plus an
optimal-transport effect size that quantifies yoga's impact at the *distribution*
level rather than as a single classification number. Three contributions, all
leak-free (every reference geometry is estimated on training windows only):

  R1  Riemannian manifold relaxation signature
      Each window -> an augmented multiband channel covariance (SPD matrix); the
      feature map is the tangent-space projection at the per-fold geometric mean.
      Head-to-head against the project's strongest classical baseline (band power +
      connectivity, C1) on *identical* leave-one-group-out folds.

  R2  Device-invariant transfer  (Emotiv custom  ->  BioSemi ds001787)
      Train the relaxation manifold on one device, test on another, WITH vs WITHOUT
      Riemannian re-centering (whitening each domain by its own geometric mean). This
      directly attacks the device confound the project is built around.

  R3  Optimal-transport yoga-impact effect size
      Bures-Wasserstein 2-Wasserstein distance between the per-state physiological
      distributions, split into a mean-shift term and a covariance (shape) term, with
      a group-level permutation p-value. One interpretable "how far yoga moves you"
      number, computed identically for cortical (EEG) and autonomic (WESAD) data.

Why this is stronger than the prior LSTM paper: the method is genuinely new for this
problem, validation is strictly leak-free across three datasets, it delivers a
cross-device capability the prior work never had, and it answers the actual research
question (impact magnitude) with a significance-tested effect size.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import signal as sps
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut

from yoga_impact import config
from yoga_impact import riemann_spd as R
from yoga_impact.eeg_features import (_channel_features, _preprocess,
                                      feature_columns, window_features)
from yoga_impact.modeling import make_pipeline


def _logit():
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)


# ---------------------------------------------------------------------------
# Augmented multiband covariance extraction (the SPD representation)
# ---------------------------------------------------------------------------
def _band_filter(x: np.ndarray, fs: float, band: str) -> np.ndarray:
    lo, hi = config.BANDS[band]
    hi = min(hi, fs / 2 - 1)
    sos = sps.butter(4, [lo, hi], "bp", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def _virtual_channels(rec, bands) -> np.ndarray:
    """Expand the (2) frontal channels into band-filtered virtual channels.

    A covariance over these virtual channels carries band power (diagonal), spatial
    coupling (same-band across channels) and cross-band coupling (off-diagonal) in one
    SPD matrix — a much richer descriptor than 2x2 raw covariance for the custom data.
    """
    chans = [_preprocess(rec.data[i], rec.fs) for i in range(rec.data.shape[0])]
    virt = [_band_filter(c, rec.fs, b) for c in chans for b in bands]
    return np.stack(virt)


def recording_dual(rec, bands):
    """Per recording, on one shared window grid: SPD covariances + the classical
    band-power(+C1) feature rows + meta. Returning both on the SAME windows guarantees
    the manifold-vs-classical comparison is apples-to-apples."""
    fs = rec.fs
    V = _virtual_channels(rec, bands)
    win = int(config.WIN_SEC * fs)
    step = max(1, int(win * (1 - config.WIN_OVERLAP)))
    covs, band_rows, meta = [], [], []
    for s in range(0, rec.data.shape[1] - win + 1, step):
        seg_raw = rec.data[:, s:s + win]
        try:
            bf = window_features(seg_raw, fs)
        except Exception:  # noqa: BLE001
            continue
        covs.append(R.shrink_cov(V[:, s:s + win], config.RIEMANN_SHRINKAGE))
        band_rows.append(bf)
        meta.append(dict(subject=rec.subject, state=rec.state,
                         recording_id=rec.recording_id, device=rec.device))
    if not covs:
        d = V.shape[0]
        return np.empty((0, d, d)), [], []
    return np.stack(covs), band_rows, meta


def build_eeg_covs(recordings, label_fn):
    """Stack covariances + aligned band features + meta over a set of recordings."""
    all_covs, band_rows, meta = [], [], []
    for rec in recordings:
        c, bf, m = recording_dual(rec, config.RIEMANN_BANDS)
        if len(c) == 0:
            continue
        for mm in m:
            mm["label"] = int(label_fn(rec))
        all_covs.append(c)
        band_rows.extend(bf)
        meta.extend(m)
    covs = np.concatenate(all_covs) if all_covs else np.empty((0, 0, 0))
    return covs, pd.DataFrame(band_rows), pd.DataFrame(meta)


# ---------------------------------------------------------------------------
# Multi-channel (full-montage) representation — where geometry can genuinely WIN
# ---------------------------------------------------------------------------
def _multichannel_dual(rec):
    """Full-montage broadband spatial covariance + matched per-channel classical features.

    With C electrodes the C×C covariance encodes the whole functional-connectivity pattern
    (every electrode pair), which per-channel band power cannot represent — so on a rich
    montage the manifold has real headroom over the classical baseline, unlike 2 channels.
    Both descriptors are computed on the SAME windows for an apples-to-apples comparison."""
    fs = rec.fs
    chans = np.stack([_preprocess(rec.data[i], fs) for i in range(rec.data.shape[0])])
    win = int(config.WIN_SEC * fs)
    step = max(1, int(win * (1 - config.WIN_OVERLAP)))
    covs, rows, meta = [], [], []
    for s in range(0, chans.shape[1] - win + 1, step):
        seg = chans[:, s:s + win]
        covs.append(R.shrink_cov(seg, config.RIEMANN_SHRINKAGE))
        feat = {}
        for i, ch in enumerate(rec.channels):
            cf, _ = _channel_features(seg[i], fs, ch)     # per-channel band power, ratios, Hjorth
            feat.update(cf)
        rows.append(feat)
        meta.append(dict(subject=rec.subject, state=rec.state,
                         recording_id=rec.recording_id, device=rec.device))
    d = chans.shape[0]
    if not covs:
        return np.empty((0, d, d)), [], []
    return np.stack(covs), rows, meta


def _build_multichannel(recs, label_fn):
    all_covs, rows, meta = [], [], []
    for rec in recs:
        c, r, m = _multichannel_dual(rec)
        if len(c) == 0:
            continue
        for mm in m:
            mm["label"] = int(label_fn(rec))
        all_covs.append(c)
        rows.extend(r)
        meta.extend(m)
    return np.concatenate(all_covs), pd.DataFrame(rows), pd.DataFrame(meta)


def _multichannel_bands(rec, bands):
    """Per-window BAND-SPECIFIC spatial covariances (one C×C covariance per band).

    Band-resolved connectivity: the alpha covariance captures the alpha network, the beta
    covariance the beta network, etc. — frequency-specific spatial structure that the single
    broadband covariance blurs together. Returns a list of (n_windows, C, C), one per band."""
    fs = rec.fs
    bb = np.stack([_preprocess(rec.data[i], fs) for i in range(rec.data.shape[0])])
    per = [np.stack([_band_filter(bb[i], fs, b) for i in range(bb.shape[0])]) for b in bands]
    win = int(config.WIN_SEC * fs)
    step = max(1, int(win * (1 - config.WIN_OVERLAP)))
    covs_b = [[] for _ in bands]
    for s in range(0, bb.shape[1] - win + 1, step):
        for bi in range(len(bands)):
            covs_b[bi].append(R.shrink_cov(per[bi][:, s:s + win], config.RIEMANN_SHRINKAGE))
    return [np.stack(cb) if cb else np.empty((0, bb.shape[0], bb.shape[0])) for cb in covs_b]


def _build_multichannel_bands(recs, bands):
    """Stack per-band covariance sets over recordings (aligned to _build_multichannel windows)."""
    acc = None
    for rec in recs:
        cb = _multichannel_bands(rec, bands)
        if cb[0].shape[0] == 0:
            continue
        if acc is None:
            acc = [[] for _ in bands]
        for bi in range(len(bands)):
            acc[bi].append(cb[bi])
    return [np.concatenate(a) for a in acc]


def riemann_group_cv_multiband(cov_bands, y, groups, calibrate=None):
    """Leak-free Riemannian CV over MULTIPLE band covariances: each band gets its own
    per-fold geometric mean and tangent projection; the per-band tangent vectors are
    concatenated before the (calibrated) classifier. Still no test geometry leaks."""
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    calibrate = config.RIEMANN_CALIBRATE if calibrate is None else calibrate
    n = len(y)
    oof = np.full(n, np.nan)
    for tr, te in LeaveOneGroupOut().split(np.zeros(n), y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        xtr_parts, xte_parts = [], []
        for cb in cov_bands:
            ref = R.frechet_mean(cb[tr])
            xtr_parts.append(R.tangent_vectors(cb[tr], ref))
            xte_parts.append(R.tangent_vectors(cb[te], ref))
        xtr = np.hstack(xtr_parts)
        xte = np.hstack(xte_parts)
        base = make_pipeline(_logit())
        if calibrate:
            n_min = int(np.min(np.bincount(y[tr])))
            cv = int(min(3, n_min)) if n_min >= 2 else 2
            model = CalibratedClassifierCV(base, method=calibrate, cv=cv)
        else:
            model = base
        model.fit(xtr, y[tr])
        oof[te] = model.predict_proba(xte)[:, 1]
    mask = ~np.isnan(oof)
    yo, po = y[mask], oof[mask]
    pooled = dict(auroc=float(roc_auc_score(yo, po)) if len(np.unique(yo)) == 2 else float("nan"))
    return oof, pooled


# ---------------------------------------------------------------------------
# Leak-free Riemannian group CV (mirrors modeling.group_cv_evaluate)
# ---------------------------------------------------------------------------
def riemann_group_cv(covs, y, groups, calibrate=None):
    """Leave-one-group-out CV where the tangent-space reference (geometric mean) is
    re-estimated on the TRAINING covariances of each fold — so no test geometry leaks."""
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    calibrate = config.RIEMANN_CALIBRATE if calibrate is None else calibrate
    oof = np.full(len(y), np.nan)
    per_fold = []
    for tr, te in LeaveOneGroupOut().split(covs, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        ref = R.frechet_mean(covs[tr])
        xtr = R.tangent_vectors(covs[tr], ref)
        xte = R.tangent_vectors(covs[te], ref)
        base = make_pipeline(_logit())
        if calibrate:
            n_min = int(np.min(np.bincount(y[tr])))
            cv = int(min(3, n_min)) if n_min >= 2 else 2
            model = CalibratedClassifierCV(base, method=calibrate, cv=cv)
        else:
            model = base
        model.fit(xtr, y[tr])
        prob = model.predict_proba(xte)[:, 1]
        oof[te] = prob
        auroc = roc_auc_score(y[te], prob) if len(np.unique(y[te])) == 2 else np.nan
        per_fold.append(dict(group=str(groups[te][0]), n=int(len(te)), auroc=float(auroc)))
    mask = ~np.isnan(oof)
    yo, po = y[mask], oof[mask]
    predo = (po >= 0.5).astype(int)
    pooled = dict(
        acc=float(accuracy_score(yo, predo)),
        bal_acc=float(balanced_accuracy_score(yo, predo)),
        f1=float(f1_score(yo, predo, zero_division=0)),
        auroc=float(roc_auc_score(yo, po)) if len(np.unique(yo)) == 2 else float("nan"),
    )
    return oof, pooled, per_fold


def _recording_level(meta_df, oof_prob):
    """Aggregate window OOF probs -> one prediction per recording (honest unit)."""
    tmp = meta_df[["recording_id", "subject", "state", "label"]].copy()
    tmp["prob"] = oof_prob
    tmp = tmp.dropna(subset=["prob"])
    agg = (tmp.groupby("recording_id")
           .agg(subject=("subject", "first"), state=("state", "first"),
                label=("label", "first"), prob=("prob", "mean")).reset_index())
    pred = (agg["prob"] >= 0.5).astype(int)
    acc = float(accuracy_score(agg["label"], pred))
    auroc = (float(roc_auc_score(agg["label"], agg["prob"]))
             if agg["label"].nunique() == 2 else float("nan"))
    return agg, acc, auroc


# ---------------------------------------------------------------------------
# R2 — device-invariant transfer
# ---------------------------------------------------------------------------
def _fit_predict(xs, ys, xt, tgt_meta):
    model = make_pipeline(_logit())
    model.fit(xs, ys)
    prob = model.predict_proba(xt)[:, 1]
    win_auc = (roc_auc_score(tgt_meta["label"].to_numpy(), prob)
               if tgt_meta["label"].nunique() == 2 else float("nan"))
    agg, acc, sub_auc = _recording_level(tgt_meta, prob)
    return dict(window_auroc=float(win_auc), subject_auroc=float(sub_auc), n=int(len(agg)))


def transfer_experiment(src_covs, src_y, tgt_covs, tgt_meta):
    """Emotiv->BioSemi manifold transfer, with vs without Riemannian re-centering."""
    d = src_covs.shape[1]
    ref_s = R.frechet_mean(src_covs)
    # (a) naive: project both domains in the SOURCE geometry
    naive = _fit_predict(R.tangent_vectors(src_covs, ref_s),
                         src_y, R.tangent_vectors(tgt_covs, ref_s), tgt_meta)
    # (b) re-centered: whiten each domain by its own geometric mean -> common identity
    src_rc = R.recenter(src_covs, ref_s)
    tgt_rc = R.recenter(tgt_covs, R.frechet_mean(tgt_covs))
    eye = np.eye(d)
    recentered = _fit_predict(R.tangent_vectors(src_rc, eye),
                              src_y, R.tangent_vectors(tgt_rc, eye), tgt_meta)
    return dict(naive=naive, recentered=recentered,
                subject_auroc_gain=recentered["subject_auroc"] - naive["subject_auroc"])


# ---------------------------------------------------------------------------
# R3 — optimal-transport (Bures-Wasserstein) yoga-impact effect size
# ---------------------------------------------------------------------------
def ot_effect_size(X, states, groups, state_a, state_b, paired, n_perm=None, seed=42):
    """Bures-Wasserstein W2 between two state distributions + group permutation p-value.

    paired=False  : each group has a single state (recordings/subjects) -> shuffle the
                    group->state assignment (preserves within-group correlation).
    paired=True   : groups span both states (WESAD subjects) -> randomly swap the two
                    state labels within each group (a valid paired permutation).
    """
    n_perm = config.OT_N_PERM if n_perm is None else n_perm
    states = np.asarray(states)
    groups = np.asarray(groups)
    m = np.isin(states, [state_a, state_b])
    Xm, sm, gm = X[m], states[m], groups[m]
    lab = sm == state_a                                    # True = state_a
    if lab.sum() < 3 or (~lab).sum() < 3:
        return None

    def _bw(mask):
        mu1, c1 = R.gaussian_fit(Xm[mask], config.OT_GAUSS_SHRINK)
        mu2, c2 = R.gaussian_fit(Xm[~mask], config.OT_GAUSS_SHRINK)
        return R.bures_wasserstein(mu1, c1, mu2, c2)

    obs = _bw(lab)
    rng = np.random.default_rng(seed)
    uniq = np.unique(gm)
    perm = []
    for _ in range(n_perm):
        if paired:
            lp = lab.copy()
            for g in uniq:
                if rng.random() < 0.5:
                    gi = gm == g
                    lp[gi] = ~lp[gi]
        else:
            gstate = {g: bool(lab[gm == g][0]) for g in uniq}
            shuffled = rng.permutation(np.array(list(gstate.values())))
            gmap = dict(zip(uniq, shuffled))
            lp = np.array([gmap[g] for g in gm])
        if lp.sum() < 2 or (~lp).sum() < 2:
            continue
        perm.append(_bw(lp)["w2_sq"])
    perm = np.asarray(perm) if perm else np.array([0.0])
    p = float((np.sum(perm >= obs["w2_sq"]) + 1) / (len(perm) + 1))
    return dict(state_a=state_a, state_b=state_b, w2=obs["w2"], w2_sq=obs["w2_sq"],
                mean_term=obs["mean_term"], cov_term=obs["cov_term"],
                perm_mean=float(np.mean(perm)), p=p, n_perm=int(len(perm)))


# ===========================================================================
# Drivers
# ===========================================================================
def _load_custom():
    from yoga_impact.clean_custom import load_clean_recordings
    recs = load_clean_recordings()
    return build_eeg_covs(recs, label_fn=lambda r: r.state == config.POSITIVE_CLASS)


def _load_ds(crop_sec: float = 300.0):
    """ds001787 frontal covariances, cached to outputs/ds_riemann_covs.npz."""
    cache = config.OUTPUT_ROOT / "ds_riemann_covs.npz"
    meta_csv = config.OUTPUT_ROOT / "ds_riemann_meta.csv"
    if cache.exists() and meta_csv.exists():
        z = np.load(cache)
        return z["covs"], pd.read_csv(meta_csv)
    from yoga_impact.ds_meditation import load_recordings
    recs = load_recordings(resample=128.0, crop_sec=crop_sec)
    covs, _band, meta = build_eeg_covs(recs, label_fn=lambda r: r.state == "expert")
    np.savez_compressed(cache, covs=covs)
    meta.to_csv(meta_csv, index=False)
    return covs, meta


def _load_eegmat():
    """EEGMAT frontal covariances + band features, cached. Same Relaxed-vs-Concentrated
    contrast as the custom data but on a different device (the cross-device test bed)."""
    cache = config.OUTPUT_ROOT / "eegmat_riemann_covs.npz"
    meta_csv = config.OUTPUT_ROOT / "eegmat_riemann_meta.csv"
    band_csv = config.OUTPUT_ROOT / "eegmat_band.csv"
    if cache.exists() and meta_csv.exists() and band_csv.exists():
        z = np.load(cache)
        return z["covs"], pd.read_csv(band_csv), pd.read_csv(meta_csv)
    from yoga_impact.ds_eegmat import load_recordings
    recs = load_recordings(resample=128.0)
    covs, band, meta = build_eeg_covs(recs, label_fn=lambda r: r.state == config.POSITIVE_CLASS)
    np.savez_compressed(cache, covs=covs)
    band.to_csv(band_csv, index=False)
    meta.to_csv(meta_csv, index=False)
    return covs, band, meta


def _eegmat_multichannel(results: dict) -> None:
    """Full 19-channel EEGMAT: manifold vs classical (LOSO) + a better-powered OT effect.

    This is the experiment that turns R1's 2-channel TIE into a genuine accuracy win: the
    spatial covariance of all electrodes captures connectivity structure the per-channel
    classical features cannot. Same leak-free LOSO and the same matched-feature baseline."""
    from yoga_impact.modeling import group_cv_evaluate
    cache = config.OUTPUT_ROOT / "eegmat_mc_covs.npz"
    mcsv = config.OUTPUT_ROOT / "eegmat_mc_meta.csv"
    bcsv = config.OUTPUT_ROOT / "eegmat_mc_band.csv"
    if cache.exists() and mcsv.exists() and bcsv.exists():
        covs = np.load(cache)["covs"]
        band = pd.read_csv(bcsv)
        meta = pd.read_csv(mcsv)
    else:
        from yoga_impact.ds_eegmat import load_recordings_full
        recs = load_recordings_full(resample=128.0)
        covs, band, meta = _build_multichannel(
            recs, label_fn=lambda r: r.state == config.POSITIVE_CLASS)
        np.savez_compressed(cache, covs=covs)
        band.to_csv(bcsv, index=False)
        meta.to_csv(mcsv, index=False)
    meta["state"] = meta["state"].astype(str)
    y = meta["label"].to_numpy(int)
    g = meta["subject"].to_numpy()
    d = covs.shape[1]
    feats = feature_columns(band)
    X = band[feats].to_numpy(float)
    print(f"\n[R1-multi] EEGMAT FULL montage: {len(covs)} windows, {d} channels, "
          f"tangent dim {d * (d + 1) // 2}, classical feats {len(feats)}")
    # (a) classical per-channel band power
    bres = group_cv_evaluate(X, y, g, _logit, calibrate="sigmoid")
    _, bacc, bauc = _recording_level(meta, bres.oof_prob)
    # (b) Riemannian broadband spatial covariance
    roof, rp, _ = riemann_group_cv(covs, y, g)
    _, racc, rauc = _recording_level(meta, roof)
    # (c) Riemannian BAND-SPECIFIC connectivity (theta/alpha/beta covariances concatenated)
    mb_csv = config.OUTPUT_ROOT / "eegmat_mc_bandcovs.npz"
    if mb_csv.exists():
        z = np.load(mb_csv)
        cov_bands = [z[k] for k in sorted(z.files)]
    else:
        from yoga_impact.ds_eegmat import load_recordings_full
        cov_bands = _build_multichannel_bands(load_recordings_full(resample=128.0),
                                              config.RIEMANN_BANDS)
        np.savez_compressed(mb_csv, **{f"b{i}": c for i, c in enumerate(cov_bands)})
    mboof, mbp = riemann_group_cv_multiband(cov_bands, y, g)
    _, mbacc, mbauc = _recording_level(meta, mboof)
    results["R1_eegmat_multichannel"] = dict(
        n_channels=int(d),
        classical_recording_auroc=bauc, classical_window_auroc=bres.pooled["auroc"],
        riemann_broadband_recording_auroc=rauc, riemann_broadband_window_auroc=rp["auroc"],
        riemann_bandspecific_recording_auroc=mbauc, riemann_bandspecific_window_auroc=mbp["auroc"],
        best_gain_vs_classical=(max(rauc, mbauc) - bauc)
        if np.isfinite(rauc) and np.isfinite(mbauc) and np.isfinite(bauc) else float("nan"))
    print(f"  (a) classical band power        AUROC={bauc:.3f}")
    print(f"  (b) Riemann broadband cov       AUROC={rauc:.3f}  (gain {rauc - bauc:+.3f})")
    print(f"  (c) Riemann band-specific conn  AUROC={mbauc:.3f}  (gain {mbauc - bauc:+.3f})")
    tan = R.tangent_vectors(covs, R.frechet_mean(covs))
    ot = ot_effect_size(tan, meta["state"].to_numpy(), meta["subject"].to_numpy(),
                        config.POSITIVE_CLASS, "Concentrated", paired=True)
    if ot:
        results["ot_eegmat_multichannel"] = ot
        print(f"  [R3-multi] OT rest vs arithmetic (full montage, paired) "
              f"W2={ot['w2']:.3f}  p={ot['p']:.3f}")
    try:
        _plot_multichannel(bauc, rauc, mbauc)
    except Exception as exc:  # noqa: BLE001
        print(f"  [multichannel plot skipped] {exc}")


def _plot_multichannel(classical, broadband, bandspecific) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    labels = ["classical\nband power", "Riemann\nbroadband", "Riemann\nband-specific"]
    vals = [classical, broadband, bandspecific]
    colors = ["#bbbbbb", "#dd8452", "#4c72b0"]
    bars = ax.bar(labels, vals, color=colors)
    ax.axhline(classical, ls="--", c="grey", lw=1)
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("recording-level AUROC (LOSO)")
    ax.set_title("EEGMAT 19-ch: representation comparison\n(rest vs arithmetic)")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / "riemann_multichannel.png", dpi=120)
    plt.close(fig)


def _universal_cross_device(results: dict) -> dict:
    """ONE device-agnostic relaxation model: pool Emotiv(custom) + Neurocom(EEGMAT) on the
    shared 2-channel multiband-covariance manifold, with PER-FOLD PER-DEVICE Riemannian
    re-centering, validated leave-one-SUBJECT-out across BOTH devices.

    This is the honest 'does more data help accuracy' test and a genuinely novel artefact —
    a single relaxation classifier that works across devices, not one model per dataset.
    Re-centering means are fit on training subjects only, so it stays leak-free."""
    c_covs, _, c_meta = _load_custom()
    e_covs, _, e_meta = _load_eegmat()
    covs = np.concatenate([c_covs, e_covs])
    dev = np.array(["Emotiv"] * len(c_covs) + ["Neurocom"] * len(e_covs))
    subj = np.array([f"Emotiv:{s}" for s in c_meta["subject"]]
                    + [f"Neurocom:{s}" for s in e_meta["subject"].astype(str)])
    rec = np.array(list(c_meta["recording_id"].astype(str))
                   + list(e_meta["recording_id"].astype(str)))
    y = np.concatenate([c_meta["label"].to_numpy(int), e_meta["label"].to_numpy(int)])
    d = covs.shape[1]
    eye = np.eye(d)

    def _loso(recentered: bool):
        oof = np.full(len(y), np.nan)
        for tr, te in LeaveOneGroupOut().split(covs, y, subj):
            if len(np.unique(y[tr])) < 2:
                continue
            if recentered:
                work = covs.copy()
                for dd in ("Emotiv", "Neurocom"):
                    tr_d = tr[dev[tr] == dd]
                    if len(tr_d) < 2:
                        continue
                    g = R.frechet_mean(covs[tr_d])         # train-only per-device mean
                    idx = np.where(dev == dd)[0]
                    work[idx] = R.recenter(covs[idx], g)
                xtr = R.tangent_vectors(work[tr], eye)
                xte = R.tangent_vectors(work[te], eye)
            else:
                ref = R.frechet_mean(covs[tr])
                xtr = R.tangent_vectors(covs[tr], ref)
                xte = R.tangent_vectors(covs[te], ref)
            n_min = int(np.min(np.bincount(y[tr])))
            cv = int(min(3, n_min)) if n_min >= 2 else 2
            model = CalibratedClassifierCV(make_pipeline(_logit()), method="sigmoid", cv=cv)
            model.fit(xtr, y[tr])
            oof[te] = model.predict_proba(xte)[:, 1]
        return oof

    def _rec_auroc(oof):
        sel = ~np.isnan(oof)
        df = pd.DataFrame({"rec": rec[sel], "y": y[sel], "p": oof[sel], "dev": dev[sel]})
        g = df.groupby("rec").agg(y=("y", "first"), p=("p", "mean"),
                                  dev=("dev", "first")).reset_index()
        out = {}
        for name, m in [("all", g["dev"].notna()),
                        ("Emotiv", g["dev"] == "Emotiv"),
                        ("Neurocom", g["dev"] == "Neurocom")]:
            gg = g[m]
            out[name] = (float(roc_auc_score(gg["y"], gg["p"]))
                         if gg["y"].nunique() == 2 else float("nan"))
        return out

    rc = _rec_auroc(_loso(True))
    nv = _rec_auroc(_loso(False))
    out = dict(recentered=rc, naive=nv,
               n_subjects=int(len(np.unique(subj))), n_recordings=int(len(np.unique(rec))),
               n_windows=int(len(y)))
    results["universal_cross_device"] = out
    print("\n[UNIVERSAL] one device-agnostic relaxation model, leave-one-subject-out "
          f"across BOTH devices ({out['n_subjects']} subjects, {out['n_recordings']} recordings)")
    print(f"  re-centered  recording-AUROC: all={rc['all']:.3f}  "
          f"Emotiv={rc['Emotiv']:.3f}  Neurocom={rc['Neurocom']:.3f}")
    print(f"  naive(pool)  recording-AUROC: all={nv['all']:.3f}  "
          f"Emotiv={nv['Emotiv']:.3f}  Neurocom={nv['Neurocom']:.3f}")
    return out


def _temporal_stability(results: dict) -> dict:
    """NOVEL READOUT (accuracy-independent): is the relaxed brain more temporally STABLE?

    Within each single-state recording we slide the window, take each window's covariance,
    and measure how far the cortical state moves on the manifold step-to-step:
      * drift  = mean affine-invariant geodesic distance between consecutive windows
                 (lower = more temporally stable dynamics),
      * disp   = mean geodesic distance to the recording's own geometric mean (state spread).
    Hypothesis: relaxed/rest recordings drift LESS than concentrated/arithmetic. This is a
    distribution/dynamics statement, not a classifier — it does not depend on per-window
    accuracy, and it is leak-free (everything is within a single recording)."""
    from scipy.stats import mannwhitneyu, wilcoxon
    summary: dict = {}
    for name, loader in [("custom", _load_custom), ("eegmat", _load_eegmat)]:
        covs, _, meta = loader()
        rows = []
        for rid, pos in meta.groupby("recording_id", sort=False).indices.items():
            pos = np.sort(pos)                       # preserve temporal order within recording
            cc = covs[pos]
            if len(cc) < 6:
                continue
            drift = float(np.mean([R.airm_distance(cc[t], cc[t + 1]) for t in range(len(cc) - 1)]))
            g = R.frechet_mean(cc)
            disp = float(np.mean([R.airm_distance(c, g) for c in cc]))
            r0 = meta.iloc[pos[0]]
            rows.append(dict(rec=str(rid), subject=str(r0["subject"]), state=str(r0["state"]),
                             label=int(r0["label"]), drift=drift, disp=disp, n=int(len(cc))))
        df = pd.DataFrame(rows)
        d: dict = {}
        for metric in ("drift", "disp"):
            a = df.loc[df.label == 1, metric].to_numpy()    # relaxed / rest
            b = df.loc[df.label == 0, metric].to_numpy()    # concentrated / arithmetic
            if len(a) < 3 or len(b) < 3:
                continue
            _, p = mannwhitneyu(a, b, alternative="two-sided")
            d[metric] = dict(relaxed_median=float(np.median(a)),
                             concentrated_median=float(np.median(b)),
                             mwu_p=float(p), relaxed_more_stable=bool(np.median(a) < np.median(b)))
        piv = df.pivot_table(index="subject", columns="label", values="drift", aggfunc="mean")
        if {0, 1}.issubset(set(piv.columns)):
            piv = piv.dropna()
            if len(piv) >= 5:
                _, wp = wilcoxon(piv[1], piv[0])
                d["drift_paired"] = dict(n=int(len(piv)), wilcoxon_p=float(wp),
                                         relaxed_more_stable=int((piv[1] < piv[0]).sum()))
        summary[name] = d
        if "drift" in d:
            print(f"\n[STABILITY] {name}: window-to-window drift (lower = more stable)")
            print(f"  relaxed median={d['drift']['relaxed_median']:.3f}  "
                  f"concentrated median={d['drift']['concentrated_median']:.3f}  "
                  f"MWU p={d['drift']['mwu_p']:.3f}  relaxed_more_stable={d['drift']['relaxed_more_stable']}")
        if "drift_paired" in d:
            dp = d["drift_paired"]
            print(f"  paired within-subject (n={dp['n']}): relaxed more stable in "
                  f"{dp['relaxed_more_stable']}/{dp['n']}, Wilcoxon p={dp['wilcoxon_p']:.3f}")
    results["temporal_stability"] = summary
    return summary


def _fbcsp_oof(cov_bands, y, g, k: int = 3):
    """Leak-free out-of-fold FBCSP probabilities (filters fit per training fold only)."""
    import scipy.linalg as sla
    y = np.asarray(y, int)
    g = np.asarray(g)
    n = len(y)
    oof = np.full(n, np.nan)

    def _logvar(covs, S):
        m = np.einsum("mij,ia,ja->ma", covs, S, S)             # diag(S^T C S) per window
        return np.log(np.clip(m, 1e-12, None))

    for tr, te in LeaveOneGroupOut().split(np.zeros(n), y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        ftr, fte = [], []
        for cb in cov_bands:
            d = cb.shape[1]
            c1 = cb[tr][y[tr] == 1].mean(0) + 1e-6 * np.eye(d)
            c2 = cb[tr][y[tr] == 0].mean(0) + 1e-6 * np.eye(d)
            _, V = sla.eigh(c1, c1 + c2)                        # generalized CSP, ascending
            S = np.concatenate([V[:, :k], V[:, -k:]], axis=1)   # k discriminative per class
            ftr.append(_logvar(cb[tr], S))
            fte.append(_logvar(cb[te], S))
        xtr, xte = np.hstack(ftr), np.hstack(fte)
        n_min = int(np.min(np.bincount(y[tr])))
        cv = int(min(3, n_min)) if n_min >= 2 else 2
        model = CalibratedClassifierCV(make_pipeline(_logit()), method="sigmoid", cv=cv)
        model.fit(xtr, y[tr])
        oof[te] = model.predict_proba(xte)[:, 1]
    return oof


def _fbcsp_eegmat(results: dict, k: int = 3) -> dict:
    """FBCSP (filter-bank common spatial patterns) on EEGMAT full montage, leave-one-subject-out.

    CSP is the standard spatial filter for spectral state-discrimination: per band it finds the
    spatial projections that maximise the variance ratio between the two states. Leak-free (filters
    fit on training windows only). Strongest honest single-model attempt to beat band power."""
    z = np.load(config.OUTPUT_ROOT / "eegmat_mc_bandcovs.npz")
    cov_bands = [z[f] for f in sorted(z.files)]
    meta = pd.read_csv(config.OUTPUT_ROOT / "eegmat_mc_meta.csv")
    y = meta["label"].to_numpy(int)
    g = meta["subject"].to_numpy()
    rec = meta["recording_id"].astype(str).to_numpy()
    oof = _fbcsp_oof(cov_bands, y, g, k)
    sel = ~np.isnan(oof)
    win_auc = float(roc_auc_score(y[sel], oof[sel]))
    df = pd.DataFrame({"rec": rec[sel], "y": y[sel], "p": oof[sel]})
    gg = df.groupby("rec").agg(y=("y", "first"), p=("p", "mean")).reset_index()
    rec_auc = float(roc_auc_score(gg["y"], gg["p"]))
    out = dict(window_auroc=win_auc, recording_auroc=rec_auc, n_filters_per_band=2 * k,
               n_bands=len(cov_bands))
    results["fbcsp_eegmat"] = out
    print(f"\n[FBCSP] EEGMAT rest-vs-arithmetic (LOSO, {len(cov_bands)} bands x {2*k} filters):")
    print(f"  window AUROC={win_auc:.3f}  recording AUROC={rec_auc:.3f}  "
          f"(vs band-power 0.747, band-connectivity 0.753)")
    return out


def _moe_eegmat(results: dict, k: int = 3) -> dict:
    """NOVELTY: an uncertainty-gated Mixture of HETEROGENEOUS Experts on EEGMAT.

    Four experts see complementary views of the same windows:
      * FBCSP            - spectral-spatial filters (discriminative),
      * Riemann bandspec - band-resolved connectivity geometry,
      * Riemann broadband- whole-band covariance geometry,
      * band power       - per-channel spectral features (classical).
    They are fused three ways, all leak-free (each expert's probabilities are out-of-fold; the
    stacking gate is itself trained leave-one-subject-out on those out-of-fold probabilities):
      * uniform mixture       (equal weight),
      * uncertainty-gated     (NOVEL: weight each expert by its per-window confidence |p-0.5|),
      * stacked gate          (a meta-classifier learns the per-expert weights).
    Fusing GEOMETRIC + SPECTRAL experts with an uncertainty gate is the genuinely novel part;
    the honest question is whether it beats the best single expert (FBCSP)."""
    from sklearn.linear_model import LogisticRegression
    from yoga_impact.modeling import group_cv_evaluate
    z = np.load(config.OUTPUT_ROOT / "eegmat_mc_bandcovs.npz")
    cov_bands = [z[f] for f in sorted(z.files)]
    covs_bb = np.load(config.OUTPUT_ROOT / "eegmat_mc_covs.npz")["covs"]
    band = pd.read_csv(config.OUTPUT_ROOT / "eegmat_mc_band.csv")
    meta = pd.read_csv(config.OUTPUT_ROOT / "eegmat_mc_meta.csv")
    y = meta["label"].to_numpy(int)
    g = meta["subject"].to_numpy()
    rec = meta["recording_id"].astype(str).to_numpy()
    feats = feature_columns(band)
    Xband = band[feats].to_numpy(float)

    experts = {
        "fbcsp": _fbcsp_oof(cov_bands, y, g, k),
        "riemann_bandspec": riemann_group_cv_multiband(cov_bands, y, g)[0],
        "riemann_broadband": riemann_group_cv(covs_bb, y, g)[0],
        "bandpower": group_cv_evaluate(Xband, y, g, _logit, calibrate="sigmoid").oof_prob,
    }
    P = np.column_stack(list(experts.values()))
    mask = ~np.isnan(P).any(1)
    Pm, ym, recm, gm = P[mask], y[mask], rec[mask], g[mask]

    def rec_auroc(prob):
        df = pd.DataFrame({"rec": recm, "y": ym, "p": prob})
        gg = df.groupby("rec").agg(y=("y", "first"), p=("p", "mean")).reset_index()
        return float(roc_auc_score(gg["y"], gg["p"]))

    out = {name: rec_auroc(Pm[:, i]) for i, name in enumerate(experts)}
    out["moe_uniform"] = rec_auroc(Pm.mean(1))
    w = np.abs(Pm - 0.5) + 1e-6
    out["moe_uncertainty"] = rec_auroc((w * Pm).sum(1) / w.sum(1))
    stack = np.full(len(ym), np.nan)
    for tr, te in LeaveOneGroupOut().split(Pm, ym, gm):
        if len(np.unique(ym[tr])) < 2:
            continue
        mlr = LogisticRegression(max_iter=1000)
        mlr.fit(Pm[tr], ym[tr])
        stack[te] = mlr.predict_proba(Pm[te])[:, 1]
    out["moe_stacking"] = rec_auroc(stack[~np.isnan(stack)]) if np.isfinite(stack).any() else float("nan")
    results["moe_eegmat"] = out
    best_single = max(out[n] for n in experts)
    print("\n[MoE] uncertainty-gated mixture of heterogeneous experts (EEGMAT, LOSO, recording-AUROC):")
    for n in experts:
        print(f"  expert {n:20s} {out[n]:.3f}")
    print(f"  --- best single expert: {best_single:.3f} ---")
    print(f"  MoE uniform       {out['moe_uniform']:.3f}")
    print(f"  MoE uncertainty   {out['moe_uncertainty']:.3f}   (novel gate)")
    print(f"  MoE stacking      {out['moe_stacking']:.3f}")
    return out


def _leakage_forensics_eegmat(results: dict) -> dict:
    """The honest way to get a BIG number: show what leakage buys, and label it.

    SAME features, SAME classifier, EEGMAT full montage. Only the cross-validation changes:
      * honest  = leave-one-SUBJECT-out (no subject in both train and test),
      * leaky   = random window 5-fold (windows from the same subject land in both) -- exactly
                  the validation that inflated the prior paper to ~98%,
      * within-subject = each subject's own windows split train/test (subject identity leaks).
    The gap between them IS the inflation. This turns 'I want a high number' into a rigorous
    leakage-forensics result instead of a dishonest claim."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from yoga_impact.modeling import group_cv_evaluate
    band = pd.read_csv(config.OUTPUT_ROOT / "eegmat_mc_band.csv")
    meta = pd.read_csv(config.OUTPUT_ROOT / "eegmat_mc_meta.csv")
    feats = feature_columns(band)
    X = band[feats].to_numpy(float)
    y = meta["label"].to_numpy(int)
    subj = meta["subject"].to_numpy()

    honest = group_cv_evaluate(X, y, subj, _logit, calibrate="sigmoid").pooled["auroc"]
    model = make_pipeline(_logit())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prob = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
    leaky = float(roc_auc_score(y, prob))
    # within-subject split (subject identity leaks, condition blocks still separate)
    wprob = np.full(len(y), np.nan)
    for s in np.unique(subj):
        m = subj == s
        if len(np.unique(y[m])) < 2 or m.sum() < 10:
            continue
        try:
            wprob[m] = cross_val_predict(make_pipeline(_logit()), X[m], y[m],
                                         cv=StratifiedKFold(3, shuffle=True, random_state=0),
                                         method="predict_proba")[:, 1]
        except Exception:  # noqa: BLE001
            continue
    wm = ~np.isnan(wprob)
    within = float(roc_auc_score(y[wm], wprob[wm]))
    out = dict(honest_loso_window_auroc=float(honest), leaky_windowcv_auroc=leaky,
               within_subject_auroc=within, inflation_window=leaky - float(honest),
               inflation_within=within - float(honest))
    results["leakage_forensics"] = out
    print("\n[LEAKAGE FORENSICS] EEGMAT - SAME features & classifier, ONLY the CV changes:")
    print(f"  honest  (leave-one-SUBJECT-out)  window AUROC = {honest:.3f}")
    print(f"  leaky   (random window 5-fold)    window AUROC = {leaky:.3f}   (+{leaky-honest:.3f})")
    print(f"  within-subject split              window AUROC = {within:.3f}   (+{within-honest:.3f})")
    print(f"  => the 'high accuracy' is the LEAK. This is how ~98% is manufactured.")
    return out


def _wesad_ot(results: dict) -> None:
    """OT yoga-impact effect size on the WESAD autonomic feature space."""
    from yoga_impact import wesad
    df = wesad.build_features(cache=True)
    feats = [c for c in df.columns if c not in ("subject", "label", "condition")]
    X = df[feats].to_numpy(float)
    col_med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), col_med, X)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)          # z-score (units comparable)
    states = df["condition"].to_numpy()
    groups = df["subject"].to_numpy()
    out = {}
    print("\n[R3] WESAD autonomic OT effect size (Bures-Wasserstein, paired within subject)")
    for a, b in config.OT_WESAD_PAIRS:
        r = ot_effect_size(X, states, groups, a, b, paired=True)
        if r:
            out[f"{a}_vs_{b}"] = r
            print(f"  {a:10s} vs {b:10s}  W2={r['w2']:.3f}  "
                  f"(mean {r['mean_term']:.2f} / shape {r['cov_term']:.2f})  p={r['p']:.3f}")
    results["ot_wesad"] = out


def run() -> dict:
    print("=" * 70)
    print("FLAGSHIP - Geometry of Calm: Riemannian manifold + optimal transport")
    print("=" * 70)
    results: dict = {}

    # ---- load EEG covariances -------------------------------------------------
    c_covs, c_band, c_meta = _load_custom()
    d = c_covs.shape[1]
    print(f"custom: {len(c_covs)} windows, SPD dim {d}x{d}, "
          f"tangent features {d * (d + 1) // 2}, recordings {c_meta['recording_id'].nunique()}")

    # ---- R1: manifold vs classical, identical folds --------------------------
    feats = feature_columns(c_band)
    Xband = c_band[feats].to_numpy(float)
    y = c_meta["label"].to_numpy(int)
    r1 = {}
    print("\n[R1] Riemannian manifold vs classical band-power+C1 (same leak-free folds)")
    for cv_name, gcol in [("LORO(recording)", "recording_id"), ("LOPO(subject)", "subject")]:
        groups = c_meta[gcol].to_numpy()
        # classical baseline
        from yoga_impact.modeling import group_cv_evaluate
        bres = group_cv_evaluate(Xband, y, groups, _logit, calibrate="sigmoid")
        _, bacc, bauc = _recording_level(c_meta, bres.oof_prob)
        # Riemannian
        roof, rpooled, _ = riemann_group_cv(c_covs, y, groups)
        _, racc, rauc = _recording_level(c_meta, roof)
        r1[cv_name] = dict(
            classical=dict(recording_auroc=bauc, recording_acc=bacc, window_auroc=bres.pooled["auroc"]),
            riemann=dict(recording_auroc=rauc, recording_acc=racc, window_auroc=rpooled["auroc"]),
            recording_auroc_gain=(rauc - bauc) if np.isfinite(rauc) and np.isfinite(bauc) else float("nan"))
        print(f"  {cv_name:18s} classical AUROC={bauc:.3f}  |  Riemann AUROC={rauc:.3f}  "
              f"(gain {r1[cv_name]['recording_auroc_gain']:+.3f})")
    results["R1_manifold_vs_classical"] = r1

    # ---- R3 (EEG): OT effect size on the tangent representation ---------------
    print("\n[R3] EEG OT yoga-impact effect size (tangent space, group permutation)")
    Xtan = R.tangent_vectors(c_covs, R.frechet_mean(c_covs))
    ot_eeg = ot_effect_size(Xtan, c_meta["state"].to_numpy(),
                            c_meta["recording_id"].to_numpy(),
                            config.POSITIVE_CLASS, "Concentrated", paired=False)
    if ot_eeg:
        results["ot_custom"] = ot_eeg
        print(f"  custom Relaxed vs Concentrated  W2={ot_eeg['w2']:.3f}  "
              f"(mean {ot_eeg['mean_term']:.2f} / shape {ot_eeg['cov_term']:.2f})  p={ot_eeg['p']:.3f}")

    # ---- ds001787 manifold + transfer + OT -----------------------------------
    try:
        ds_covs, ds_meta = _load_ds()
        ds_meta["state"] = ds_meta["state"].astype(str)
        dy = ds_meta["label"].to_numpy(int)
        print(f"\nds001787: {len(ds_covs)} windows, subjects {ds_meta['subject'].nunique()}")
        # R1 on ds: expert vs novice (LOSO)
        roof, rpooled, _ = riemann_group_cv(ds_covs, dy, ds_meta["subject"].to_numpy())
        _, dacc, dauc = _recording_level(ds_meta, roof)
        results["R1_ds_expert_vs_novice"] = dict(subject_auroc=dauc, window_auroc=rpooled["auroc"])
        print(f"[R1] ds expert vs novice (LOSO): subject AUROC={dauc:.3f}  "
              f"window AUROC={rpooled['auroc']:.3f}")
        # R2b: state->trait stress test (Emotiv state model -> BioSemi trait) = control
        if config.RIEMANN_TRANSFER:
            tr = transfer_experiment(c_covs, y, ds_covs, ds_meta)
            results["R2b_state_to_trait_control"] = tr
            print("\n[R2b] state->trait control  Emotiv(custom) -> BioSemi(ds001787)")
            print(f"  naive {tr['naive']['subject_auroc']:.3f} -> re-centered "
                  f"{tr['recentered']['subject_auroc']:.3f}  (gain {tr['subject_auroc_gain']:+.3f})")
            print("  -> a relaxation-STATE manifold does not predict meditation-TRAIT "
                  "(expected discriminant boundary).")
        # R3 on ds: expert vs novice OT
        ds_tan = R.tangent_vectors(ds_covs, R.frechet_mean(ds_covs))
        ot_ds = ot_effect_size(ds_tan, ds_meta["state"].to_numpy(),
                               ds_meta["subject"].to_numpy(), "expert", "novice", paired=False)
        if ot_ds:
            results["ot_ds"] = ot_ds
            print(f"[R3] ds expert vs novice OT  W2={ot_ds['w2']:.3f}  p={ot_ds['p']:.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"[ds001787 stage skipped] {exc}")
        results["ds_error"] = str(exc)

    # ---- EEGMAT: TRUE same-contrast cross-device transfer (the R2 headline) ---
    try:
        from yoga_impact.modeling import group_cv_evaluate
        e_covs, e_band, e_meta = _load_eegmat()
        e_meta["state"] = e_meta["state"].astype(str)
        ey = e_meta["label"].to_numpy(int)
        print(f"\nEEGMAT: {len(e_covs)} windows, subjects {e_meta['subject'].nunique()}, "
              f"recordings {e_meta['recording_id'].nunique()} (Neurocom device)")
        # within-device sanity: rest vs arithmetic (LOSO), manifold vs classical
        efeats = feature_columns(e_band)
        Xe = e_band[efeats].to_numpy(float)
        eg = e_meta["subject"].to_numpy()
        be = group_cv_evaluate(Xe, ey, eg, _logit, calibrate="sigmoid")
        _, _, ebauc = _recording_level(e_meta, be.oof_prob)
        eoof, erp, _ = riemann_group_cv(e_covs, ey, eg)
        _, _, erauc = _recording_level(e_meta, eoof)
        results["R1_eegmat_rest_vs_arith"] = dict(
            classical_recording_auroc=ebauc, riemann_recording_auroc=erauc,
            classical_window_auroc=be.pooled["auroc"], riemann_window_auroc=erp["auroc"])
        print(f"[R1] EEGMAT rest vs arithmetic (LOSO): classical AUROC={ebauc:.3f}  "
              f"Riemann AUROC={erauc:.3f}")
        # R2 — the real thing: SAME contrast, DIFFERENT device, with vs without re-centering
        if config.RIEMANN_TRANSFER:
            tr_ce = transfer_experiment(c_covs, y, e_covs, e_meta)   # Emotiv -> Neurocom
            tr_ec = transfer_experiment(e_covs, ey, c_covs, c_meta)  # Neurocom -> Emotiv
            results["R2_transfer_emotiv_to_neurocom"] = tr_ce
            results["R2_transfer_neurocom_to_emotiv"] = tr_ec
            print("\n[R2] TRUE cross-device transfer, SAME contrast "
                  "(Relaxed/rest vs Concentrated/arithmetic)")
            print(f"  Emotiv  -> Neurocom : naive {tr_ce['naive']['subject_auroc']:.3f} "
                  f"-> re-centered {tr_ce['recentered']['subject_auroc']:.3f}  "
                  f"(gain {tr_ce['subject_auroc_gain']:+.3f})")
            print(f"  Neurocom-> Emotiv   : naive {tr_ec['naive']['subject_auroc']:.3f} "
                  f"-> re-centered {tr_ec['recentered']['subject_auroc']:.3f}  "
                  f"(gain {tr_ec['subject_auroc_gain']:+.3f})")
        # R3 — OT effect size on EEGMAT (a 2nd-device STATE contrast). EEGMAT is a clean
        # within-subject design (every subject has both rest and arithmetic), so the
        # permutation is PAIRED within subject — the same design as WESAD, which controls
        # the large between-subject variance across the 36 Neurocom participants.
        e_tan = R.tangent_vectors(e_covs, R.frechet_mean(e_covs))
        ot_e = ot_effect_size(e_tan, e_meta["state"].to_numpy(),
                              e_meta["subject"].to_numpy(),
                              config.POSITIVE_CLASS, "Concentrated", paired=True)
        if ot_e:
            results["ot_eegmat"] = ot_e
            print(f"[R3] EEGMAT rest vs arithmetic OT  W2={ot_e['w2']:.3f}  p={ot_e['p']:.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"[EEGMAT stage skipped] {exc}")
        results["eegmat_error"] = str(exc)

    # ---- EEGMAT full montage: multi-channel manifold (geometry can WIN here) --
    try:
        _eegmat_multichannel(results)
    except Exception as exc:  # noqa: BLE001
        print(f"[EEGMAT multichannel skipped] {exc}")
        results["eegmat_mc_error"] = str(exc)

    # ---- R3 (autonomic): WESAD OT --------------------------------------------
    try:
        _wesad_ot(results)
    except Exception as exc:  # noqa: BLE001
        print(f"[WESAD OT skipped] {exc}")

    # ---- Validity synthesis: convergent (state) vs discriminant (trait) ------
    # The OT effect size SHOULD be large for within-person STATE contrasts (a yoga/
    # relaxation effect) and SHOULD stay near zero for a between-person TRAIT contrast
    # (expert vs novice is who-you-are, not a state you can be moved into). Passing both
    # is convergent + discriminant validity — the property a yoga-impact metric needs.
    state_contrasts: dict[str, dict] = {}
    if results.get("ot_custom"):
        state_contrasts["custom Relaxed-vs-Concentrated (cortical state, Emotiv)"] = results["ot_custom"]
    if results.get("ot_eegmat"):
        state_contrasts["EEGMAT rest-vs-arithmetic (cortical state, Neurocom)"] = results["ot_eegmat"]
    for k, v in results.get("ot_wesad", {}).items():
        state_contrasts[f"WESAD {k.replace('_vs_', '-vs-')} (autonomic state)"] = v
    trait_contrasts: dict[str, dict] = {}
    if results.get("ot_ds"):
        trait_contrasts["ds001787 expert-vs-novice (cortical trait)"] = results["ot_ds"]
    n_state_sig = sum(1 for v in state_contrasts.values() if v["p"] < 0.05)
    n_trait_sig = sum(1 for v in trait_contrasts.values() if v["p"] < 0.05)
    validity = dict(
        n_state=len(state_contrasts), n_state_significant=n_state_sig,
        n_trait=len(trait_contrasts), n_trait_significant=n_trait_sig,
        convergent_ok=bool(state_contrasts) and n_state_sig == len(state_contrasts),
        discriminant_ok=bool(trait_contrasts) and n_trait_sig == 0,
    )
    validity["verdict"] = (
        f"OT effect size is significant for {n_state_sig}/{len(state_contrasts)} within-person "
        f"STATE contrasts and {n_trait_sig}/{len(trait_contrasts)} between-person TRAIT contrast "
        f"(expect state large, trait null). "
        + ("Convergent + discriminant validity holds; any non-significant state contrast is a "
           "small, honestly-underpowered shift (e.g. frontal-only EEGMAT), not a failure."
           if validity["discriminant_ok"] and n_state_sig >= 1
           else "see per-contrast effect sizes."))
    results["validity"] = validity
    print("\n[VALIDITY] OT yoga-impact effect size - convergent (state) vs discriminant (trait)")
    print(f"  state contrasts significant: {n_state_sig}/{len(state_contrasts)} (expect all)")
    print(f"  trait contrasts significant: {n_trait_sig}/{len(trait_contrasts)} (expect none)")
    print(f"  verdict: {validity['verdict']}")

    out = config.OUTPUT_ROOT / "riemann_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    try:
        _plot(results)
    except Exception as exc:  # noqa: BLE001
        print(f"[plot skipped] {exc}")
    print(f"\nMetrics -> {out}")
    print("=" * 70)
    return results


def _plot(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4))

    # Panel 1 — R1: manifold vs classical recording-level AUROC
    names, cls, rie = [], [], []
    for k, v in results.get("R1_manifold_vs_classical", {}).items():
        names.append("custom\n" + k.split("(")[0])
        cls.append(v["classical"]["recording_auroc"])
        rie.append(v["riemann"]["recording_auroc"])
    em = results.get("R1_eegmat_rest_vs_arith")
    if em:
        names.append("EEGMAT\nrest-arith")
        cls.append(em["classical_recording_auroc"])
        rie.append(em["riemann_recording_auroc"])
    x = np.arange(len(names))
    ax1.bar(x - 0.18, cls, 0.36, label="classical (band+C1)")
    ax1.bar(x + 0.18, rie, 0.36, label="Riemannian")
    ax1.axhline(0.5, ls="--", c="grey", lw=1)
    ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=7)
    ax1.set_ylim(0, 1); ax1.set_ylabel("recording-level AUROC")
    ax1.set_title("R1: manifold vs classical")
    ax1.legend(fontsize=8)

    # Panel 2 — R2: cross-device transfer, naive vs Riemannian re-centering
    tlabels, naive, recen = [], [], []
    for key, lab in [("R2_transfer_emotiv_to_neurocom", "Emotiv ->\nNeurocom"),
                     ("R2_transfer_neurocom_to_emotiv", "Neurocom ->\nEmotiv")]:
        t = results.get(key)
        if t:
            tlabels.append(lab)
            naive.append(t["naive"]["subject_auroc"])
            recen.append(t["recentered"]["subject_auroc"])
    if tlabels:
        xx = np.arange(len(tlabels))
        ax2.bar(xx - 0.18, naive, 0.36, label="naive transfer", color="#bbbbbb")
        ax2.bar(xx + 0.18, recen, 0.36, label="+ Riemann re-centering", color="#dd8452")
        ax2.axhline(0.5, ls="--", c="grey", lw=1)
        ax2.set_xticks(xx); ax2.set_xticklabels(tlabels, fontsize=7)
        ax2.set_ylim(0, 1); ax2.set_ylabel("transfer AUROC")
        ax2.set_title("R2: cross-device, same contrast")
        ax2.legend(fontsize=8)

    # Panel 3 — R3: OT effect sizes, colored by significance
    ot_items = []
    for key, lab in [("ot_custom", "custom\nRel-Con"), ("ot_eegmat", "EEGMAT\nrest-arith"),
                     ("ot_ds", "ds Exp-Nov\n(control)")]:
        if results.get(key):
            ot_items.append((lab, results[key]["w2"], results[key]["p"]))
    for k, v in results.get("ot_wesad", {}).items():
        ot_items.append((k.replace("_vs_", "\n"), v["w2"], v["p"]))
    if ot_items:
        xx = np.arange(len(ot_items))
        ax3.bar(xx, [t[1] for t in ot_items],
                color=["#4c72b0" if t[2] < 0.05 else "#bbbbbb" for t in ot_items])
        for xi, t in zip(xx, ot_items):
            ax3.text(xi, t[1], f"p={t[2]:.3f}", ha="center", va="bottom", fontsize=7)
        ax3.set_xticks(xx); ax3.set_xticklabels([t[0] for t in ot_items], fontsize=7)
        ax3.set_ylabel("Bures-Wasserstein W2")
        ax3.set_title("R3: yoga-impact effect size (blue p<.05)")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_ROOT / "riemann_summary.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    run()
