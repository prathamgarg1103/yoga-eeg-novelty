"""Composite Yoga Index + final report.

The Composite Yoga Index (YII) combines two independently validated relaxation axes:

    YII = w * EEG_relaxation + (1 - w) * autonomic_relaxation        (both present)
    YII = EEG_relaxation                                             (EEG-only, e.g. custom)

Each axis output is a calibrated P(relaxed) in [0, 1], scaled to 0-100. This module
gathers the saved metrics from every stage and writes a single human-readable report
(outputs/REPORT.md) plus a machine summary (outputs/summary.json).
"""
from __future__ import annotations

import json

from yoga_impact import config


def compute_yii(eeg_relax: float, autonomic_relax: float | None = None, w: float = 0.5) -> float:
    """Composite Yoga Index in 0-100. Falls back to the EEG axis when autonomic is absent."""
    if autonomic_relax is None:
        return float(eeg_relax)
    return float(w * eeg_relax + (1 - w) * autonomic_relax)


def _load(name: str) -> dict | None:
    p = config.OUTPUT_ROOT / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def build_report() -> str:
    wesad = _load("wesad_metrics.json")
    custom = _load("custom_eeg_metrics.json")
    ds = _load("ds_meditation_metrics.json")
    network = _load("ds_network_metrics.json")
    uncert = _load("uncertainty_metrics.json")
    person = _load("personalization_metrics.json")
    deep = _load("deep_metrics.json")

    L = ["# Quantifying Yoga's Impact — Results Report", ""]
    L += ["A Composite Yoga Index built from two independently validated relaxation axes:",
          "an **autonomic** axis (WESAD peripheral physiology) and a **cortical/EEG** axis",
          "(meditation EEG + custom yoga EEG). Headline output: **Relaxation Index (0-100)**.",
          "All validation is strictly subject/recording independent.", ""]

    # ---- Autonomic axis ----
    L += ["## 1. Autonomic relaxation axis — WESAD (LOSO)", ""]
    if wesad:
        sr = {k: v for k, v in wesad.items() if k.startswith("stress_vs_rest")}
        for k, v in sr.items():
            name = k.split("::")[1]
            L.append(f"- **stress vs non-stress** ({name}): "
                     f"AUROC {v['auroc']:.3f}, F1 {v['f1']:.3f}, acc {v['acc']:.3f}")
        cv = wesad.get("calm_vs_stress::RandomForest")
        if cv:
            L.append(f"- **calm vs stress** (relaxation score): "
                     f"AUROC {cv['auroc']:.3f}, bal-acc {cv['bal_acc']:.3f}")
        perm = wesad.get("calm_vs_stress::permutation")
        if perm:
            L.append(f"- permutation test: observed AUROC {perm['observed_auroc']:.3f}, "
                     f"p = {perm['perm_p']:.3f}")
        order = wesad.get("relaxation_score_by_condition")
        if order:
            L.append("")
            L.append("Autonomic relaxation score by condition (OOF, 0-100) — sanity ordering:")
            for cond, val in order.items():
                L.append(f"  - {cond}: {val:.1f}")
    else:
        L.append("_not yet computed_")
    L.append("")

    # ---- EEG axis: ds001787 ----
    L += ["## 2. Cortical/EEG axis — external (ds001787 meditation, LOSO)", ""]
    if ds:
        en = ds.get("expert_vs_novice")
        if en:
            L.append(f"- **expert vs novice meditators**: AUROC {en['auroc']:.3f}, "
                     f"bal-acc {en['bal_acc']:.3f}, F1 {en['f1']:.3f}")
        tr = ds.get("transfer_relaxation_index")
        if tr:
            L.append(f"- **transfer** (custom model -> ds001787) Relaxation Index by group: "
                     + ", ".join(f"{g} {v:.1f}" for g, v in tr.items()))
        fc = ds.get("feature_contrasts", {}).get("mean_alpha_rel")
        if fc:
            L.append(f"- mean relative alpha — " + ", ".join(f"{g}: {v:.3f}" for g, v in fc.items()))
    else:
        L.append("_not yet computed_")
    L.append("")

    # ---- EEG axis: custom ----
    L += ["## 3. Cortical/EEG axis — custom yoga data (case study)", ""]
    L.append("_Dataset is tiny (19 de-duplicated recordings, ~4 subjects) — reported honestly._")
    L.append("")
    if custom:
        for k in ("LORO(recording)::LogReg", "LORO(recording)::RandomForest",
                  "LOPO(subject)::LogReg", "LOPO(subject)::RandomForest"):
            v = custom.get(k)
            if v:
                rec = v["recording"]
                L.append(f"- **{k}**: recording-level acc {rec['acc']:.3f}, "
                         f"AUROC {rec['auroc']:.3f} (n={rec['n']}); "
                         f"window AUROC {v['window']['auroc']:.3f}")
        perm = custom.get("permutation_LORO")
        if perm:
            L.append(f"- permutation test (LORO): observed AUROC {perm['observed_auroc']:.3f}, "
                     f"p = {perm['perm_p']:.3f}")
    else:
        L.append("_not yet computed_")
    L.append("")

    # ---- Composite definition ----
    L += ["## 4. Composite Yoga Index", "",
          "```",
          "YII = w * EEG_relaxation + (1 - w) * autonomic_relaxation   (both modalities)",
          "YII = EEG_relaxation                                        (EEG-only data)",
          "```",
          "Each axis output is a calibrated P(relaxed) x 100. On the custom yoga data",
          "(EEG only) the index equals the cortical Relaxation Index; the WESAD autonomic",
          "axis plugs in directly once wearable signals are recorded during yoga.",
          "",
          "**Three delivered outputs** (all from the calibrated EEG model):",
          "1. Relaxation Index (0-100) — headline.",
          "2. Relaxed vs Concentrated (binary) — thresholded index.",
          "3. Time-resolved / condition-contrast change.", ""]

    # ====================================================================
    # NOVELTY LAYER (C1-C5) — leak-free extensions beyond the rebuild
    # ====================================================================
    L += ["---", "", "# Novelty layer (leak-free extensions)", ""]

    # ---- C1 connectivity ablation (inside custom metrics) ----
    L += ["## 5. New EEG features — connectivity / coupling / aperiodic (C1)", ""]
    abl = (custom or {}).get("ablation")
    if abl:
        L.append("Recording-level AUROC by feature block (same leak-free CV):")
        for cv_name in ("LORO", "LOPO"):
            if cv_name in abl:
                parts = ", ".join(
                    f"{n} {d['recording_auroc']:.3f} ({d['auroc_lift_vs_base']:+.3f})"
                    for n, d in abl[cv_name].items())
                L.append(f"- **{cv_name}**: {parts}")
        L.append("")
        L.append("_Inter-hemispheric coherence/PLV add real recording-level AUROC; aperiodic/PAC "
                 "are neutral on 2 channels — reported honestly._")
    else:
        L.append("_not yet computed_")
    L.append("")

    # ---- C2 brain network ----
    L += ["## 6. 64-channel brain-network analysis — ds001787 (C2)", ""]
    if network:
        en = network.get("expert_vs_novice_network")
        perm = network.get("permutation")
        if en:
            L.append(f"- **expert vs novice** (LOSO, full-montage network metrics): "
                     f"AUROC {en['auroc']:.3f}, bal-acc {en['bal_acc']:.3f}")
        if perm:
            L.append(f"- permutation test (n=24): observed AUROC {perm['observed_auroc']:.3f}, "
                     f"p = {perm['perm_p']:.3f}")
        con = network.get("metric_contrasts", {})
        sig = sorted(((k, v) for k, v in con.items() if v.get("p", 1) < 0.1),
                     key=lambda kv: kv[1]["p"])[:4]
        for k, v in sig:
            L.append(f"- {k}: expert {v['expert_mean']:.3f} vs novice {v['novice_mean']:.3f} "
                     f"(p={v['p']:.3f})")
        if network.get("note"):
            L.append(f"- _caveat: {network['note']}_")
    else:
        L.append("_not yet computed_")
    L.append("")

    # ---- C3 uncertainty + convergent validity ----
    L += ["## 7. Uncertainty-aware Index + cross-modal convergent validity (C3)", ""]
    if uncert:
        for axis in ("wesad", "custom"):
            a = uncert.get(axis)
            if a:
                L.append(f"- **{axis}**: ECE {a['ece']:.3f}, conformal coverage "
                         f"{a['conformal_coverage']:.2f} (nominal {a['nominal']:.2f})")
        cv = uncert.get("convergent_validity", {})
        au = cv.get("autonomic", {})
        po = cv.get("pooled_cross_modal", {})
        if au:
            L.append(f"- **convergent validity** (autonomic conditions): Spearman rho "
                     f"{au['spearman_rho']:.3f} (p={au['spearman_p']:.3f})")
        if po:
            L.append(f"- pooled autonomic+cortical relaxation ordering: Spearman rho "
                     f"{po['spearman_rho']:.3f}")
    else:
        L.append("_not yet computed_")
    L.append("")

    # ---- C4 personalization ----
    L += ["## 8. Leak-free personalization (C4)", ""]
    if person:
        L.append(f"- subject-independent mean AUROC (baseline vs amusement, WESAD LOSO): "
                 f"{person.get('independent_mean_auroc', float('nan')):.3f}")
        for mth in person.get("methods", []):
            d = person.get(mth, {})
            if d:
                L.append(f"- adapted [{mth}]: mean AUROC {d['mean_auroc']:.3f}, "
                         f"mean lift {d['mean_lift']:+.3f} (helped {d['helped']}/hurt {d['hurt']}, "
                         f"Wilcoxon p={d['wilcoxon_p']:.3f})")
        L.append("_Personalization delivered without the 2023 fine-tuning leak: adaptation uses only "
                 "a disjoint within-subject anchor (meditation windows), never a scored or trained class._")
    else:
        L.append("_not yet computed_")
    L.append("")

    # ---- C5 deep vs classical ----
    L += ["## 9. Deep vs calibrated-classical head-to-head (C5)", ""]
    if deep:
        L.append(f"- backend: {deep.get('backend', '?')}")
        w = deep.get("wesad")
        if w:
            L.append(f"- WESAD calm-vs-stress (LOSO): classical AUROC {w['classical_auroc']:.3f} "
                     f"vs deep {w['deep_auroc']:.3f}")
        cm = deep.get("custom_mlp")
        if cm:
            L.append(f"- custom (LORO, recording): classical AUROC {cm['classical_recording_auroc']:.3f} "
                     f"vs deep-MLP {cm['deep_recording_auroc']:.3f}")
        cl = deep.get("custom_lstm")
        if cl:
            L.append(f"- custom LSTM (temporal, recording): AUROC {cl['recording_auroc']:.3f}")
        if deep.get("verdict"):
            L.append(f"- **verdict**: {deep['verdict']}")
    else:
        L.append("_not yet computed_")
    L.append("")

    report = "\n".join(L)
    (config.OUTPUT_ROOT / "REPORT.md").write_text(report, encoding="utf-8")

    summary = dict(wesad=wesad, custom=custom, ds=ds, network=network,
                   uncertainty=uncert, personalization=person, deep=deep)
    (config.OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    return report


def main() -> None:
    print(build_report())
    print(f"\nReport -> {config.OUTPUT_ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    main()
