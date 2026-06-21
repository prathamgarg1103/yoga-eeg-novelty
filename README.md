# Quantifying Yoga's Impact with Machine Learning & Multimodal Data

Turning *"yoga's impact"* into a single calibrated **Relaxation Index (0–100)** — a
**Composite Yoga Index** that fuses an **autonomic** axis (heart-rate variability, EDA) and a
**cortical/EEG** axis (frontal alpha/theta), validated **leak-free across subjects, datasets,
and EEG devices**. Not another within-dataset classifier — an interpretable, physiologically
grounded relaxation score.

> The algorithms are deliberately simple and grounded. The contribution is the **framing +
> rigor**: a cross-modal calibrated index, device-confound control, and contamination-aware,
> subject/recording-independent validation.

## Results (all subject/recording-independent)

**Autonomic axis — WESAD (15 subjects, LOSO)**
- Calm vs stress: **AUROC 0.980**, acc 0.942, permutation **p = 0.007**
- Stress vs non-stress (benchmark): **AUROC 0.973**
- Relaxation score by condition: **meditation 94 > baseline 92 ≫ stress 20** (correct physiology)

**Cortical/EEG axis — custom yoga data (19 de-duplicated recordings)**
- Leave-one-recording-out: recording-level **acc 0.895 / AUROC 0.911**; permutation **p = 0.005**
- **Relaxation Index: Relaxed 75 vs Concentrated 32**

**Cortical/EEG axis — ds001787 external (24 subjects, LOSO)**
- Expert meditators show higher relative alpha (**0.48 vs 0.35**) — corroborates the relaxation signature
- Expert-vs-novice classification is at chance (honest null from 2 frontal channels)

**Flagship — "The Geometry of Calm" (Riemannian manifold + optimal transport)**
- **Device-portable relaxation signature**: a model trained on one EEG device classifies the
  same state on a *different* device — **Neurocom→Emotiv AUROC 0.88**, Emotiv→Neurocom 0.63
  (well above chance), using PhysioNet **EEGMAT** as the second device.
- **Optimal-transport yoga-impact effect size** (Bures-Wasserstein W₂, group-permutation):
  large & significant on real state shifts (WESAD meditation-vs-stress W₂ 6.95, custom
  Relaxed-vs-Concentrated 1.72, all **p = 0.002**) and correctly **null** on a between-person
  *trait* control (expert-vs-novice, p = 0.90) — convergent + discriminant validity.
- Riemannian representation is *competitive* with tuned classical features (band-specific
  connectivity 0.753 vs 0.747 on the 19-ch montage) — a principled descriptor, reported
  honestly as a tie/marginal-edge, not an inflated win.
- Full method, math, code map, and the runnable worked example:
  [`yoga_impact/docs/FLAGSHIP_Geometry_of_Calm.md`](yoga_impact/docs/FLAGSHIP_Geometry_of_Calm.md).

## Why this is a rebuild

The prior work reported ~98% accuracy that was an **artifact of contaminated data**: 120 custom
files collapse to **19 unique signals**, with the *same recordings copied across different
subject folders* and one Relaxed recording mislabeled as Concentrated — so the old validation
leaked identical signals across train/test with wrong labels. This project detects that
(`audit.py` + `integrity.py`), cleans it (`clean_custom.py`), and validates honestly.

## Structure & usage

```
yoga_impact/        # the package (see yoga_impact/README.md for module map)
requirements.txt    # numpy scipy pandas scikit-learn mne neurokit2 lightgbm matplotlib
```

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python -m yoga_impact.run_all          # full pipeline -> yoga_impact/outputs/
```

Datasets: WESAD and ds001787 are public downloads; the custom yoga EEG is private; **EEGMAT**
(PhysioNet) is **auto-downloaded** on first run by `ds_eegmat.ensure_downloaded()`. Place the
manual datasets under `datasets/` (paths in `yoga_impact/config.py`).

### The three pipelines (what to show)
1. **Baseline rebuild** — classical band power + calibrated classifier, leak-free LOSO/LORO.
2. **Novelty layer (C1–C5)** — connectivity, 64-ch brain networks, conformal uncertainty,
   leak-free personalization, deep-vs-classical (`REPORT.md` §5–9).
3. **Flagship — Geometry of Calm (R1–R3)** — Riemannian manifold + optimal transport;
   device-portable signature + significance-tested effect size (`REPORT.md` §10).

## Honesty notes

- The custom yoga set is tiny (19 recordings, ~4 subjects) → a **case study**; the quantitative
  backbone is WESAD + ds001787 (+ EEGMAT for cross-device).
- Neutral-state recordings (different EEG device) are excluded from the headline task to avoid a
  device-vs-brain-state confound.
- Cross-device transfer of *absolute-scale* classical features is unreliable — but the **Riemannian
  (relative/geometric) representation does transfer across devices** (flagship R2), which is the
  point of the geometric approach.
