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

Datasets are **not** included (large / licensed / private): WESAD and ds001787 are public
downloads; the custom yoga EEG is private. Place them under `datasets/` (paths in
`yoga_impact/config.py`).

## Honesty notes

- The custom yoga set is tiny (19 recordings, ~4 subjects) → a **case study**; the quantitative
  backbone is WESAD + ds001787.
- Neutral-state recordings (different EEG device) are excluded from the headline task to avoid a
  device-vs-brain-state confound.
- Cross-device transfer of absolute-scale features is unreliable and flagged as a limitation.
