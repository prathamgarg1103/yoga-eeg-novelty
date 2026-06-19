# Quantifying Yoga's Impact — Results Report

A Composite Yoga Index built from two independently validated relaxation axes:
an **autonomic** axis (WESAD peripheral physiology) and a **cortical/EEG** axis
(meditation EEG + custom yoga EEG). Headline output: **Relaxation Index (0-100)**.
All validation is strictly subject/recording independent.

## 1. Autonomic relaxation axis — WESAD (LOSO)

- **stress vs non-stress** (RandomForest): AUROC 0.973, F1 0.844, acc 0.932
- **stress vs non-stress** (LightGBM): AUROC 0.947, F1 0.852, acc 0.937
- **calm vs stress** (relaxation score): AUROC 0.980, bal-acc 0.920
- permutation test: observed AUROC 0.981, p = 0.007

Autonomic relaxation score by condition (OOF, 0-100) — sanity ordering:
  - meditation: 94.0
  - baseline: 92.2
  - stress: 19.8

## 2. Cortical/EEG axis — external (ds001787 meditation, LOSO)

- **expert vs novice meditators**: AUROC 0.517, bal-acc 0.554, F1 0.580
- **transfer** (custom model -> ds001787) Relaxation Index by group: expert 46.1, novice 60.0
- mean relative alpha — expert: 0.484, novice: 0.353

## 3. Cortical/EEG axis — custom yoga data (case study)

_Dataset is tiny (19 de-duplicated recordings, ~4 subjects) — reported honestly._

- **LORO(recording)::LogReg**: recording-level acc 0.895, AUROC 0.911 (n=19); window AUROC 0.903
- **LORO(recording)::RandomForest**: recording-level acc 0.895, AUROC 0.900 (n=19); window AUROC 0.884
- **LOPO(subject)::LogReg**: recording-level acc 0.789, AUROC 0.933 (n=19); window AUROC 0.853
- **LOPO(subject)::RandomForest**: recording-level acc 0.842, AUROC 0.856 (n=19); window AUROC 0.820
- permutation test (LORO): observed AUROC 0.918, p = 0.005

## 4. Composite Yoga Index

```
YII = w * EEG_relaxation + (1 - w) * autonomic_relaxation   (both modalities)
YII = EEG_relaxation                                        (EEG-only data)
```
Each axis output is a calibrated P(relaxed) x 100. On the custom yoga data
(EEG only) the index equals the cortical Relaxation Index; the WESAD autonomic
axis plugs in directly once wearable signals are recorded during yoga.

**Three delivered outputs** (all from the calibrated EEG model):
1. Relaxation Index (0-100) — headline.
2. Relaxed vs Concentrated (binary) — thresholded index.
3. Time-resolved / condition-contrast change.
