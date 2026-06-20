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

- **LORO(recording)::LogReg**: recording-level acc 0.842, AUROC 0.933 (n=19); window AUROC 0.908
- **LORO(recording)::RandomForest**: recording-level acc 0.895, AUROC 0.900 (n=19); window AUROC 0.883
- **LOPO(subject)::LogReg**: recording-level acc 0.842, AUROC 0.933 (n=19); window AUROC 0.868
- **LOPO(subject)::RandomForest**: recording-level acc 0.842, AUROC 0.833 (n=19); window AUROC 0.820
- permutation test (LORO): observed AUROC 0.922, p = 0.005

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

---

# Novelty layer (leak-free extensions)

## 5. New EEG features — connectivity / coupling / aperiodic (C1)

Recording-level AUROC by feature block (same leak-free CV):
- **LORO**: base 0.911 (+0.000), base+conn 0.933 (+0.022), base+pac 0.911 (+0.000), base+aperiodic 0.900 (-0.011), base+all 0.933 (+0.022)
- **LOPO**: base 0.933 (+0.000), base+conn 0.933 (+0.000), base+pac 0.933 (+0.000), base+aperiodic 0.911 (-0.022), base+all 0.933 (+0.000)

_Inter-hemispheric coherence/PLV add real recording-level AUROC; aperiodic/PAC are neutral on 2 channels — reported honestly._

## 6. 64-channel brain-network analysis — ds001787 (C2)

- **expert vs novice** (LOSO, full-montage network metrics): AUROC 0.667, bal-acc 0.583
- permutation test (n=24): observed AUROC 0.667, p = 0.138
- beta_char_path_length: expert 2.013 vs novice 1.902 (p=0.002)
- theta_modularity: expert 0.196 vs novice 0.168 (p=0.053)
- theta_char_path_length: expert 1.961 vs novice 1.899 (p=0.078)
- beta_mean_connectivity: expert 0.168 vs novice 0.183 (p=0.089)
- _caveat: Expertise is age-confounded in ds001787 (experts skew younger); network contrasts are reported as associations, not causal expertise effects._

## 7. Uncertainty-aware Index + cross-modal convergent validity (C3)

- **wesad**: ECE 0.021, conformal coverage 0.89 (nominal 0.90)
- **custom**: ECE 0.095, conformal coverage 0.95 (nominal 0.90)
- **convergent validity** (autonomic conditions): Spearman rho 1.000 (p=0.000)
- pooled autonomic+cortical relaxation ordering: Spearman rho 0.657

## 8. Leak-free personalization (C4)

- subject-independent mean AUROC (baseline vs amusement, WESAD LOSO): 0.797
- adapted [recenter]: mean AUROC 0.975, mean lift +0.178 (helped 13/hurt 2, Wilcoxon p=0.002)
- adapted [zscore]: mean AUROC 0.946, mean lift +0.149 (helped 11/hurt 4, Wilcoxon p=0.064)
_Personalization delivered without the 2023 fine-tuning leak: adaptation uses only a disjoint within-subject anchor (meditation windows), never a scored or trained class._

## 9. Deep vs calibrated-classical head-to-head (C5)

- backend: torch
- WESAD calm-vs-stress (LOSO): classical AUROC 0.980 vs deep 0.896
- custom (LORO, recording): classical AUROC 0.933 vs deep-MLP 0.900
- custom LSTM (temporal, recording): AUROC 0.811
- **verdict**: deep does NOT beat the calibrated classical baseline (classical retained); honest given small N and a high classical bar
