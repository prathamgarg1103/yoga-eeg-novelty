# yoga_impact — Quantifying Yoga's Impact from Multimodal Physiological Data

A clean, leak-free rebuild. The goal is a **Relaxation Index (0–100)** plus a
**Composite Yoga Index** that fuses two independently validated relaxation axes:

| Axis | Dataset | What it measures | Validation |
|------|---------|------------------|------------|
| Autonomic (body) | WESAD (15 subj, peripheral physiology) | calm vs stress (HRV/EDA/EMG/Resp) | leave-one-subject-out |
| Cortical (EEG) | ds001787 (24 subj) + custom yoga (4 subj) | frontal alpha/theta relaxation | LOSO / LORO + transfer |

`Composite Yoga Index = w·EEG_relaxation + (1−w)·autonomic_relaxation`, and reduces to
the EEG axis for EEG-only recordings (the current custom data).

## Why a rebuild was necessary

The prior pipelines failed for concrete, fixable reasons, confirmed by `audit.py` /
`integrity.py`:

- **The custom yoga data was contaminated.** 120 files collapse to **19 unique signals**;
  4 signals were copied across *different subject* folders, and one Relaxed recording was
  filed under Concentrated. The reported "97.8%" was an artifact of duplicated signals
  leaking across train/test with corrupted labels.
- **No single dataset is multimodal+co-recorded**, so the old 5-modality cross-attention
  fusion was untrainable; and the old continuous targets (meditation "stage", recovery
  index) had no ground truth.

The rebuild fixes all of these: de-duplication, title-based relabelling, strictly
subject/recording-independent validation, real labels only, and simple calibrated models.

## Pipeline / modules

```
config.py            paths, bands, windowing, seeds
io_custom.py         device-aware loaders (Emotiv F7/F8, Muse AF7/AF8) -> Recording
audit.py             inventory + leakage/label flags  -> outputs/manifest.csv
integrity.py         signal-hash duplicate detection   -> outputs/integrity.csv
clean_custom.py      de-dup + title labels (19 recs)   -> outputs/custom_clean.csv
eeg_features.py      device-agnostic EEG features (band power, ratios, asymmetry, Hjorth)
modeling.py          leak-free group CV + calibration + permutation test
wesad.py             WESAD loading + autonomic features (neurokit2 + scipy)
wesad_model.py       WESAD LOSO: stress-vs-rest + calm-vs-stress relaxation score
eeg_model_custom.py  custom yoga EEG: Relaxed-vs-Concentrated -> Relaxation Index
ds_meditation.py     ds001787 external validation (A7/B10 = F7/F8) + transfer
ds_eegmat.py         PhysioNet EEGMAT loader (Neurocom F7/F8) — 2nd EEG device
riemann_spd.py       SPD-manifold primitives (Fréchet mean, tangent map, Bures-Wasserstein)
riemann.py           FLAGSHIP: Riemannian manifold + optimal-transport effect size (R1–R3)
composite.py         Composite Yoga Index + outputs/REPORT.md
run_all.py           orchestrator
```

## Flagship pipeline — "The Geometry of Calm"

The third, flagship pipeline represents each window by the *geometry* of its multiband
channel covariance (a point on the SPD manifold) and quantifies yoga's impact as the
optimal-transport (Bures-Wasserstein) distance between state distributions. It adds a
**device-portable** relaxation signature (a model trained on one EEG device works on
another: Neurocom→Emotiv AUROC 0.88) and a significance-tested effect size with convergent
(state) and discriminant (trait) validity. Full explanation —
including the math, the code map, the results, and the honest limitations — is in
[`docs/FLAGSHIP_Geometry_of_Calm.md`](docs/FLAGSHIP_Geometry_of_Calm.md).

## How to run

```bash
# from E:\yogic, using the project venv
.venv/Scripts/python.exe -m yoga_impact.run_all
# or individual stages:
.venv/Scripts/python.exe -m yoga_impact.run_all --stages wesad custom ds report
```

Outputs (metrics JSON, score CSVs, plots, REPORT.md) land in `yoga_impact/outputs/`.

## Honesty notes

- The custom yoga set is tiny (19 recordings, ~4 subjects, Moyukh has only Relaxed). Its
  numbers are a **case study**, not a headline accuracy claim. The quantitative backbone is
  WESAD + ds001787.
- Neutral (Muse) is excluded from the headline EEG task because device ≡ class would make a
  classifier detect the *device* rather than the brain state.
