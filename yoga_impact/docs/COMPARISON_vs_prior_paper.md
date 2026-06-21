# This project vs. the prior published paper

**Prior paper:** Kansal, Mannat, Bhardwaj, Anand, Kaur, Kansal — *"A Novel Deep Learning
Approach to Quantify the Yoga-Induced Cognitive Changes"* (Thapar; same custom yoga EEG data).
Facts below are quoted from the PDF (`docs/2023_YOGA_EEG_NAACJournal_FINAL_12Dec2025.pdf`).

## What the prior paper actually did
- **Task:** 3-class mental state (neutral / concentration / relaxation) from EEG frequency-band features.
- **Phase I:** trained on a *public* Muse dataset ("EEG Brainwave Dataset for Mental State
  Classification", channels TP9/AF7/AF8/TP10) with a **70/30 train–test split** → GRU **94.6%**,
  **LSTM 96.5%**.
- **Phase II:** 4 male subjects (avg age 21), pre/post yoga; "real-time validation" **81.6%**;
  fine-tuned LSTM on subject S1 **82.3%** (vs GRU 68.5%, SVM 59.7%).
- **Validation:** a 70/30 split. **No subject-independent (leave-one-subject-out) evaluation,
  no calibration, no permutation test, no effect size, no negative control.**
- Acknowledged limitation: "small homogeneous sample size".

## The core problem: the headline number is a validation artifact
A **70/30 random split** puts windows from the *same recording/subject* in both train and test.
EEG windows seconds apart are near-identical, so the model can memorise the person/recording
rather than the brain state. On the project's own custom data this is compounded by
**contamination** (audit: 120 files collapse to **19 unique signals**, several copied across
*different* subject folders) — under a random split, byte-identical signals appear in train and
test, which alone can produce ~97%.

**We proved this directly** (`riemann._leakage_forensics_eegmat`, EEGMAT, *same features and
classifier, only the CV scheme changes*):

| Validation scheme | window AUROC |
|---|---|
| honest — leave-one-subject-out | **0.718** |
| leaky — random window split (the prior paper's 70/30 protocol) | **0.984** |
| within-subject split | **1.000** |

The +0.27 gap is the inflation. The prior paper's 96.5% lives in the **0.98 row** — reproducible,
and not real.

## Head-to-head

| Dimension | Prior paper (Kansal et al.) | This project |
|---|---|---|
| **Validation** | 70/30 random split (not subject-independent) | leave-one-**subject**/recording-out + in-fold calibration + group **permutation tests** |
| **Headline number** | 96.5% (public Muse set, leaky split); 81.6–82.3% yoga (4 subj) | **autonomic AUROC 0.98** (WESAD, LOSO, perm p=0.007) — honest |
| **EEG state decoding** | 82.3% (1 subject, fine-tuned, 70/30) | **0.79–0.80** honest LOSO (EEGMAT, FBCSP/ensemble), reported as the true ceiling |
| **Data integrity** | contaminated set used as-is | contamination **detected, audited, de-duplicated** (120→19), device confound removed |
| **Datasets / devices** | 1 public Muse set + 4-subject yoga (1 device) | WESAD + ds001787 + EEGMAT + custom — **4 datasets, 3 EEG devices + autonomic** |
| **Method** | LSTM (2-phase, transfer learning) | **Riemannian SPD-manifold + optimal-transport** effect size; FBCSP; calibrated classical (deep tested, loses) |
| **Calibration** | none | calibrated probabilities (**ECE 0.021**, conformal coverage ≈ nominal) |
| **Effect size / significance** | none (accuracy only) | **Bures–Wasserstein W₂** with group-permutation p (state shifts p=0.002) |
| **Negative controls** | none | **trait null** (expert-vs-novice p=0.90) → discriminant validity; deep-vs-classical (TOST) |
| **Cross-device transfer** | none | relaxation manifold **transfers across devices** (Neurocom→Emotiv AUROC 0.88) |
| **Novel scientific readout** | none | per-subject responder map; information-geometric **temporal-stability** finding (p=0.011) |
| **Reproducibility** | not released | one command (`run_all`), EEGMAT auto-downloaded, deterministic |

## How to state this honestly in the write-up
- We do **not** claim to beat 96.5% with a bigger number — we **explain** it: under the prior
  paper's 70/30 protocol our own pipeline also reaches ~0.98, and under subject-independent CV
  the honest number is ~0.72–0.80. The gap is the artifact, demonstrated in-study.
- The genuine, defensible advances are **rigor** (leak-free, multi-dataset, permutation-tested),
  **method novelty** (Riemannian geometry + optimal transport), **calibration**, **cross-device
  portability**, and **honest negative results** — none of which the prior paper has.
- The prior paper is a reasonable first effort that openly notes its small sample; the issue is
  the *validation protocol*, which inflates the headline. Fixing that protocol is itself the
  contribution.

## One-line summary
The prior paper answers *"can a model score high on yoga EEG?"* with a number inflated by a
70/30 split on contaminated data. This project answers *"what is real after you remove the
leak?"* — a calibrated **0.98 autonomic** relaxation index, an honest **~0.79** EEG ceiling, a
novel device-portable Riemannian + optimal-transport method, and a controlled demonstration of
exactly how the 96.5% was manufactured.
