# Quantifying Yoga's Impact with Machine Learning
## A Calibrated, Leak-Free, Cross-Device Index of Physiological Relaxation — Final Compilation

**Author:** Pratham Garg · Thapar Institute of Engineering & Technology
**Problem statement:** *"Harnessing Machine Learning and Multimodal Data to Quantify Yoga's Impact"*
**Builds on (and corrects):** Kansal et al. (2023), *"A Novel Deep Learning Approach to Quantify the Yoga-Induced Cognitive Changes"*
**Evolved from my own earlier capstone:** MESMI — *Multimodal Explainable Stress & Meditation Intelligence* (CPG-26)

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [The Problem, and Why It Is Hard](#2-the-problem-and-why-it-is-hard)
3. [The Research Journey — How I Got Here](#3-the-research-journey--how-i-got-here)
4. [The Two Projects and All Their Pipelines](#4-the-two-projects-and-all-their-pipelines)
5. [The Core Idea — the Composite Yoga Index](#5-the-core-idea--the-composite-yoga-index)
6. [Headline Results](#6-headline-results)
7. [Novelty Layer — Five Leak-Free Contributions (C1–C5)](#7-novelty-layer--five-leak-free-contributions-c1c5)
8. [The Flagship — The Geometry of Calm (R1–R3)](#8-the-flagship--the-geometry-of-calm-r1r3)
9. [Comparison with the Previous Published Paper](#9-comparison-with-the-previous-published-paper)
10. [Why Every Claim Is Defensible](#10-why-every-claim-is-defensible)
11. [Honest Limitations](#11-honest-limitations)
12. [How to Reproduce Everything](#12-how-to-reproduce-everything)
13. [Conclusion](#13-conclusion)

---

## 1. Executive Summary

I built a system that measures **how relaxed a person is** from their physiology — a single,
calibrated **Relaxation Index (0–100)** — and validated it the *honest* way: every number comes
from data where **the model has never seen the test person before** (leave-one-subject-out).

The headline result is an **AUROC of 0.98** for separating a calm state from a stress state on
independent subjects, with a permutation test (**p = 0.007**) proving it is not luck. On top of that
strong-but-honest baseline I added two layers of genuine novelty: (1) a set of brain-connectivity and
uncertainty contributions, and (2) a **flagship "Geometry of Calm" method** that represents each
moment of physiology as a point on a Riemannian manifold and measures yoga's effect with **optimal
transport**. Crucially, I also ran a controlled experiment that **reproduces and explains the prior
paper's 96.5%** — showing it was a validation artifact, not a real signal.

**The contribution is not a bigger accuracy number; it is rigor, a new method, cross-device
portability, calibrated uncertainty, and honesty.**

---

## 2. The Problem, and Why It Is Hard

Yoga and meditation are claimed to reduce stress and sharpen focus. The scientific question is:
**can we put a defensible number on that change from physiological signals (EEG + autonomic body
signals)?** Three traps make this genuinely hard — and most prior work ignores all three.

1. **Subject leakage.** EEG/physiology windows recorded seconds apart are nearly identical. A random
   train/test split puts the *same person's* windows on both sides, so the model memorises the
   person, not the brain state. This single mistake inflates accuracy by ~25 percentage points.
2. **Device confounds.** Different EEG headsets (Emotiv, Muse, Neurocom, BioSemi) have different
   channels and sampling rates. A model can secretly learn *"which device"* instead of *"which
   mental state."*
3. **Tiny, contaminated real-world data.** The custom yoga EEG set *looked* like 120 recordings but
   is actually only **19 unique signals**, with files copied across different "subjects" and at
   least one mislabelled state.

**The entire project is designed to defeat these three traps — and that design is the contribution.**

---

## 3. The Research Journey — How I Got Here

This project was not a straight line. I tried an approach, found its faults, and shifted. The numbers
exposed each fault and justified each pivot.

```
Prior work (96.5%, but leaky)  →  MESMI (deep ties/loses to RF)  →  Clean rebuild (120→19, honest 0.98)
        →  Novelty layer (+0.022 / +0.178, deep still loses)  →  Flagship (cross-device 0.88, OT p=0.002)
```

### Act 1 — The prior work that "looked perfect"
The published baseline reached **LSTM 96.5%** (Muse, 70/30 split) and **81.6–82.3%** on 4 yoga
subjects. It looked solved — but it used **only EEG**, and 98% on 4 noisy subjects felt too good to
be true. **Fault sensed:** one signal can't capture a whole-body state, and that number likely hides
a validation problem. → *Build something richer.*

### Act 2 — MESMI: the ambitious version that humbled me
I built a multimodal, explainable estimator: **CEEMDAN** + nonlinear dynamics (PSR/RQA/SampEn) +
per-signal **BiLSTM** + **cross-attention** + continuous biomarker outputs + **contrastive SHAP**.
**Fault, with numbers:** under honest subject-grouped folds, the deep+attention stack only **tied or
lost to a plain SVM/RF** (binary 0.881 vs SVM 0.892; four-class 0.604 vs RF 0.676), and my headline
novelty — cross-attention — was statistically no better than averaging (CCC 0.736 vs 0.732,
**p = 0.625**). Meditation-depth regression failed (CCC ≈ 0). **Lesson:** I was over-engineering;
*rigor beats sophistication.* → *Strip back and audit the data.*

### Act 3 — The clean rebuild: when I finally checked the data
Simplest defensible model + a data audit first. **What I found:** **120 files = 19 unique signals**,
copied across "subjects," one mislabelled, plus a device confound — the smoking gun behind the prior
98%. **What the rebuild produced:** autonomic axis **AUROC 0.980** (p=0.007), cortical axis **0.933**
(p=0.005) — honest. **New fault:** honesty cost me my headline number; a plain baseline wasn't novel
yet. → *Add novelty that survives honest validation.*

### Act 4 — The novelty layer: ideas that survive scrutiny
Five leak-free extensions: connectivity (**+0.022**), 64-ch brain networks (0.667, p=0.138), calibrated
uncertainty (**ECE 0.021**, ρ=1.000), honest personalization (**+0.178**, p=0.002), and deep-vs-classical
(**deep still loses**). **Fault:** gains were incremental, and a controlled test proved accuracy was
ceiling-bound — same model, honest CV **0.718** vs leaky 70/30 **0.984** vs within-subject **1.000**.
Chasing a bigger number means *re-introducing the leak.* → *Need a new method and a new question.*

### Act 5 — The flagship: The Geometry of Calm
Riemannian-manifold representation (relative → device-portable) + **optimal-transport** effect size +
a second EEG device for a true cross-device test. **Where it landed:** a **device-portable signature**
(Neurocom→Emotiv **0.878**, control correctly fails at 0.438), an **effect-size with validity** (OT
significant on **4/5** state contrasts at p=0.002, correctly null on the trait control at p=0.902),
and the honest accuracy ceiling stated plainly (FBCSP 0.787). **This was the destination — every
earlier fault pushed me here.**

---

## 4. The Two Projects and All Their Pipelines

The `E:\yogic` workspace holds **two complete projects** and **sixteen pipelines**. The thread
connecting them: every pipeline is validated so the model never sees the test person during training.

### Project A — `yoga_impact` (the current headline project, 10 pipelines)
Orchestrated by `run_all.py`; shared leak-free engine in `modeling.py`; config in `config.py`.

| # | Pipeline | Purpose | Honest result |
|---|---|---|---|
| A1 | Data-integrity audit (`audit.py`, `integrity.py`, `clean_custom.py`) | Detect & fix contamination | 120 → **19 unique signals** |
| A2 | Autonomic axis (`wesad_model.py`) | Relaxation from body signals (WESAD, LOSO) | **AUROC 0.98**, p=0.007 |
| A3 | Cortical/EEG axis (`eeg_model_custom.py`, `eeg_connectivity.py`) | Relaxed vs Concentrated (custom, LORO) | **0.933**, p=0.005 |
| A4 | External validation (`ds_meditation.py`) | Does it generalise? (ds001787) | alpha 0.48 > 0.35; honest null |
| A5 | 64-ch brain network (`ds_network.py`) | Network-level analysis | 0.667; beta path p=0.002 |
| A6 | Uncertainty + validity (`uncertainty.py`) | Calibration & convergent validity | ECE 0.021; ρ=1.000 |
| A7 | Personalization (`personalization.py`) | Leak-free subject adaptation | **+0.178**, p=0.002 |
| A8 | Deep vs classical (`deep_baseline.py`) | Does deep win? | No — 0.811/0.900 < 0.933 |
| A9 | **Flagship Riemann+OT** (`riemann.py`, `riemann_spd.py`) | New method + effect size | cross-device **0.88**; OT p=0.002 |
| A10 | Composite report (`composite.py`) | Assemble the Index | Relaxation Index (0–100) |

### Project B — `mesmi` (the earlier ambitious capstone, 6 pipelines)
Orchestrated by `run_all.py`: `extract_features → train_eval → explain`. Evaluation is honest
(subject-wise GroupKFold) — the deep stack simply didn't out-earn its complexity.

| # | Pipeline | Purpose | Honest result |
|---|---|---|---|
| B1 | Feature extraction | CEEMDAN → PSR/RQA/SampEn | 5 mod × 4 IMF × 11 feat |
| B2 | Classification (BiLSTM + cross-attention) | Multimodal state decoding | ties/loses to SVM/RF |
| B3 | Continuous regression | Biomarker scores | stress CCC **0.76** > RF 0.61 |
| B4 | Meditation-EEG multimodal | depth/MW/tiredness | CCC ≈ 0 (honest failure) |
| B5 | Explainability | contrastive SHAP + attention maps | per-modality reasoning |
| B6 | Reporting | figures/tables/artifacts | results pack |

---

## 5. The Core Idea — the Composite Yoga Index

I fuse two **independently validated** relaxation axes into one interpretable score:

```
Yoga Impact Index = w · EEG_relaxation + (1 − w) · autonomic_relaxation     (both modalities)
Yoga Impact Index =        EEG_relaxation                                   (EEG-only data)
```

Each axis outputs a **calibrated probability P(relaxed) × 100** → the **Relaxation Index (0–100)**,
the single headline number. Three delivered views: (1) the Relaxation Index, (2) Relaxed-vs-
Concentrated (binary), and (3) a pre/post or condition-contrast change ("how much did yoga move you").

**Coverage:** four datasets, three EEG devices + autonomic — WESAD (autonomic), ds001787 (BioSemi
64-ch), EEGMAT (Neurocom 19-ch), and the custom yoga set (Emotiv F7/F8).

---

## 6. Headline Results

*All numbers are strictly subject/recording independent.*

**Autonomic axis (WESAD, leave-one-subject-out) — the headline**
- Calm vs stress: **AUROC 0.980**, balanced accuracy 0.920; permutation **p = 0.007**.
- Relaxation score by condition: **meditation 94 > baseline 92 ≫ stress 20** — correct physiology.

**Cortical/EEG axis (custom yoga, leave-one-recording-out) — honest case study (19 recordings)**
- Relaxed vs Concentrated: **recording-level AUROC 0.933**; permutation **p = 0.005**.
- Relaxation Index: **Relaxed ≈ 77 vs Concentrated ≈ 35.**

**The honest EEG ceiling.** On the larger EEGMAT set the true subject-independent ceiling is **~0.79**
(FBCSP). I report this rather than a leak-inflated number — the headline accuracy lives in the
autonomic axis (0.98).

---

## 7. Novelty Layer — Five Leak-Free Contributions (C1–C5)

Each contribution reuses the same leak-free cross-validation, and **each reports its result even when
my novel method loses.**

| # | Contribution | Result | Verdict |
|---|---|---|---|
| **C1** | Brain-connectivity features (coherence/PLV, PAC, 1/f slope) | AUROC **0.911 → 0.933 (+0.022)** | Real, honest gain |
| **C2** | 64-channel brain network (wPLI graphs) | **0.667**; beta path-length **p = 0.002** | Honestly non-sig overall (p=0.138); age-confound flagged |
| **C3** | Uncertainty + convergent validity (conformal, ECE) | **ECE 0.021**; Spearman **ρ = 1.000** | Calibrated & consistent |
| **C4** | Leak-free personalization | **+0.178 AUROC (0.797→0.975)**, p=0.002, 13/15 | Prior paper's idea, without its leak |
| **C5** | Deep vs calibrated-classical | LSTM 0.811 / MLP 0.900 < classical **0.933** | Deep does **not** win |

**Why C4 and C5 matter:** they re-test the prior paper's *own claims* under honest conditions —
personalization genuinely helps (without the leak); deep learning genuinely does not. That is a
direct, evidence-based correction of the prior work.

---

## 8. The Flagship — The Geometry of Calm (R1–R3)

Two ideas no prior yoga-EEG paper has used.

**The representation.** Each window of physiology becomes a **symmetric positive-definite (SPD)
covariance matrix** — a point on a curved Riemannian manifold (built from scratch in pure NumPy/
SciPy). The reference mean is fit on **training windows only** (leak-free), then test points are
projected into the tangent space and classified. Because the representation is *relative/geometric*,
it factors out per-device scale — making it **device-portable**.

**The effect size.** Instead of one accuracy number, I quantify the *distributional shift* between two
states using the **Bures–Wasserstein W₂ distance** (decomposed into mean-shift + shape-change) with a
group-permutation p-value — directly answering *how much, and how significantly, does yoga move you.*

**R1 — manifold vs strongest classical (identical leak-free folds):** custom **0.933 = 0.933 (tie)**;
EEGMAT 19-ch band-specific connectivity **0.753 vs 0.747** — competitive, reported honestly.

**R2 — true cross-device transfer (the special result):** Neurocom→Emotiv **AUROC 0.878**,
Emotiv→Neurocom 0.630 — both well above chance. Negative control (state→trait transfer) **0.438 —
correctly fails.** A model trained on one EEG device classifies the same state on a *different*
device; the prior single-device pipeline never could.

**R3 — optimal-transport effect size with significance:**

| Contrast | Type | W₂ | p-value |
|---|---|---|---|
| WESAD meditation vs stress | autonomic state | 6.95 | **0.002** |
| WESAD baseline vs stress | autonomic state | 5.90 | **0.002** |
| WESAD meditation vs baseline | autonomic state | 4.22 | **0.002** |
| Custom Relaxed vs Concentrated | cortical state | 1.72 | **0.002** |
| EEGMAT rest vs arithmetic | cortical state (frontal-only) | 0.46 | 0.379 *(honestly underpowered)* |
| **ds001787 expert vs novice** | **trait — negative control** | 0.97 | **0.902 (correctly null)** |

**Validity: 4/5 state contrasts significant, 0/1 trait significant** — exactly the convergent +
discriminant pattern a real measurement should show.

**Bonus:** a novel information-geometric **temporal-stability** finding — focused states are *more*
temporally stable than rest (**p = 0.011**), consistent across two devices.

---

## 9. Comparison with the Previous Published Paper

**The prior paper** (Kansal et al., 2023): a two-phase LSTM, 3-class EEG states, **70/30 random
split** → **96.5%**; 4-subject yoga **81.6–82.3%**. No leave-one-subject-out, no calibration, no
permutation test, no effect size, no negative control.

**The decisive issue — leakage, proved directly.** Same features, same classifier, only the CV scheme
changes:

| Validation scheme | Window AUROC |
|---|---|
| **Honest** — leave-one-subject-out | **0.718** |
| **Leaky** — random 70/30 split (the prior protocol) | **0.984** |
| **Within-subject** split | **1.000** |

The **+0.27 gap is the inflation.** Their 96.5% lives in the 0.984 row — reproducible, and not real.

**Side-by-side:**

| Dimension | Prior paper (2023) | This project |
|---|---|---|
| Validation | 70/30 random split | LOSO/LORO + calibration + **permutation tests** |
| Headline | 96.5% (leaky) | **AUROC 0.98** autonomic (LOSO, p=0.007) — honest |
| Data integrity | contaminated, used as-is | **audited & cleaned** (120 → 19) |
| Datasets / devices | 1 Muse set + 1-device yoga | **4 datasets, 3 EEG devices + autonomic** |
| Method | two-phase LSTM | **Riemannian manifold + optimal transport**; FBCSP; calibrated classical |
| Calibration | none | **ECE 0.021** |
| Effect size / significance | none | **Bures–Wasserstein W₂**, p=0.002 |
| Negative controls | none | **trait null p=0.90**; deep-vs-classical |
| Cross-device transfer | none | **0.88** (Neurocom→Emotiv) |
| Reproducibility | not released | one command, EEGMAT auto-downloaded |

**The honest framing:** *I do not claim to beat 96.5% with a bigger number — I explain it. Under their
protocol my pipeline also reaches ~0.98; under honest validation the truth is ~0.72–0.80. The gap is
the artifact, and I demonstrate it in-study. Fixing the protocol is itself the contribution.*

---

## 10. Why Every Claim Is Defensible

- ✅ **Leak-free everywhere** — leave-one-subject/recording-out; no person in both train and test.
- ✅ **In-fold calibration** — probabilities are honest (ECE 0.021); the score means what it says.
- ✅ **Permutation tests** — headline results carry a null distribution and a p-value (0.002–0.007).
- ✅ **Negative controls** — trait contrasts that *should* be null *are* null (p = 0.90); state→trait
  transfer correctly fails (0.438).
- ✅ **Four datasets, three EEG devices + autonomic** — not one dataset, not one device.
- ✅ **Data integrity audited** — contamination detected and fixed (120 → 19); device confound removed.
- ✅ **Reproducible** — one command (`run_all`); EEGMAT auto-downloaded; deterministic seeds.
- ✅ **Honest negatives reported** — the manifold ties on small data, EEGMAT OT is underpowered, deep
  learning loses — all stated, none hidden or tuned away.

---

## 11. Honest Limitations

- The custom yoga EEG set is tiny (**19 recordings, ~4 subjects**) and is treated as a **case study**,
  not a headline; the quantitative weight is carried by the public datasets.
- No dataset has a **co-recorded** EEG + autonomic subject during yoga, so the Composite Index fusion
  is demonstrated axis-by-axis; full fusion activates the moment a wearable is recorded alongside EEG.
- No dataset has a **continuous within-session yoga transition**, so a literal moment-by-moment dose-
  response curve isn't computable; I deliver per-person displacement and temporal trajectories instead.
- The honest EEG ceiling (~0.79) is genuinely lower than leaky literature numbers — **by design**.
  Pushing it higher would mean reintroducing the exact leak this project exists to remove.

---

## 12. How to Reproduce Everything

```bash
# Project A — yoga_impact (current)
python -m yoga_impact.run_all
#   subset: python -m yoga_impact.run_all --stages wesad custom riemann report
#   outputs: yoga_impact/outputs/REPORT.md, summary.json, riemann_metrics.json, plots

# Project B — MESMI (earlier)
cd mesmi && python run_all.py        # extract -> train_eval -> explain
#   outputs: mesmi/training/ (metrics.json, SHAP, attention maps)
```

WESAD and ds001787 are public downloads; the custom yoga EEG is private; **EEGMAT** is
auto-downloaded on first run.

---

## 13. Conclusion

The prior paper answered *"can a model score high on yoga EEG?"* with a number inflated by a 70/30
split on contaminated data. **This project answers *"what is real after you remove the leak?"*** — a
calibrated **0.98 autonomic** relaxation index, an honest **~0.79** EEG ceiling, a **novel device-
portable Riemannian + optimal-transport** method with convergent/discriminant validity, and a
controlled demonstration of exactly **how the 96.5% was manufactured.**

The real result is the journey itself — *ambition → humility → rigor → genuine novelty* — and at
every turn, the numbers are what forced the next step.

---

*Companion documents: `RESEARCH_JOURNEY_FOR_TEACHER.md` · `ALL_PIPELINES_FOR_TEACHER.md` ·
`PRIOR_PAPER_COMPARISON_FOR_TEACHER.md` · `yoga_impact/docs/PROJECT_EXPLAINER_FOR_TEACHER.md` ·
`yoga_impact/docs/FLAGSHIP_Geometry_of_Calm.md`.*
