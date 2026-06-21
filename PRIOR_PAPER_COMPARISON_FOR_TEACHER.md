# Comparison with the Previous Published Paper
### How this work relates to — and corrects — Kansal et al. (2023)

**Prior paper:** Kansal, Mannat, Bhardwaj, Anand, Kaur, Kansal —
*"A Novel Deep Learning Approach to Quantify the Yoga-Induced Cognitive Changes"* (Thapar; same
custom yoga EEG data). All facts below are taken from the paper's own text
(`yoga_impact/docs/_prior_paper.txt`) and from this project's saved results — nothing estimated.

---

## 1. What the prior paper actually did

- **Task:** 3-class mental state — *neutral / concentration / relaxation* — from EEG frequency-band
  features.
- **Method:** a **two-phase LSTM** (64 units, batch 32, learning rate 0.001, 15 epochs, early
  stopping; fine-tuned with LR 0.0005 for 10 epochs).
- **Phase I:** trained on a **public Muse dataset** (channels TP9/AF7/AF8/TP10), with a **70/30
  random split** → GRU 94.6%, **LSTM 96.5%**.
- **Phase II:** a small yoga cohort (4 subjects), pre/post yoga; "real-time validation" **81.6%**;
  LSTM fine-tuned on subject S1 → **82.3%** (vs GRU 68.5%, SVM 59.7%).
- **Validation:** a 70/30 split only. **No** leave-one-subject-out, **no** calibration, **no**
  permutation test, **no** effect size, **no** negative control. The paper itself acknowledges a
  "small homogeneous sample size."

---

## 2. The one issue that changes everything: validation leakage

Their headline 96.5% comes from a **70/30 random split**. EEG windows recorded seconds apart are
almost identical, so a random split puts windows from the *same person* in both train and test —
the model memorises the **person**, not the **brain state**. On the custom yoga data this is worse,
because the set is contaminated: an audit shows **120 files collapse to 19 unique signals**, with
several byte-identical signals copied across *different* subject folders. Under a random split,
identical signals appear on both sides.

### I proved this directly — the most defensible result in the project
Using the **same features and the same classifier**, changing **only the cross-validation scheme**
(`riemann._leakage_forensics_eegmat`, on the EEGMAT dataset):

| Validation scheme | Window AUROC |
|---|---|
| **Honest** — leave-one-subject-out | **0.718** |
| **Leaky** — random 70/30 window split (the prior paper's protocol) | **0.984** |
| **Within-subject** split | **1.000** |

The **+0.27 gap is the inflation**. The prior paper's 96.5% lives in the **0.984 row** — it is
reproducible, and it is not a real measure of generalisation.

---

## 3. Side-by-side comparison

| Dimension | Prior paper (Kansal et al., 2023) | This project (yoga_impact) |
|---|---|---|
| **Validation** | 70/30 random split (not subject-independent) | leave-one-**subject/recording**-out + in-fold calibration + **permutation tests** |
| **Headline number** | 96.5% (leaky split, public Muse set) | **AUROC 0.98** autonomic (WESAD, LOSO, **p = 0.007**) — honest |
| **EEG state decoding** | 82.3% (1 subject, fine-tuned, 70/30) | **0.79–0.93** honest LOSO/LORO, reported as the true ceiling |
| **Data integrity** | contaminated set used as-is | contamination **detected, audited, de-duplicated** (120 → 19) |
| **Datasets / devices** | 1 Muse set + 4-subject yoga (1 device) | **4 datasets, 3 EEG devices + autonomic** (WESAD, ds001787, EEGMAT, custom) |
| **Method** | two-phase LSTM (transfer learning) | **Riemannian SPD-manifold + optimal transport**; FBCSP; calibrated classical (deep tested, loses) |
| **Calibration** | none | calibrated probabilities (**ECE 0.021**, conformal coverage ≈ nominal) |
| **Effect size / significance** | none (accuracy only) | **Bures–Wasserstein W₂** with group-permutation p (state shifts **p = 0.002**) |
| **Negative controls** | none | **trait null** (expert-vs-novice **p = 0.90**) → discriminant validity; deep-vs-classical |
| **Cross-device transfer** | none | relaxation manifold **transfers across devices** (Neurocom→Emotiv **0.88**) |
| **Novel scientific readout** | none | per-subject responder map; information-geometric **temporal-stability** finding (**p = 0.011**) |
| **Reproducibility** | not released | one command (`run_all`), EEGMAT auto-downloaded, deterministic |

---

## 4. Their three claims, re-tested honestly

I took the prior paper's own claims and re-ran them under leak-free conditions. This is the
strongest part for a viva — it is a direct, evidence-based engagement with the prior work.

1. **"Deep learning (LSTM) is the right tool."**
   → Under identical leak-free folds, **deep loses**: LSTM 0.811, MLP 0.900 vs calibrated classical
   **0.933** (custom) and RandomForest **0.980** (WESAD). My earlier MESMI project showed the same
   thing independently (its BiLSTM+attention tied/lost to SVM/RF). Their LSTM edge was a validation
   artifact, not a modelling advantage.

2. **"Subject-specific adaptation improves accuracy."**
   → True — but they delivered it *with* a leak (fine-tuning on a scored subject). I reproduced the
   benefit **without** the leak (anchoring on a disjoint state): **+0.178 AUROC (0.797 → 0.975),
   Wilcoxon p = 0.002, helped 13/15**. I validate their *idea* while fixing their *method*.

3. **"High accuracy proves the effect."**
   → Replaced with something stronger: a **significance-tested effect size** (Bures–Wasserstein W₂)
   that quantifies *how much* yoga moves you, with **convergent validity** (4/5 state contrasts
   significant) and **discriminant validity** (the trait contrast correctly null, p = 0.90).

---

## 5. How to state this honestly (the examiner-proof framing)

> I do **not** claim to beat 96.5% with a bigger number — I **explain** it. Under the prior paper's
> 70/30 protocol my own pipeline also reaches ~0.98; under subject-independent validation the
> honest truth is ~0.72–0.80. **The gap is the artifact, and I demonstrate it in-study.** The
> genuine, defensible advances are **rigor** (leak-free, multi-dataset, permutation-tested), **method
> novelty** (Riemannian geometry + optimal transport), **calibration**, **cross-device portability**,
> and **honest negative results** — none of which the prior paper has. The prior paper is a
> reasonable first effort that openly notes its small sample; the issue is the *validation
> protocol*, and **fixing that protocol is itself the contribution.**

---

## 6. One-line summary

The prior paper answers *"can a model score high on yoga EEG?"* with a number inflated by a 70/30
split on contaminated data. **This project answers *"what is real after you remove the leak?"*** — a
calibrated **0.98 autonomic** relaxation index, an honest **~0.79** EEG ceiling, a **novel device-
portable Riemannian + optimal-transport** method with convergent/discriminant validity, and a
controlled demonstration of exactly **how the 96.5% was manufactured**.

---

*A more technical version of this comparison is in `yoga_impact/docs/COMPARISON_vs_prior_paper.md`.
The full project map (both projects, all 16 pipelines) is in `ALL_PIPELINES_FOR_TEACHER.md`.*

> **One factual caution before you present:** the prior paper's text lists its Phase II cohort as
> *4 subjects (2M/2F)*, while one of the older project documents says "4 male." Use "4-subject
> cohort" unless you confirm the split from the PDF.
