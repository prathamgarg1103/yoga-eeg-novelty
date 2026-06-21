# Every Pipeline in the `yogic` Workspace — A Complete Map
### For presentation — Pratham Garg, Thapar Institute of Engineering & Technology

This document is a full scan of the `E:\yogic` workspace. It contains **two complete research
projects**, and within them **sixteen distinct pipelines**. Everything below is taken directly
from the code and the saved result files — no estimates.

| Project | Folder | What it is | Status |
|---|---|---|---|
| **A — yoga_impact** | `yoga_impact/` | The current, disciplined **Composite Yoga Index** — a calibrated, leak-free, cross-device relaxation index | **Headline project** |
| **B — MESMI** | `mesmi/` | An earlier, ambitious **multimodal + explainable** stress/meditation estimator (Capstone CPG-26) | Predecessor; lessons folded into A |

**The single thread connecting both:** every pipeline is validated so the model **never sees the
test person during training** (leave-one-subject-out or subject-wise grouping). That discipline is
the spine of the whole workspace.

---

## PART 0 — The big picture (all 16 pipelines at a glance)

### Project A — yoga_impact (10 pipelines, orchestrated by `yoga_impact/run_all.py`)

| # | Pipeline | Key files | One-line result |
|---|---|---|---|
| A1 | **Data-integrity / contamination audit** | `audit.py`, `integrity.py`, `clean_custom.py` | 120 custom files → **19 unique signals**; cleaned |
| A2 | **Autonomic relaxation axis (WESAD)** | `wesad.py`, `wesad_model.py` | calm-vs-stress **AUROC 0.98** (LOSO, p=0.007) |
| A3 | **Cortical/EEG custom-yoga axis** | `io_custom.py`, `eeg_features.py`, `eeg_connectivity.py`, `eeg_model_custom.py` | Relaxed-vs-Concentrated **AUROC 0.933** (LORO, p=0.005) |
| A4 | **External EEG validation (ds001787)** | `ds_meditation.py` | expert alpha 0.48 > novice 0.35; honest null on 2 ch |
| A5 | **64-channel brain-network (C2)** | `ds_network.py`, `graph_metrics.py` | network AUROC 0.667; beta path-length p=0.002 |
| A6 | **Uncertainty + convergent validity (C3)** | `uncertainty.py` | **ECE 0.021**, conformal ≈ nominal, Spearman ρ=1.000 |
| A7 | **Leak-free personalization (C4)** | `personalization.py` | **+0.178 AUROC** (0.797→0.975), p=0.002 |
| A8 | **Deep-vs-classical (C5)** | `deep_baseline.py` | deep **loses**: LSTM 0.811 / MLP 0.900 < classical 0.933 |
| A9 | **FLAGSHIP — Geometry of Calm (R1–R3)** | `riemann.py`, `riemann_spd.py`, `ds_eegmat.py` | device-portable signature **0.88**; OT effect size p=0.002 |
| A10 | **Composite Index + report** | `composite.py`, `modeling.py` (shared) | assembles the **Relaxation Index (0–100)** + `REPORT.md` |

### Project B — MESMI (6 pipelines, orchestrated by `mesmi/run_all.py`)

| # | Pipeline | Key files | One-line result |
|---|---|---|---|
| B1 | **Nonlinear feature extraction** | `extract_features.py`, `extract_emd.py`, `features.py` | CEEMDAN → PSR+RQA+SampEn; 5 mod × 4 IMF × 11 feat |
| B2 | **Classification (BiLSTM + cross-attention)** | `train_eval.py`, `model.py`, `experiments.py` | binary ~0.88, 4-class ~0.60 — **ties/loses to SVM/RF** |
| B3 | **Continuous biomarker regression** | `train_eval_reg.py`, `regression_experiments.py`, `make_targets_wesad.py` | stress-score **CCC 0.76** (neural > RF 0.61) |
| B4 | **Meditation-EEG multimodal (ds001787)** | `extract_features_med.py`, `med_labels.py` | depth regression ≈ chance — **honest failure** |
| B5 | **Explainability (contrastive SHAP)** | `explain.py`, `explain_reg.py` | per-modality SHAP + 5×5 attention maps |
| B6 | **Reporting / artifacts** | `make_report.py`, `make_research_report.py`, `render_artifacts.py` | figures, tables, results pack |

---

# PART 1 — Project A: yoga_impact (the headline project)

**What it is:** turns "yoga's impact" into one calibrated **Relaxation Index (0–100)** — a
**Composite Yoga Index** = `w·EEG_relaxation + (1−w)·autonomic_relaxation`. Two independently
validated axes (cortical/EEG + autonomic), four datasets, three EEG devices.

**Shared engine (used by every A-pipeline):** `modeling.py` — the leak-free cross-validation core
(leave-one-group-out + **in-fold probability calibration** + permutation tests + recording-level
aggregation). `config.py` centralises every path, band, and window so nothing is hard-coded.

Run everything: `python -m yoga_impact.run_all` → artifacts land in `yoga_impact/outputs/`.

---

### A1 — Data-integrity / contamination audit
- **Files:** `audit.py` (inventory), `integrity.py` (signal hashing), `clean_custom.py` (rebuild).
- **Why it exists:** the custom yoga EEG looked like 120 recordings; signal-hashing proved it is
  only **19 unique signals**, with files copied across *different* subject folders, one Relaxed
  recording mislabelled Concentrated, and a device confound (Emotiv vs Muse).
- **What it does:** de-duplicates by signal hash, trusts the device-internal title as ground
  truth, keeps Emotiv Relaxed/Concentrated only, sets `recording_id = signal hash` so windows can
  never leak across CV folds.
- **Result:** a clean 19-recording set → `outputs/custom_clean.csv`. This single step explains the
  prior "97.8–96.5%": duplication + leakage + bad labels.
- **Say to the teacher:** *"Before modelling I proved the data was contaminated and fixed it — the
  prior high accuracy was partly counting copies of the same signal in train and test."*

### A2 — Autonomic relaxation axis (WESAD) — the headline number
- **Files:** `wesad.py` (loads chest ECG/EDA/EMG/Resp/Temp @700 Hz), `wesad_model.py` (model).
- **Method:** HRV / EDA / respiration / EMG features → RandomForest & LightGBM → **leave-one-
  subject-out** with in-fold calibration.
- **Results:** calm-vs-stress **AUROC 0.980** (bal-acc 0.920), **permutation p = 0.007**;
  stress-vs-rest AUROC 0.973. Relaxation score by condition: **meditation 94 > baseline 92 ≫
  stress 20** — exactly correct physiology. Top features: HRV mean-NN, HRV mean-HR, respiration-
  rate variability, EDA.
- **Say to the teacher:** *"This is the rigorous, honest headline: 0.98 AUROC on subjects the
  model has never seen, with a permutation test proving it isn't luck."*

### A3 — Cortical/EEG custom-yoga axis (case study) + C1 ablation
- **Files:** `io_custom.py` (Emotiv F7/F8 loader), `eeg_features.py` (band power + ratios +
  asymmetry), `eeg_connectivity.py` (**C1**: coherence/PLV, theta→beta PAC, 1/f slope),
  `eeg_model_custom.py`.
- **Method:** 8 s windows (50% overlap) → features → calibrated LogReg/RF → **leave-one-recording-
  out** and **leave-one-subject-out**; `recording_id` keyed on the signal hash (no window leak).
- **Results:** Relaxed-vs-Concentrated **recording-level AUROC 0.933** (acc 0.842; RF acc 0.895),
  **permutation p = 0.005**; Relaxation Index Relaxed ≈ 77 vs Concentrated ≈ 35.
  **C1 ablation:** inter-hemispheric connectivity lifts AUROC **0.911 → 0.933 (+0.022)**; PAC and
  aperiodic are neutral on 2 channels — reported honestly.
- **Say to the teacher:** *"Tiny dataset, treated as a case study — but still honestly above
  chance, and my new connectivity feature adds real signal."*

### A4 — External EEG validation (ds001787 meditation)
- **Files:** `ds_meditation.py` (BioSemi 64-ch BDF; F7/F8 mapped to A7/B10).
- **Purpose:** does the relaxation signature generalise to an independent meditation dataset?
- **Results:** expert meditators show higher relative alpha (**0.48 vs 0.35**) — corroborates the
  signature; expert-vs-novice *classification* sits near chance (AUROC 0.52), reported as an
  **honest null** from only 2 frontal channels.
- **Say to the teacher:** *"The physiological direction replicates on outside data, and I don't
  oversell the weak 2-channel classifier."*

### A5 — 64-channel brain-network analysis (C2)
- **Files:** `ds_network.py`, `graph_metrics.py`.
- **Method:** full BioSemi-64 montage → weighted phase-lag-index (wPLI) connectivity graphs →
  networkx graph-theory metrics (path length, modularity, efficiency, clustering) → LOSO.
- **Results:** expert-vs-novice network **AUROC 0.667** (beats the 0.52 frontal baseline) but
  **permutation p = 0.138** (n = 24) — *honestly non-significant overall*; the sharp single
  contrast is **beta characteristic path length expert > novice, p = 0.002**. **Age-confound
  flagged** (experts skew younger) — reported as association, not causation.
- **Say to the teacher:** *"A richer brain-network view; I report the one significant contrast and
  the honest non-significant overall test, and I flag the age confound myself."*

### A6 — Uncertainty + cross-modal convergent validity (C3)
- **Files:** `uncertainty.py`.
- **Method:** leave-group-out **conformal prediction** (coverage), **ECE / reliability**
  calibration, and Spearman rank validity across conditions.
- **Results:** WESAD **ECE 0.021**, conformal coverage 0.89 ≈ nominal 0.90; **convergent validity
  Spearman ρ = 1.000** (meditation > baseline > amusement > stress); pooled cross-modal ρ = 0.66.
- **Say to the teacher:** *"The Index isn't just accurate — its probabilities are calibrated and
  it orders mental states in the physiologically correct sequence."*

### A7 — Leak-free personalization (C4)
- **Files:** `personalization.py`.
- **Method:** subject-adaptive recentering using a **disjoint within-subject anchor** (meditation
  windows) — never a scored or trained class, so no leakage.
- **Results:** WESAD baseline-vs-amusement **0.797 → 0.975 AUROC, +0.178**, helped **13 of 15**,
  Wilcoxon **p = 0.002**.
- **Say to the teacher:** *"The prior paper's 'personalization' leaked. I deliver the same benefit
  honestly — adaptation uses a different state as the anchor, never the answer."*

### A8 — Deep-vs-classical head-to-head (C5)
- **Files:** `deep_baseline.py` (PyTorch LSTM + MLP).
- **Method:** identical leak-free folds; deep models vs the calibrated classical baseline.
- **Results:** WESAD classical **0.980** vs deep 0.896; custom classical **0.933** vs MLP 0.900,
  LSTM 0.811. **Deep does not win.**
- **Say to the teacher:** *"Under honest validation, deep learning doesn't beat a well-calibrated
  classical model on this data — so I keep the simpler model."*

### A9 — FLAGSHIP: The Geometry of Calm (R1–R3)
- **Files:** `riemann_spd.py` (SPD-manifold math in pure NumPy/SciPy — matrix sqrt/log/exp,
  Fréchet mean, tangent projection, Bures–Wasserstein), `riemann.py` (the pipeline),
  `ds_eegmat.py` (second device: PhysioNet **EEGMAT**, Neurocom 19-ch, auto-downloaded).
- **Method:** each window → augmented multiband covariance → an SPD point on a Riemannian
  manifold → tangent-space projection at the **per-fold geometric mean** (leak-free).
- **R1 — manifold vs strongest classical:** custom **0.933 = 0.933 (tie)**; EEGMAT 19-ch band-
  specific connectivity **0.753 vs 0.747** (marginal edge) — competitive, reported honestly.
- **R2 — true cross-device transfer (the special result):** Neurocom→Emotiv **AUROC 0.878**,
  Emotiv→Neurocom 0.630 — both well above chance → **device-portable relaxation signature**.
  Negative control state→trait transfer = 0.438 (**correctly fails**).
- **R3 — optimal-transport effect size:** Bures–Wasserstein W₂ (mean-shift + shape) with group-
  permutation p — WESAD meditation-vs-stress **6.95**, custom Relaxed-vs-Concentrated **1.72**
  (all **p = 0.002**); trait control expert-vs-novice **p = 0.902 (correctly null)**. Validity:
  **4/5 state contrasts significant, 0/1 trait** = convergent + discriminant.
- **Bonus forensics & extras (in `riemann.py`):** controlled leakage forensics (same model:
  honest 0.718 → leaky 0.984 → within-subject 1.000), FBCSP ceiling **0.787**, and a novel
  **information-geometric temporal-stability** finding (focus more stable than rest, p = 0.011).
- **Say to the teacher:** *"This is the publishable core: a new method (Riemannian geometry +
  optimal transport) that transfers across EEG devices and measures how much yoga moves you, with
  significance and negative controls."*

### A10 — Composite Index + report
- **Files:** `composite.py` (assembles the index and writes `outputs/REPORT.md`), `modeling.py`
  (shared CV/calibration engine).
- **Output:** the **Relaxation Index (0–100)** plus the binary and pre/post views, and the full
  results report (Sections 1–10).

---

# PART 2 — Project B: MESMI (the earlier, ambitious design)

**What it is:** *Multimodal Explainable Stress & Meditation Intelligence* — not a binary
classifier but a **physiologically interpretable biomarker estimator** producing continuous
scores. It is genuine prior work of mine (Capstone CPG-26), and the lessons from it directly
shaped the discipline of yoga_impact.

**Pipeline shape (`mesmi/run_all.py`):** `extract_features` → `train_eval` → `explain`.
**Honesty note:** MESMI's evaluation *is* subject-grouped (subject-wise GroupKFold, 15 subjects →
3 held out per fold) — so it is not leaky. The honest finding is simply that its heavy deep stack
**did not out-earn its complexity** on classification; that, plus the lack of calibration /
permutation tests / effect sizes, is exactly what yoga_impact then added.

---

### B1 — Nonlinear feature extraction
- **Files:** `extract_features.py`, `extract_emd.py`, `features.py` (config in `mesmi/config.py`).
- **Method (Novelties 1–2):** **CEEMDAN** decomposition (10 noise trials, first 4 IMFs, Hurst-
  exponent gating) → per-IMF **tripartite nonlinear features**: Phase-Space Reconstruction (area,
  density, spread), Recurrence Quantification Analysis (%REC, DET, LAM, ENTR), and Sample Entropy
  (scales 1–2) — **11 features per IMF**.
- **Tensor:** 5 modalities × 4 IMFs × 11 features per 30 s window.
- **Say to the teacher:** *"I replaced plain EMD with CEEMDAN to kill mode-mixing, then described
  each component three ways — geometry, determinism, complexity."*

### B2 — Classification (BiLSTM + cross-attention)
- **Files:** `train_eval.py`, `model.py` (architecture), `experiments.py` (ablation).
- **Architecture (Novelty 3):** a 2-layer **BiLSTM** per modality (hidden 48 → 96-d embedding) →
  4-head **cross-attention** (5×5) → **learned-query attention pooling** (the interpretable
  "which modality mattered" vector) → classification head.
- **Modalities:** WESAD chest = ECG/EDA/EMG/Resp/Temp (no EEG in WESAD).
- **Results (subject-wise GroupKFold, WESAD):** binary stress-vs-rest ~**0.88** (SVM 0.892, RF
  0.886, attention 0.881); four-class ~**0.60** (RF 0.676, SVM 0.662, MESMI 0.604). **The deep /
  attention model ties or loses to a plain SVM/RF.**
- **Say to the teacher:** *"Honest ablation: cross-attention didn't beat simple baselines on
  classification — which is why I later rebuilt simpler and proved this rigorously in yoga_impact."*

### B3 — Continuous biomarker regression (Novelty 4)
- **Files:** `train_eval_reg.py`, `regression_experiments.py`, `regression_metrics.py`,
  `make_targets_wesad.py`.
- **Method:** continuous outputs instead of labels — a **stress score (0–100)** mapped to STAI-6
  anxiety; loss = Huber + Concordance-Correlation (CCC) loss; targets z-scored on train-fold only.
- **Results (WESAD stress score, GroupKFold):** neural fusion **CCC ≈ 0.74–0.76** (mean-pool
  0.757, attention 0.736, concat 0.732) **beats** Ridge 0.512 and RF 0.605 — but attention ≈
  concat ≈ mean (Δccc −0.008, Wilcoxon p = 0.625), i.e. the *attention* mechanism itself is not
  the source of the gain.
- **Say to the teacher:** *"On the harder regression task the neural model genuinely helps (CCC
  0.76 vs 0.61), but the fancy attention isn't what's doing the work — an important honest nuance."*

### B4 — Meditation-EEG multimodal pipeline (ds001787)
- **Files:** `extract_features_med.py`, `med_labels.py`.
- **Method:** the same 5-modality model reused on OpenNeuro meditation data (EEG/GSR/Resp/Plet/
  Temp @256→64 Hz); targets = depth / mind-wandering / tiredness from behavioural probes.
- **Results (24 subjects, 141 windows):** regression CCC ≈ **0 / negative across all models** — an
  **honest failure**: the task is too hard for this small, noisy set.
- **Say to the teacher:** *"I report the negative too — meditation-depth regression didn't work on
  141 windows, and I say so plainly."*

### B5 — Explainability (Novelty 5)
- **Files:** `explain.py`, `explain_reg.py`.
- **Method:** SHAP (DeepExplainer) per modality/feature, plus **contrastive SHAP** = SHAP(stress)
  − SHAP(meditation) ("why stress *and not* meditation"), plus the saved 5×5 cross-attention
  weight heatmap (which modality attended to which).
- **Say to the teacher:** *"Beyond accuracy, the model explains its reasoning the way a clinician
  would — why this state, and why not the other."*

### B6 — Reporting / artifacts
- **Files:** `make_report.py`, `make_research_report.py`, `make_results_pack.py`,
  `render_artifacts.py`, `render_reg.py`.
- **Output:** confusion matrices, attention heatmaps, contrastive-SHAP plots, metrics JSON,
  research-style report (in `mesmi/training/` and `mesmi/docs/`).

---

# PART 3 — How the two projects connect (the story to tell)

1. **MESMI came first** — ambitious: CEEMDAN, tripartite nonlinear features, BiLSTM + cross-
   attention, continuous biomarkers, contrastive explainability.
2. **MESMI was measured honestly** — and its heavy deep stack tied/lost to SVM/RF on
   classification, while its attention mechanism added nothing significant. The neural regression
   helped, but the meditation task failed. Useful, honest negatives.
3. **yoga_impact is the disciplined rebuild** — it keeps MESMI's good DNA (multimodality, a
   continuous calibrated index, explainability, honest evaluation) and adds the rigor MESMI
   lacked: leave-one-subject-out everywhere, probability **calibration**, **permutation tests**,
   **effect sizes**, **negative controls**, **cross-device transfer**, and a controlled
   **leakage-forensics** demonstration.
4. **The payoff is a closed loop:** MESMI *hinted* that deep doesn't out-earn classical;
   yoga_impact's C5 *proved* it under leak-free folds. Hypothesis → honest negative → cleaner,
   evidence-led design. That arc is the most impressive thing to show.

---

# PART 4 — How to run each project

```bash
# Project A — yoga_impact (current)
python -m yoga_impact.run_all
#   or a subset:
python -m yoga_impact.run_all --stages wesad custom riemann report
#   outputs: yoga_impact/outputs/REPORT.md, summary.json, riemann_metrics.json, plots

# Project B — MESMI (earlier)
cd mesmi && python run_all.py     # extract -> train_eval -> explain
#   outputs: mesmi/training/ (metrics.json, plots, SHAP, attention maps)
```

---

# PART 5 — The 30-second summary for the teacher

> I built **two projects**. **MESMI** was my ambitious first attempt — a multimodal, explainable
> biomarker estimator with CEEMDAN, nonlinear dynamics, BiLSTM + cross-attention, and contrastive
> SHAP. I evaluated it honestly, found the heavy model didn't beat simple baselines, and learned
> the real lesson: **rigor matters more than complexity**. So I rebuilt it as **yoga_impact** — a
> single calibrated **Relaxation Index (0–100)**, validated leave-one-subject-out across four
> datasets and three EEG devices, headlined by an honest **AUROC 0.98**, and topped with a novel
> **Riemannian-geometry + optimal-transport** method that transfers across EEG devices and
> measures how much yoga moves you — with significance tests, negative controls, and a controlled
> demonstration of exactly how prior 96–98% numbers were inflated by data leakage.

*Detailed write-ups: `yoga_impact/docs/PROJECT_EXPLAINER_FOR_TEACHER.md` (yoga_impact deep dive),
`yoga_impact/docs/FLAGSHIP_Geometry_of_Calm.md` (flagship method + math), and
`PRIOR_PAPER_COMPARISON_FOR_TEACHER.md` (comparison with the 2023 published paper).*
