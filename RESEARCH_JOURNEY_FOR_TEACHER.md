# My Research Journey — How This Project Evolved
### Pratham Garg · Thapar Institute of Engineering & Technology

> This is the story of *how I got here* — told through the approaches I tried, the faults I found,
> and why each fault made me shift. The numbers are included at every step, because the numbers are
> what exposed each fault and justified each pivot. The point isn't "look how high my accuracy is" —
> it's "watch how the evidence forced my thinking to mature."

**The journey in one line:**

```
Prior work (96.5%, but leaky)  →  MESMI (deep ties/loses to RF)  →  Clean rebuild (120→19 signals, honest 0.98)
        →  Novelty layer (+0.022 / +0.178, deep still loses)  →  Flagship: Geometry of Calm (cross-device 0.88, OT p=0.002)
```

---

## ACT 1 — The starting point: the prior work that "looked perfect"

**What existed.** The published baseline (Kansal et al., 2023) classified yoga EEG states with
signal decomposition + an LSTM:
- Public Muse data, **70/30 split**: GRU **94.6%**, **LSTM 96.5%**.
- Yoga cohort (4 subjects): "real-time" **81.6%**, fine-tuned LSTM **82.3%**.
- An even earlier EMD + PSR + ExtraTrees pipeline reported **97.8%** on 4 participants.

On paper, the problem looked *solved*.

**What I was thinking.** "If it's already 96–98%, where's the project? And why does 98% on 4 noisy
subjects feel too good to be true?" The approach also used **only EEG** — one signal — when stress
and calm are obviously a *whole-body* event.

**The fault I sensed (not yet proven).** A single signal can't capture a system-level state, and a
number that high on tiny data usually hides a validation problem.

**So I shifted to:** building something *richer* — multimodal and explainable. That became MESMI.

---

## ACT 2 — MESMI: the ambitious version (and why it humbled me)

**What I built.** MESMI — *Multimodal Explainable Stress & Meditation Intelligence*. Deliberately big:
**CEEMDAN** decomposition + a **tripartite nonlinear feature set** (phase-space geometry + recurrence
+ entropy) + a **BiLSTM per body signal** (ECG/EDA/EMG/Resp/Temp) fused with **cross-attention** +
**continuous outputs** (0–100 stress score, meditation depth, recovery) + **contrastive SHAP**.

**What I was thinking.** "Stress isn't binary and it isn't one signal. Fuse every body system, let
them attend to each other, output continuous clinical scores — this *has* to beat the prior work."

**Where it fell short — the faults, with numbers (all subject-grouped GroupKFold, honest):**
1. **Complexity didn't pay off.** Binary stress-vs-rest: attention model **0.881** vs **SVM 0.892 /
   RF 0.886**. Four-class: MESMI **0.604** vs **RF 0.676 / SVM 0.662**. The heavy deep + attention
   stack **tied or lost to a plain SVM/RF.**
2. **My headline novelty did nothing.** On the stress-score regression, cross-attention CCC **0.736**
   vs plain concat **0.732** vs mean-pool **0.757** — attention vs concat **Δ = −0.008, p = 0.625**.
   The fancy mechanism was statistically indistinguishable from averaging. (The neural encoder *did*
   help overall — CCC **~0.74–0.76 vs RF 0.605 / ridge 0.512** — but the *attention* wasn't the
   reason.)
3. **Some tasks just failed.** Meditation-depth regression: CCC **≈ 0 / negative** across every
   model. I reported it; I didn't hide it.
4. **The premise had a crack.** No *co-recorded* multimodal yoga subject existed, and the custom EEG
   data was starting to look suspicious.

**What I learned (the turning point).** I was **over-engineering** — mistaking *complexity* for
*contribution*. A result is only worth the validation behind it; **rigor beats sophistication**.

**So I shifted to:** stripping everything back — start simple, validate ruthlessly, and *audit the
data* before trusting any model.

---

## ACT 3 — The clean rebuild: when I finally checked the data

**What I did.** Rebuilt from zero with one rule: *the simplest model I can defend completely* —
classical band-power features + a calibrated classifier, validated **leave-one-subject-out**. Step
one was a **data audit**.

**What I was thinking.** "Before modelling anything, prove the data is real. If the prior 98% was an
illusion, it shows up here."

**What I found — the prior work's fault, now proven with numbers:**
- The custom dataset *looked* like **120 recordings** but collapsed to **19 unique signals**.
- The **same recordings were copied across different 'subject' folders**, one state was
  **mislabelled**, and "neutral" used a **different device** (a hidden confound).
- The smoking gun: under the prior 70/30 split, **identical signals appeared in both train and
  test** — the ~98% was memorising copies, not learning brain states.

**What the rebuild produced — honest numbers:**
- **Autonomic axis (WESAD, leave-one-subject-out):** calm-vs-stress **AUROC 0.980**, permutation
  **p = 0.007**; relaxation score **meditation 94 > baseline 92 ≫ stress 20** (correct physiology).
- **Cortical/EEG axis (custom, leave-one-recording-out):** **AUROC 0.933**, acc 0.84–0.90,
  permutation **p = 0.005** — honest, and still well above chance.

**The new fault I now faced.** The honest EEG signal was *tiny and weak*, and a plain (if correct)
baseline didn't yet feel *novel*. Honesty had cost me my headline number.

**So I shifted to:** adding novelty that survives honest validation — each idea tested leak-free.

---

## ACT 4 — The novelty layer: adding ideas that survive scrutiny

**What I added (five leak-free extensions), with numbers:**
- **C1 — connectivity features:** custom AUROC **0.911 → 0.933 (+0.022)**. Real, modest gain.
- **C2 — 64-channel brain network:** expert-vs-novice **AUROC 0.667** (beats 0.52 frontal) but
  **permutation p = 0.138** (honestly non-significant); one sharp contrast — beta path length
  **p = 0.002**. Age confound flagged.
- **C3 — calibrated uncertainty:** **ECE 0.021**, conformal coverage **0.89 ≈ 0.90**; convergent
  validity **Spearman ρ = 1.000**.
- **C4 — honest personalization:** **0.797 → 0.975 AUROC (+0.178)**, helped **13/15**, Wilcoxon
  **p = 0.002** — the prior paper's "adaptation" benefit, *without* its leak.
- **C5 — deep vs classical:** WESAD classical **0.980** vs deep 0.896; custom classical **0.933** vs
  MLP 0.900, **LSTM 0.811**. **Deep still loses.**

**What I was thinking.** "If I claim novelty, every piece must earn its place under the same strict
validation — even when my own idea loses."

**Where it fell short — and what it confirmed:**
- Gains were **incremental** (e.g. +0.022).
- C5 **closed the loop on MESMI**: under honest folds, deep learning *still* doesn't beat calibrated
  classical. MESMI hinted it; now it was proven.
- I hit the wall directly. A controlled experiment — **same model, only the validation scheme
  changes** — gave honest **0.718**, leaky 70/30 **0.984**, within-subject **1.000**. The **+0.27
  gap** *is* the prior paper's inflation. Accuracy was **ceiling-bound**: any bigger number meant
  re-introducing the very leak I'd removed.

**The realization.** I was still playing the prior paper's game — *"who gets the bigger number"* —
the exact game that manufactures irreproducible results. I needed a **different kind of
contribution**: a new *method* and a new *question*.

**So I shifted to:** the flagship — change the representation, and change the question.

---

## ACT 5 — The flagship: *The Geometry of Calm*

**What I built.** A genuinely new approach for this problem:
- Represent each moment as a point on a **curved (Riemannian) manifold** — a *relative, geometric*
  description that can be **device-portable**.
- Measure yoga's impact as a **distribution shift** via **optimal transport** (Bures–Wasserstein) —
  *how much*, and in what way, the calm state moves from the stressed state — with significance tests
  and negative controls.
- Add a **second EEG device** (PhysioNet EEGMAT) to actually test cross-device transfer.

**What I was thinking.** "Stop asking 'can I score high?' Ask 'what is real, and does it generalise?'
A representation that works across devices, and an effect-size that says *how much* yoga moves you, is
worth more than another decimal of accuracy."

**Where it landed — with numbers:**
- **Device-portable signature:** train on one EEG device, test on another — **Neurocom→Emotiv AUROC
  0.878**, Emotiv→Neurocom 0.630 (both well above chance). Negative control (state→trait transfer)
  **0.438 — correctly fails.** The prior single-device pipeline could never do this.
- **Effect-size with validity (optimal transport W₂, group-permutation p):** WESAD meditation-vs-
  stress **W₂ 6.95**, custom Relaxed-vs-Concentrated **1.72** (all **p = 0.002**); trait control
  expert-vs-novice **p = 0.902 (correctly null)**. Result: **4/5 state contrasts significant, 0/1
  trait** = convergent + discriminant validity.
- **Honest accuracy ceiling, stated plainly:** FBCSP on EEGMAT **0.787**; the manifold ties the best
  classical (**0.933 = 0.933**; 19-ch **0.753 vs 0.747**) — competitive, not an inflated win.
- **A bonus finding:** focused states are *more* temporally stable than rest (**p = 0.011**) — a
  genuinely new, accuracy-free readout.

**This wasn't luck — it was the destination.** Every earlier fault pushed me here: the prior work's
fragile 96.5% → MESMI's deep model tying RF → the data's 120→19 contamination → the 0.72-vs-0.98
leakage gap → finally, a method whose contribution is **portability + validity**, not a number.

---

## What this journey shows

| Stage | What I tried | The fault (with numbers) | Why I shifted |
|---|---|---|---|
| 1 · Prior work | EEG-only LSTM, 96.5% | 70/30 split; fragile on 4 subjects | Build something richer |
| 2 · MESMI | Multimodal + cross-attention + XAI | Deep **0.604** < RF **0.676**; attention Δccc **−0.008 (p=0.625)** | Rigor over complexity |
| 3 · Clean rebuild | Simple model + data audit | Data was **120→19** copies; honest axis **0.98** | Rebuild honestly |
| 4 · Novelty layer | 5 leak-free extensions | Gains small (**+0.022**); leak gap **0.718 vs 0.984** | Need a new *method*, not a bigger number |
| 5 · Flagship | Riemannian geometry + optimal transport | Cross-device **0.88**; OT **p=0.002**; trait null **p=0.90** | Contribution = portability + validity |

**The real result of this project is the journey, backed by the numbers at every turn:** I learned
to distrust a number that's too good (96.5% → 0.718 honest), to value validation over architecture
(deep 0.604 < RF 0.676), to audit my data before trusting my model (120 → 19), and to redefine the
question when the old one stopped being worth answering (cross-device 0.88, OT p=0.002). That
progression — *ambition → humility → rigor → genuine novelty* — is what I'm presenting.

---

*Companion documents: `ALL_PIPELINES_FOR_TEACHER.md` (every pipeline in detail) ·
`PRIOR_PAPER_COMPARISON_FOR_TEACHER.md` (vs the published paper) ·
`yoga_impact/docs/PROJECT_EXPLAINER_FOR_TEACHER.md` (the technical deep dive).*
