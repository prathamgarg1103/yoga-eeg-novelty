This is a synthesis task, not a coding task. I have everything I need: the 10 proposals, three adversarial reviews, and the measured numbers. Let me reconcile them into a decisive plan.

# RECONCILED PLAN: "Quantifying Yoga's Impact"

All three reviewers independently verified the code is genuinely leak-free and converged on the same verdict: **this is ONE strong paper plus rigor scaffolding, not ten papers.** Only two proposals (idx 0, idx 1) contain a genuinely NEW scientific question; everything else is rigor or framing. The accuracy ceiling (~0.79 EEGMAT LOSO) is a property of the data, not a problem to solve.

---

## 1. RANKED TOP 3 THINGS TO DO

**#1 — Write the EMBC spine paper (idx 2 + fold in idx 3, 4, 5).**
- *What:* One paper led by the autonomic Relaxation Index (WESAD calm-vs-stress AUROC 0.98, perm p=0.007, ECE 0.021), with SPD tangent-space + Bures-Wasserstein OT method, cross-device transfer (Neurocom→Emotiv 0.88), and foregrounded nulls; EEG accuracy demoted to supporting evidence.
- *Why it beats Kansal et al.:* Their 98% is a within-subject/window leakage artifact on one device that cannot calibrate, transfer, or report a null. The 0.98 here survives leave-one-subject-out and is calibrated, portable, and paired with a discriminant trait-null control (p=0.90) they structurally lack.
- *Honest expected outcome:* Submittable 4-page EMBC manuscript from already-measured numbers, no new computation. Competitive; acceptance plausible.
- *Effort:* Medium (writing). *Biggest risk:* EMBC 4-page limit forces cuts; the abstract must NOT let the 0.98 autonomic number bleed into an implied EEG-state claim — that conflation is the one thing a reviewer will correctly punish.

**#2 — Subject-cluster BCa bootstrap CIs + TOST on every headline (idx 6, absorbing idx 9's learning curve).**
- *What:* Resample whole held-out subjects/recordings (NOT windows) of the fixed out-of-fold scores → 95% BCa CIs on every headline AUROC; add a pre-declared TOST equivalence margin (ΔAUROC=0.05) to formalize "deep does not beat classical" and "FBCSP +0.04 over band-power"; add the AUROC-vs-n learning curve (12/24/36 subjects) on EEGMAT.
- *Why it beats Kansal et al.:* They reported a single point with no uncertainty — exactly what hid the leak. A CI is the first thing a reviewer demands; the TOST converts the strongest negative result into a rigorous pre-registered claim.
- *Honest expected outcome (estimates):* WESAD 0.98 CI ~[0.92,1.00], lower bound likely >0.90; EEGMAT ~0.787 CI ~[0.70,0.86]; band-power vs FBCSP CIs overlap (the +0.04 is within noise); deep-vs-classical concludes equivalence-or-classical-favored; learning curve probably still mildly rising at n=36 (sample-limited, not method-limited).
- *Effort:* Low. *Biggest risk:* If anyone resamples windows instead of subjects, the CI silently re-leaks the within-subject correlation — the exact prior-paper sin. Cluster bootstrap is non-negotiable.

**#3 — Per-subject Bures-Wasserstein responder profiles (idx 0).**
- *What:* For each WESAD subject, compute their own W2 (meditation vs own stress/baseline) and split it into mean_term (set-point shift) vs cov_term (dispersion change) → a 2-axis responder map; report the distribution (median/IQR), the fraction whose med-vs-stress shift exceeds their med-vs-baseline shift, and the mean-vs-shape Spearman.
- *Why it beats Kansal et al.:* It escapes the accuracy ceiling entirely by changing the question from "does yoga work" (ceiling-bound) to "for whom and how" (responder science) — impossible from pooled-accuracy framing, and structurally leak-free (within-subject, no train/test split).
- *Honest expected outcome:* A right-skewed W2 distribution with visible heterogeneity, mean_term dominating for most subjects, a minority shape-dominant. n=15 → descriptive/case-study ONLY, never a powered responder-typing claim.
- *Effort:* Low. *Biggest risk:* Over-interpreting a Spearman on 15 points (or 4 custom subjects); the mean/shape cutoff must be fixed a priori (ratio>0.5). Note the per-subject magnitudes inherit pooled z-scoring in `_wesad_ot` (~line 769) and must be described as living in a pooled metric.

**Bundle, do not ship separately:** Freeze idx 7 (analysis-plan + Holm) alongside #2 — but frame it as an **analysis-plan freeze, not a blind pre-registration** (results are already measured; overclaiming "pre-registered" invites a credibility hit). idx 1 (trajectory geometry) and idx 8 (levers ablation) are exploratory/supplement only, explicitly under the exploratory side of the freeze.

---

## 2. PAPER SKELETON

**Title:** *The Geometry of Calm: A Calibrated, Leak-Free, Cross-Device Index of Physiological Relaxation*

**Abstract (5 sentences):**
1. Quantifying yoga/relaxation needs a calibrated, transferable physiological index, yet prior EEG work reporting ~98% accuracy (Kansal et al. 2023, LSTM) is inflated by within-subject/window data leakage and offers no calibration, transfer, or negative controls.
2. We build a Composite Relaxation Index on two validated axes — autonomic (WESAD, 15 subjects) and cortical (EEG) — trained strictly under leave-one-subject/recording-out CV with in-fold calibration and permutation tests.
3. The autonomic calm-vs-stress index reaches AUROC 0.98 (permutation p=0.007), is well-calibrated (ECE 0.021, conformal coverage ~0.89), and shows monotone convergent validity (meditation>baseline>amusement>stress, Spearman ~1.0).
4. Our method contribution, "Geometry of Calm," combines augmented multiband channel-covariance (SPD) tangent-space classification with an accuracy-free Bures-Wasserstein optimal-transport effect size that is large and significant on real state shifts (all p=0.002) yet correctly NULL on a between-person trait control (p=0.90), establishing convergent and discriminant validity, and the relaxation manifold transfers across EEG devices (Neurocom→Emotiv AUROC 0.88).
5. We honestly bound the limits — EEG state-decoding has a leak-free ceiling of ~0.79 (EEGMAT, 36 subjects) far below leaky literature, trait decoding is at chance, and deep models do not beat calibrated classical methods — turning the prior work's fatal flaw into an explicit leak-free benchmark protocol.

**Ordered contribution list:**
1. A calibrated autonomic Relaxation Index that survives subject-independent CV (0.98, p=0.007, ECE 0.021) — a usable probabilistic readout, not just a label.
2. A leak-free subject/recording-independent benchmark + explicit leakage forensics exposing the inflation mechanism (the idx 4 table, folded in here).
3. The "Geometry of Calm" method: SPD tangent-space + accuracy-free Bures-Wasserstein OT effect size, scoped honestly as a novel application/validation design (the underlying Riemannian/BW math is standard — say so).
4. Cross-device portability of the relaxation manifold (Neurocom→Emotiv 0.88).
5. Honest negative results as a feature: trait null (discriminant validity, p=0.90), EEG ~0.79 ceiling, deep<classical (formalized via TOST), reported with subject-cluster CIs.

**Best-fit venue:** **IEEE EMBC (full paper).** All three reviewers agree: flagship engineering-in-medicine venue, rewards rigorous physiological signal-processing methodology over leaderboard accuracy, tolerates modest-n + negative results when method/rigor is the contribution. Spin out the benchmark pillar to Sensors (MDPI) ONLY if length forces it — otherwise that is salami-slicing reviewers will penalize.

---

## 3. DO NOT DO (p-hack / leakage traps)

1. **Do NOT bootstrap or split at the window level — ever.** Resample/shuffle whole subjects/recordings. Window-level resampling is the exact statistical sin that inflated the prior paper and would silently re-leak within-subject correlation.
2. **Do NOT quote the best cell of the idx 8 levers ablation as "our result."** Report the whole 3×3×3×3 band; permutation-test ONLY the pre-declared default (k=3, shrink=0.10, θ-α-β, 8 s). Quoting the argmax collapses it into p-hacking.
3. **Do NOT choose the mean-vs-shape responder cutoff (idx 0) to produce a clean split.** Fix ratio>0.5 a priori in the analysis-plan freeze.
4. **Do NOT call it "pre-registered."** Results are already measured this session; call it an analysis-plan freeze. Overclaiming a blind pre-reg is itself a credibility objection.
5. **Do NOT let the trajectory-geometry descriptors (idx 1) roam.** Three new descriptors × 2 datasets × 2 metrics against an already-marginal p=0.011 is a garden of forking paths. Pre-commit one primary descriptor + the existing MWU/Wilcoxon, report all arms including nulls, label it exploratory.
6. **Do NOT let the autonomic 0.98 imply an EEG-state 0.98.** Keep the two axes visibly separate or a reviewer correctly cries conflation.
7. **Do NOT present the leaky-vs-leak-free table as a controlled within-study contrast unless BOTH numbers are computed here on the SAME corpus.** If the "leaky" column is external literature, label it apples-to-oranges. Cleanest: actually run a within-subject split on EEGMAT to show the inflation gap directly.
8. **Do NOT report any custom-EEG finding (~4 subjects) as a population claim.** Case study, everywhere, always.
9. **Do NOT tune until Holm-significant.** Apply Holm over the small frozen confirmatory family; report that p~0.011 (temporal) and the custom case study MAY drop below threshold — that is the honest point.

---

## 4. BRUTALLY HONEST ASSESSMENT

Yes, this is publishable at EMBC as-is — but **only because the contribution is rigor + the right question, not a number.** The headline that carries the paper is the autonomic WESAD 0.98 (p=0.007), and that is genuinely yours: it survives leave-one-subject-out, it is calibrated, and it is paired with a discriminant trait-null (p=0.90) that almost no applied paper in this niche bothers to include. The minimum to make it submittable is small and entirely additive: (a) attach subject-cluster BCa CIs + the TOST to every headline (idx 6) so no number stands as a naked point estimate — its absence is your single most certain rejection; (b) write the EMBC spine with the validity triangle and the leakage-forensics table folded in; (c) add the per-subject responder map (idx 0) as the one fresh result that escapes the ceiling. That is roughly a week of work, mostly writing and cheap re-analysis on existing out-of-fold predictions. **Where chasing accuracy is a trap:** the EEG state-decoding number is irreducibly ~0.79 on honest splits, the +0.04 FBCSP-over-band-power gap is within noise, MoE fusion is not worth it, and deep loses to classical — so any effort spent trying to push the EEG AUROC upward is wasted motion that, if it "succeeds," almost certainly means you reintroduced leakage. The expert-vs-novice trait result is at chance (0.667, n.s.) and must stay a negative control, not be salvaged. Frame the EEG ceiling as a finding about the data, lead with the autonomic index and the methodology, and the paper is honest, defensible, and strictly better than the work it replaces.