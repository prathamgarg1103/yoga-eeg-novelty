# The Geometry of Calm — Flagship Pipeline (R1–R3)

A full explanation of the third, flagship pipeline in `yoga_impact`: what it does, the
mathematics behind it, exactly how the code implements it, how leakage is prevented, the
honest results, and how to run it.

> **One-sentence version.** Instead of describing each physiological window by a list of
> band-power numbers, we describe it by the *geometry* of its channel covariance (a point
> on a curved manifold of positive-definite matrices), and we quantify "how much yoga
> moves you" as the **optimal-transport distance** between the relaxed and active state
> distributions — a single, interpretable, significance-tested effect size that behaves
> identically for brain (EEG) and body (autonomic) data and transfers across EEG devices.

---

## 1. Why this pipeline exists

The project already has two solid pipelines: a leak-free classical rebuild (band power +
calibrated classifier) and a "novelty layer" (C1–C5: connectivity, brain networks,
conformal uncertainty, personalization, deep-vs-classical). Both answer *"can we classify
relaxed vs not?"*. They do **not** answer the question the research is actually about —
*"how much does yoga move a person's physiological state, and is that signal real and
portable?"* — and they do not give a device-independent representation.

This flagship adds exactly that, with three contributions:

| # | Contribution | Question it answers | Headline result |
|---|--------------|---------------------|-----------------|
| **R1** | Riemannian manifold representation | Is a geometric descriptor at least as good as tuned band power? | Ties the strongest classical baseline (0.933 = 0.933) |
| **R2** | Device-portable relaxation signature | Does a relaxation model trained on one EEG device work on another? | **Yes** — Neurocom→Emotiv AUROC **0.88**, Emotiv→Neurocom **0.63** |
| **R3** | Optimal-transport yoga-impact effect size | How *much* does yoga move you, and is it significant? | Large & significant on real state shifts (p=0.002); correctly null on a trait control (p=0.90) |

Every number is produced by the same strictly leak-free machinery the rest of the project
uses (`modeling.group_cv_evaluate`), and **every reference geometry is estimated on
training windows only** (Section 6).

---

## 2. The data — four datasets, three EEG devices

| Dataset | Modality / device | Contrast | Role here |
|---------|-------------------|----------|-----------|
| **custom yoga** | EEG, **Emotiv** F7/F8 @128 Hz | Relaxed vs Concentrated | primary cortical state; transfer **source** |
| **EEGMAT** (PhysioNet eegmat 1.0.0) | EEG, **Neurocom** 10-20 @500→128 Hz | rest vs mental arithmetic | second device; transfer **target** |
| **ds001787** | EEG, **BioSemi** 64-ch (F7/F8 = A7/B10) | expert vs novice meditator | **trait** negative control |
| **WESAD** | autonomic (ECG/EDA/EMG/Resp/Temp) | meditation / baseline / stress | autonomic state, cross-modal generality |

The key addition for this pipeline is **EEGMAT**: rest ≈ *Relaxed* and arithmetic ≈
*Concentrated*, recorded on a completely different device. That is the same-contrast,
different-device pair the other datasets could not provide, and it is what makes the R2
cross-device claim a genuine test rather than a wish.

---

## 3. The representation — an augmented multiband covariance

A raw EEG window from the custom set is just two channels (F7, F8). A 2×2 covariance is
too thin to be interesting. So before taking a covariance we **expand the two channels
into band-filtered "virtual channels"**:

```
for each channel c in {F7, F8}:
    for each band b in {theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz)}:
        virtual_channel[c,b] = bandpass(c, b)
```

That gives `2 channels × 3 bands = 6` virtual channels. The 6×6 covariance `C` of those
channels packs three kinds of information into one matrix:

- **diagonal** → band power in each channel/band (e.g. F7-alpha power),
- **same-band off-diagonal** → spatial coupling (F7-alpha ↔ F8-alpha),
- **cross-band off-diagonal** → cross-frequency coupling (F7-theta ↔ F8-beta).

`C` is symmetric and (after shrinkage) **positive-definite** — i.e. a point on the SPD
manifold, which is what the rest of the method operates on. The tangent map turns each
6×6 `C` into a `6·7/2 = 21`-dimensional feature vector.

**Code:** `riemann._virtual_channels` builds the virtual channels;
`riemann_spd.shrink_cov` computes the covariance with Ledoit-Wolf-style shrinkage
`C ← (1−λ)·C + λ·(tr C / d)·I` (λ = `config.RIEMANN_SHRINKAGE` = 0.10) so the matrix is
always well-conditioned even for short windows.

---

## 4. Why geometry — the SPD manifold in plain terms

Covariance matrices do **not** live in flat Euclidean space; they live on a curved cone of
positive-definite matrices. Averaging them entry-by-entry, or feeding their raw entries to
a linear classifier, ignores that curvature and is known to generalize poorly across
sessions, subjects, and devices. Working with the correct (affine-invariant) geometry is
what gives Riemannian EEG methods their well-documented robustness on small data — exactly
our situation.

Four operations are all we need. Each is implemented in `riemann_spd.py` using only an
eigendecomposition of symmetric matrices (`numpy.linalg.eigh`), so there are no heavy
dependencies.

### 4.1 Matrix functions on SPD inputs
For an SPD matrix `C = V diag(w) Vᵀ`, any scalar function `f` is applied to the
eigenvalues: `f(C) = V diag(f(w)) Vᵀ`. With `f = √, 1/√, log, exp` we get
`sqrtm`, `invsqrtm`, `logm`, `expm`. (Eigenvalues are clipped at 1e-12 for stability.)

### 4.2 Affine-invariant distance (AIRM)
The geodesic distance between two SPD matrices `A`, `B`:

```
d(A,B) = || log( A^{-1/2} · B · A^{-1/2} ) ||_F
       = sqrt( Σ_i log²(λ_i) ),  λ_i = eigenvalues of A^{-1/2} B A^{-1/2}
```

This is invariant to any invertible linear transform of the signals — including a change
of electrode gain. That invariance is precisely why the representation can survive a
change of EEG device. **Code:** `riemann_spd.airm_distance`.

### 4.3 Fréchet (geometric) mean
The "average" covariance on the manifold is the matrix minimizing the sum of squared
geodesic distances. We compute it by the standard Karcher-flow iteration (average the
log-mapped matrices in the tangent space, map back, repeat):

```
G ← G^{1/2} · exp( mean_i log( G^{-1/2} C_i G^{-1/2} ) ) · G^{1/2}
```

started at the arithmetic mean; it converges in a handful of iterations. **Code:**
`riemann_spd.frechet_mean`. This `G` is the per-fold reference (Section 6).

### 4.4 Tangent-space projection (the feature map)
We flatten the curved manifold into a Euclidean space *around the reference `G`* so an
ordinary classifier can be used:

```
T_i = upper_vec( log( G^{-1/2} · C_i · G^{-1/2} ) )
```

`upper_vec` keeps the diagonal and the upper triangle, scaling off-diagonals by √2 so
Euclidean distance in `T`-space locally equals the manifold distance. The resulting 21-d
vectors are what go into logistic regression. **Code:** `riemann_spd.tangent_vectors`.

### 4.5 Re-centering (the domain-adaptation step)
Whitening a whole dataset by its own geometric mean maps that mean to the identity:

```
C_i ← G^{-1/2} · C_i · G^{-1/2}      (now the dataset is centered at I)
```

Do this to two devices separately and they are placed in a common frame, which is the
mechanism we test for cross-device transfer in R2. **Code:** `riemann_spd.recenter`.

---

## 5. Optimal transport — the yoga-impact effect size

Classification tells you *whether* two states differ; it does not tell you *how far apart*
they are. For "how much does yoga move you" we measure the distance between the entire
**distributions** of the relaxed and active states.

We model each state's set of tangent vectors as a Gaussian and use the closed-form
**2-Wasserstein (Bures-Wasserstein) distance** between Gaussians:

```
W₂² = ||μ₁ − μ₂||²            ← MEAN term: how far the center of the state moved
     + Tr( Σ₁ + Σ₂ − 2 (Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2} )   ← SHAPE term: change in spread/shape
```

Reporting the two terms separately is interpretable: e.g. for WESAD meditation-vs-stress
the **shape** term (25.98) is even larger than the **mean** term (22.29) — stress not only
shifts autonomic state, it widens and reshapes its variability. **Code:**
`riemann_spd.bures_wasserstein` (returns `w2`, `mean_term`, `cov_term`) and
`riemann_spd.gaussian_fit` (shrinkage mean/covariance of the tangent vectors).

### Significance — a group-aware permutation test
A W₂ number alone could arise by chance, so `riemann.ot_effect_size` runs a permutation
test that **respects the grouping** so it is not inflated by within-group correlation:

- **unpaired** (each recording/subject is a single state, e.g. custom, ds): shuffle the
  group→state assignment, preserving the class counts;
- **paired** (each subject contributes *both* states, e.g. WESAD, EEGMAT): randomly swap
  the two state labels *within each subject*, which controls the large between-subject
  variance.

`p = (#{permuted W₂² ≥ observed} + 1) / (n_perm + 1)`, with `n_perm = 500`.

### Worked example — one window, end to end (runnable)

Everything above is concrete. The script `riemann_example.py` traces a single real window
through the whole chain and prints every intermediate, so the math can be checked on actual
numbers:

```bash
.venv/Scripts/python.exe -m yoga_impact.riemann_example
```

```
recording: subject=Akshat  state=Relaxed  device=Emotiv  fs=128 Hz  samples=19270
virtual channels (6): ['F7-theta', 'F7-alpha', 'F7-beta', 'F8-theta', 'F8-alpha', 'F8-beta']

[1] window covariance C  (6x6, shrinkage=0.1):
[[ 78.427   3.767   0.08   26.079   2.12    0.067]
 [  3.767  51.292   6.797   2.048  22.758   4.853]
 [  0.08    6.797  71.027   0.065   4.959  47.701]
 [ 26.079   2.048   0.065 103.586   3.873   0.086]
 [  2.12   22.758   4.959   3.873  39.781   5.644]
 [  0.067   4.853  47.701   0.086   5.644  69.468]]
    eigenvalues(C): [ 20.997  23.516  61.627  66.538 119.107 121.797]  -> all > 0 (SPD): True

[2] Frechet (geometric) mean G over all 785 custom windows:
    eigenvalues(G): [ 10.403  11.469  13.202  36.581  50.118 124.776]
    geodesic distance d(C, G) = 2.5503

[3] tangent vector T = upper_vec( log( G^-1/2 C G^-1/2 ) ), dim=21:
    [ 1.432  0.233  0.858  1.505 -0.169  0.542 -0.016  0.003 -0.443 -0.012
      0.003  0.006 -0.01  -0.913 -0.002  0.003 -0.004  0.205 -0.032  0.004 -0.024]
    ||T|| = 2.5503  (equals d(C,G) = 2.5503: the tangent map is an isometry)

[4] Bures-Wasserstein W2 between Relaxed (n=515) and Concentrated (n=270) in tangent space:
    mean term ||mu1-mu2||^2  = 2.698
    shape term Tr(S1+S2-2..) = 0.247
    W2 = sqrt(mean + shape)  = 1.716
```

Three things to read off this trace:

- **[1]** the covariance is genuinely SPD (all six eigenvalues > 0). Its largest off-diagonals
  are the cross-band F7-beta ↔ F8-beta (47.7) and same-channel theta↔alpha couplings — the
  structure band power alone would throw away.
- **[3]** `||T||` equals the geodesic distance `d(C, G)` to four decimals — confirming the
  √2-scaled tangent vectorisation is a true isometry, so Euclidean operations on `T` are
  faithful to the manifold.
- **[4]** the `W₂ = 1.716` computed here is exactly the headline custom Relaxed-vs-Concentrated
  effect size in the §7 results table (W₂ = 1.72), and the mean term (2.70) dominates the
  shape term (0.25) — the relaxed/concentrated difference is mostly a *shift* of the cortical
  state, not a change in its spread.

---

## 6. How leakage is prevented (the part reviewers check first)

The whole project exists because the prior work leaked. The flagship keeps the same
discipline. The only new estimated object is the tangent-space reference `G`, and it is
**re-estimated inside every CV fold on the training covariances only**:

```python
# riemann.riemann_group_cv  (leave-one-group-out)
for tr, te in LeaveOneGroupOut().split(covs, y, groups):
    ref  = frechet_mean(covs[tr])          # <-- fit on TRAIN windows only
    xtr  = tangent_vectors(covs[tr], ref)
    xte  = tangent_vectors(covs[te], ref)  # test projected with the train reference
    model = CalibratedClassifierCV(make_pipeline(LogReg), method="sigmoid", cv=...)
    model.fit(xtr, y[tr]); oof[te] = model.predict_proba(xte)[:, 1]
```

- **Group = recording** (LORO) or **subject** (LOPO/LOSO); no window from a held-out
  group is ever seen in training.
- The geometric mean, the imputer, the scaler, and the probability calibrator are all fit
  on the training fold only.
- Window predictions are aggregated to the **recording** level (the honest unit) before
  scoring — `riemann._recording_level`.
- For R3, the descriptive global mean is only used to define a fixed coordinate system for
  an *effect-size* (not a trained classifier), and significance comes from a
  group/permutation test — so there is nothing to leak.

---

## 7. The three contributions, with results

All results are deterministic (fixed seeds) and reproduced by
`python -m yoga_impact.run_all --stages riemann report`.

### R1 — manifold vs the strongest classical baseline (identical folds)

| Task (CV) | Classical (band+C1) | Riemannian | Gain |
|-----------|---------------------|------------|------|
| custom LORO (recording) | 0.933 | 0.933 | +0.000 |
| custom LOPO (subject) | 0.933 | 0.933 | +0.000 |
| EEGMAT rest-vs-arith (LOSO) | 0.710 | 0.700 | −0.010 |
| ds expert-vs-novice (LOSO, frontal) | — | 0.444 | weak trait |

**Honest reading.** On two frontal channels the manifold **ties** the tuned classical
features — it is a principled, device-portable representation, *not* a higher accuracy
number, and we do not tune it until it "wins". The same Relaxed/Concentrated contrast
reproduces on the Neurocom device (0.71), confirming it is not an Emotiv artifact.

**Does more channels help?** EEGMAT carries a full 19-channel 10-20 montage, so we tested
three representations on it (rest vs arithmetic, LOSO), reporting all three rather than the
best (`riemann._eegmat_multichannel`, figure `outputs/riemann_multichannel.png`):

| Representation (19-ch) | AUROC | vs classical |
|------------------------|-------|--------------|
| classical per-channel band power (323 feats) | 0.747 | — |
| Riemannian **broadband** covariance (190-d) | 0.688 | −0.059 |
| Riemannian **band-specific** connectivity (3×190-d) | **0.753** | **+0.006** |

The broadband covariance *loses* — a single covariance over 1–45 Hz blurs the
frequency-specific structure that drives this spectral task. Giving the manifold the right
representation (a separate covariance per band = band-resolved connectivity) recovers and
*marginally edges* a strong 323-feature classical baseline. Honest conclusion: the manifold
is **competitive on a rich montage but not a decisive accuracy win**, so the flagship's
headline value remains R2 (portability) and R3 (the effect-size framework), not raw AUROC.
We did not tune the representation past these three principled choices.

### R2 — device-portable relaxation signature  ← the headline win

Train on one EEG device, test on another, **same** Relaxed/Concentrated contrast:

| Direction | Naive transfer | + Re-centering |
|-----------|----------------|----------------|
| **Neurocom → Emotiv** | **0.878** | 0.889 |
| **Emotiv → Neurocom** | **0.630** | 0.610 |
| *control:* Emotiv state → BioSemi **trait** | 0.465 | **0.438** |

**Honest reading.** Both cross-device transfers are clearly above chance — a relaxation
model trained on one headset reads the same state on a different headset. The asymmetry is
expected and honest: training on EEGMAT's 36 subjects transfers better than training on
the 19-recording custom set. **Re-centering itself was roughly neutral (±0.02)** here, so
we credit the portability to the relative/geometric representation, *not* to re-centering.
The control confirms discriminant behaviour: a relaxation-**state** model correctly fails
to predict a meditation-**trait** (expert vs novice).

### R3 — optimal-transport yoga-impact effect size

| Contrast | W₂ | mean / shape | p | type |
|----------|----|--------------|----|------|
| custom Relaxed vs Concentrated (Emotiv) | 1.72 | 2.70 / 0.25 | **0.002** | state |
| EEGMAT rest vs arithmetic (Neurocom) | 0.46 | 0.14 / 0.07 | 0.379 | state (underpowered) |
| WESAD meditation vs stress | 6.95 | 22.29 / 25.98 | **0.002** | state |
| WESAD meditation vs baseline | 4.22 | 10.76 / 7.09 | **0.002** | state |
| WESAD baseline vs stress | 5.90 | 13.61 / 21.16 | **0.002** | state |
| ds expert vs novice (**control**) | 0.97 | 0.42 / 0.51 | 0.902 | trait |

**Honest reading.** The effect size is large and significant for genuine within-person
state shifts (Emotiv cortical + all autonomic), and **correctly stays null on the
between-person trait control** — convergent + discriminant validity in one figure. The one
non-significant state contrast, EEGMAT, is a genuinely *small* frontal-only shift (W₂=0.46)
that is underpowered at the distribution level even with the correct paired test; we report
it as-is rather than tuning it to fire. **Validity: 4/5 state significant, 0/1 trait
significant.**

---

## 8. Code map

```
riemann_spd.py    pure-numpy SPD-manifold primitives
  shrink_cov            shrinkage covariance of (channels × samples)
  sqrtm/invsqrtm/
  logm/expm/powm_spd    matrix functions via eigendecomposition
  airm_distance         affine-invariant geodesic distance
  frechet_mean          geometric mean (Karcher flow)
  tangent_vectors       project SPD → 21-d tangent feature vectors (the feature map)
  recenter              whiten a domain by its geometric mean (domain adaptation)
  bures_wasserstein     2-Wasserstein between Gaussians, split into mean/shape terms
  gaussian_fit          shrinkage mean + covariance of tangent vectors

riemann.py        the pipeline
  _band_filter, _virtual_channels, recording_dual, build_eeg_covs
                        build augmented multiband covariances + aligned band features
  riemann_group_cv      leak-free Riemannian LOGO CV (per-fold Fréchet reference)
  riemann_group_cv_multiband   same, over several band covariances concatenated in tangent
  _multichannel_dual / _build_multichannel        full-montage broadband cov + classical feats
  _multichannel_bands / _build_multichannel_bands per-band (theta/alpha/beta) spatial covs
  _eegmat_multichannel  R1 three-way: classical vs broadband vs band-specific (+ multichan OT)
  _recording_level      aggregate window OOF probs → one prediction per recording
  transfer_experiment   R2: cross-device transfer, naive vs re-centered
  ot_effect_size        R3: Bures-Wasserstein W₂ + group/paired permutation p
  _load_custom/_load_ds/_load_eegmat   dataset covariance builders (cached to outputs/)
  _wesad_ot             R3 on the WESAD autonomic feature space (paired within subject)
  run                   orchestrates R1→R2→R3 + validity synthesis, writes metrics+plot
  _plot / _plot_multichannel   3-panel summary + the 19-ch representation comparison

ds_eegmat.py      PhysioNet EEGMAT loader: load_recordings (F7/F8) +
                  load_recordings_full (19-ch 10-20 montage) → Recording objects
riemann_example.py   runnable worked example (one window → C → G → T → W₂); §5 trace
config.py         RIEMANN_* / OT_* parameters (bands, shrinkage, n_perm, WESAD pairs)
```

Config knobs (all in `config.py`):

| Param | Value | Meaning |
|-------|-------|---------|
| `RIEMANN_BANDS` | theta, alpha, beta | virtual-channel bands |
| `RIEMANN_SHRINKAGE` | 0.10 | covariance shrinkage toward scaled I |
| `RIEMANN_CALIBRATE` | "sigmoid" | in-fold probability calibration |
| `OT_N_PERM` | 500 | permutations for the W₂ p-value |
| `OT_GAUSS_SHRINK` | 0.05 | shrinkage for the per-state Gaussian fits |
| `OT_WESAD_PAIRS` | (med,stress),(med,base),(base,stress) | autonomic contrasts scored |

---

## 9. How to run

```bash
# full flagship + report (covariances are cached after the first run)
.venv/Scripts/python.exe -m yoga_impact.run_all --stages riemann report

# just the flagship module
.venv/Scripts/python.exe -m yoga_impact.riemann
```

First run downloads/loads EEGMAT and ds001787 raw signals and builds covariance caches;
subsequent runs read the caches and finish in seconds.

**Outputs (`yoga_impact/outputs/`):**

| File | Contents |
|------|----------|
| `riemann_metrics.json` | every R1/R2/R3 number + validity block (machine-readable) |
| `riemann_summary.png` | 3-panel figure: R1 manifold-vs-classical, R2 transfer, R3 effect size |
| `eegmat_riemann_covs.npz`, `eegmat_riemann_meta.csv`, `eegmat_band.csv` | EEGMAT cache |
| `ds_riemann_covs.npz`, `ds_riemann_meta.csv` | ds001787 cache |
| `REPORT.md` §10 | the flagship written into the project report |

The raw EEGMAT EDFs live in `datasets/eegmat/` (36 subjects × 2 conditions = 72 files).

---

## 10. Honest limitations (what this does *not* claim)

- **R1 is a tie, not a win.** On two frontal channels the manifold matches but does not
  beat tuned band power. The value of the representation is robustness + portability, not a
  higher AUROC, and we did not p-hack it upward.
- **Re-centering was ~neutral** on these data; portability comes from the representation
  itself. We do not claim re-centering as the hero.
- **EEGMAT's OT effect is non-significant** (small frontal-only shift, short task) — reported
  honestly as underpowered, not hidden.
- **ds expert/novice is age-confounded** and weak on 2 channels; it is used only as a
  *trait* negative control, not as a positive result.
- **Custom data is tiny** (19 recordings, ~4 subjects) and is treated as a case study; the
  quantitative weight is carried by the public datasets.

---

## 11. Why this is stronger than the prior LSTM paper

The 2023 paper (*A Novel DL Approach to Quantify Yoga-Induced Cognitive Changes*) reported
high accuracy on a single device with a validation leak (the project's C5 stage shows the
deep edge was a validation artifact). This flagship is better on four concrete axes:

1. **New method** — Riemannian geometry + optimal transport; neither appears in the prior work.
2. **Rigour** — strictly leak-free validation across **four datasets and three EEG devices**,
   with permutation tests, instead of one leaky dataset.
3. **A demonstrated capability the prior pipeline never had** — a **device-portable**
   relaxation signature (R2): a state model trained on one EEG device works on another.
4. **It answers the actual research question** — *how much* does yoga move you — with a
   significance-tested, mean-vs-shape-decomposed effect size that shows convergent (state)
   and discriminant (trait) validity, rather than a single accuracy number.

None of these claims rests on an inflated metric: the tie (R1), the neutral re-centering,
and the underpowered EEGMAT effect are all stated plainly. That honesty is itself part of
why the contribution is defensible.
